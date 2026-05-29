from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from robotdataset.oxe.utils import infer_kind


# Column names used by different robotics HF dataset formats to identify episodes
_EPISODE_ID_COLUMNS = ("episode_index", "episode_id", "traj_id", "trajectory_id")

# Column names used to identify tasks within a dataset
_TASK_ID_COLUMNS = ("task_index", "task_id", "task")

# Columns that are dataset bookkeeping metadata, not step observations
_META_COLUMNS = frozenset({
    "episode_index", "episode_id", "frame_index", "timestamp",
    "task_index", "task_id", "index", "traj_id", "trajectory_id",
})


def _require_datasets() -> Any:
    try:
        import datasets
        return datasets
    except ImportError:
        raise RuntimeError(
            "Table30v2Dataset requires the 'datasets' package. "
            "Install it with: pip install 'robotdataset[hf]'"
        )


def _episode_id_column(features: Any) -> str:
    """Find the column name used to identify episodes in this dataset."""
    for col in _EPISODE_ID_COLUMNS:
        if col in features:
            return col
    raise ValueError(
        f"No episode ID column found. Expected one of: {_EPISODE_ID_COLUMNS}. "
        f"Got: {list(features.keys())}"
    )


def _maybe_episode_id_column(features: Any) -> Optional[str]:
    """Find episode ID column, returning None when absent."""
    for col in _EPISODE_ID_COLUMNS:
        if col in features:
            return col
    return None


def _task_id_column(features: Any) -> Optional[str]:
    """Return the column name used for task IDs, or None if not present."""
    for col in _TASK_ID_COLUMNS:
        if col in features:
            return col
    return None


class _FilteredDataset:
    """Lightweight filtered view of an HF dataset.

    Supports iteration and column-list access (``dataset[column_name]``),
    which is all that ``get_episode_ids`` and ``build_missing_episodes``
    require.  Compatible with both real HF Dataset objects and test fakes.
    """

    def __init__(self, base: Any, keep_task_ids: frozenset) -> None:
        col = _task_id_column(base.features)
        self._rows: List[Dict[str, Any]] = [
            dict(row) for row in base if row[col] in keep_task_ids
        ]
        self.features = base.features

    def __iter__(self):
        return iter(self._rows)

    def __getitem__(self, key: Any) -> Any:
        if isinstance(key, str):
            return [r[key] for r in self._rows]
        return self._rows[key]

    def __len__(self) -> int:
        return len(self._rows)


def filter_by_tasks(hf_dataset: Any, tasks: List[int]) -> "_FilteredDataset":
    """Return a view of ``hf_dataset`` restricted to rows from the given task IDs.

    The returned object supports the same iteration and column-access interface
    as the input dataset, so it can be used as a drop-in replacement everywhere
    ``hf_dataset`` is consumed.

    Args:
        hf_dataset: A loaded HuggingFace Dataset (or compatible object).
        tasks: Task IDs to keep.  Rows whose task column value is not in this
            list are excluded.

    Raises:
        ValueError: If no task ID column is found in ``hf_dataset.features``.
    """
    col = _task_id_column(hf_dataset.features)
    if col is None:
        raise ValueError(
            f"Task filtering requested but no task ID column found in this dataset. "
            f"Expected one of: {_TASK_ID_COLUMNS}. "
            f"Got: {list(hf_dataset.features.keys())}"
        )
    return _FilteredDataset(hf_dataset, frozenset(tasks))


def load_hf_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[Path] = None,
    config_name: Optional[str] = None,
) -> Any:
    """Load a HuggingFace dataset for the given split.

    The datasets library handles download and caching internally.
    ``cache_dir`` is passed as the HF cache root so dataset files are
    co-located with the robotdataset TED memmaps.
    """
    datasets = _require_datasets()
    kwargs: Dict[str, Any] = {}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir / "hf_cache")
    if config_name is not None:
        return datasets.load_dataset(dataset_name, config_name, split=split, **kwargs)
    return datasets.load_dataset(dataset_name, split=split, **kwargs)


def get_episode_ids(hf_dataset: Any) -> List[int]:
    """Return sorted unique episode IDs present in the dataset.

    If no explicit episode-ID column exists, each row is treated as one episode
    and IDs are inferred from row indices.
    """
    col = _maybe_episode_id_column(hf_dataset.features)
    if col is not None:
        return sorted(set(hf_dataset[col]))
    return list(range(len(hf_dataset)))


def _convert_leaf(val: Any) -> Any:
    """Convert a single HF dataset leaf value to a torch.Tensor or str."""
    # PIL Image → uint8 numpy → torch
    try:
        from PIL.Image import Image as PILImage
        if isinstance(val, PILImage):
            return torch.from_numpy(np.ascontiguousarray(np.array(val, dtype=np.uint8)))
    except ImportError:
        pass

    if isinstance(val, torch.Tensor):
        return val

    if isinstance(val, np.ndarray):
        if val.dtype.kind in {"S", "U", "O"}:
            return val.tolist() if val.ndim > 0 else str(val.flat[0])
        return torch.from_numpy(np.ascontiguousarray(val))

    if isinstance(val, np.generic):
        if val.dtype.kind in {"S", "U", "O"}:
            item = val.item()
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)
        return torch.tensor(val.item())

    if isinstance(val, list) and val and isinstance(val[0], (int, float)):
        try:
            return torch.tensor(val, dtype=torch.float32)
        except Exception:
            pass

    # bool must come before int since bool is a subclass of int
    if isinstance(val, bool):
        return torch.tensor(val)
    if isinstance(val, (int, float)):
        return torch.tensor(val)

    if isinstance(val, (bytes, bytearray)):
        try:
            return val.decode("utf-8")
        except Exception:
            return val

    return val


def _convert_tree(val: Any) -> Any:
    """Recursively convert nested HF values to torch-friendly objects."""
    if isinstance(val, dict):
        return {k: _convert_tree(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_convert_tree(v) for v in val]
    if isinstance(val, tuple):
        return tuple(_convert_tree(v) for v in val)
    return _convert_leaf(val)


def _to_nested(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Reconstruct a nested dict from dot-separated flat keys.

    E.g. {"observation.image": v1, "observation.state": v2}
    → {"observation": {"image": v1, "state": v2}}
    """
    nested: Dict[str, Any] = {}
    for key, val in flat.items():
        parts = key.split(".")
        d = nested
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        d[parts[-1]] = val
    return nested


def hf_episode_to_oxe_format(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Convert a list of HF dataset rows to an OXE-style episode dict.

    The result is compatible with ``episode_to_ted_steps()``:
        {"steps": [{observation: {...}, action: T, reward: T, is_last: T, ...}, ...]}

    Handles:
    - Dot-separated column names (LeRobot style) → nested dicts
    - ``next.*`` sub-keys → top-level ``reward`` / ``is_last`` / ``is_terminal``
    - PIL images → uint8 HWC tensors
    - Numpy arrays / Python lists → torch tensors
    """
    steps = []
    for row in rows:
        converted = {k: _convert_tree(v) for k, v in row.items()}
        nested = _to_nested(converted)

        # Lift LeRobot-style "next" sub-dict to OXE top-level field names
        next_dict = nested.pop("next", {})
        if "reward" in next_dict:
            nested.setdefault("reward", next_dict["reward"])
        if "done" in next_dict:
            nested.setdefault("is_last", next_dict["done"])
        if "terminated" in next_dict:
            nested.setdefault("is_terminal", next_dict["terminated"])

        # Remove bookkeeping metadata columns
        for col in _META_COLUMNS:
            nested.pop(col, None)

        steps.append(nested)
    return {"steps": steps}


def hf_row_to_oxe_episode(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one HF row representing an entire episode to OXE-like format."""
    converted = _convert_tree(dict(row))
    if "steps" in converted and isinstance(converted["steps"], list):
        return {"steps": converted["steps"]}
    return {"steps": [converted]}


def infer_modalities_from_storage(td: Any) -> Dict[str, Dict[str, Any]]:
    """Return ``{path: spec_dict}`` by inspecting a loaded TED TensorDict.

    Infers modalities directly from the built storage rather than the source
    dataset schema, so it works regardless of the upstream data format.
    """
    modalities: Dict[str, Dict[str, Any]] = {}

    def _visit(t: Any, prefix: str = "") -> None:
        for key in t.keys():
            path = f"{prefix}/{key}" if prefix else str(key)
            val = t.get(key)
            if hasattr(val, "keys"):
                _visit(val, path)
            elif isinstance(val, torch.Tensor):
                dtype_str = str(val.dtype).replace("torch.", "")
                shape = tuple(int(d) for d in val.shape[1:])
                modalities[path] = {
                    "path": path,
                    "kind": infer_kind(path),
                    "dtype": dtype_str,
                    "shape": shape,
                    "source": "storage",
                }

    _visit(td)
    return modalities
