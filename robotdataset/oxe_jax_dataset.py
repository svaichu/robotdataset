from __future__ import annotations

import hashlib
import json
import pickle
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from robotdataset._common import _get_cache_dir
from robotdataset.oxe.bucket import discover_dataset_versions, discover_datasets_from_bucket

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:  # pragma: no cover - optional dependency
    _tqdm_cls = None  # type: ignore[assignment]

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

try:
    import tensorflow_datasets as tfds
except ImportError:  # pragma: no cover - optional dependency
    tfds = None

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - optional dependency
    jnp = None


OXE_BUCKET_URL = "gs://gresearch/robotics"

_DATASET_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_TF_TENSOR_TYPES = (tf.Tensor,) if tf is not None else tuple()

_EPISODE_SENTINEL = "_steps.json"
_COMBINED_SENTINEL = "_complete.json"
_COMBINED_META = "combined_meta.json"


@dataclass(frozen=True)
class ModalitySpec:
    path: str
    kind: str
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    source: str


def _progress(
    iterable: Iterable,
    *,
    desc: str = "",
    unit: str = "it",
    total: Optional[int] = None,
    leave: bool = True,
) -> Iterable:
    if _tqdm_cls is not None:
        return _tqdm_cls(iterable, desc=desc, unit=unit, total=total, leave=leave, dynamic_ncols=True)
    return iterable


def _require_tf_stack() -> None:
    if tf is None or tfds is None:
        raise RuntimeError(
            "OXEJAXDataset requires tensorflow and tensorflow-datasets. "
            "Install those packages to load OXE datasets."
        )


def _require_jax() -> None:
    if jnp is None:
        raise RuntimeError(
            "OXEJAXDataset requires jax. Install jax to use the JAX batch path."
        )


def normalize_version_key(version: str) -> Tuple[int, ...]:
    parts = [int(chunk) for chunk in re.split(r"[^0-9]+", version) if chunk]
    return tuple(parts) if parts else (0,)


def latest_version(versions: Sequence[str]) -> str:
    return sorted(versions, key=normalize_version_key)[-1]


def infer_kind(path: str, value: Any = None) -> str:
    lowered = path.lower()
    if any(token in lowered for token in ("image", "camera", "rgb", "video", "frame", "color", "fisheye", "depth")):
        return "image"
    if any(token in lowered for token in ("language", "instruction", "text", "caption", "prompt")):
        return "text"
    if any(token in lowered for token in ("action", "policy", "torque", "velocity")):
        return "action"
    if any(token in lowered for token in ("observation", "state", "proprio", "joint", "pose", "ee")):
        return "state"
    if isinstance(value, (str, bytes, bytearray)):
        return "text"
    if isinstance(value, np.ndarray) and value.ndim >= 3:
        return "image"
    return "generic"


def _is_numeric_array(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.dtype.kind not in {"O", "U", "S"}
    if isinstance(value, np.generic):
        return value.dtype.kind not in {"O", "U", "S"}
    return isinstance(value, (bool, int, float))


def _tf_to_numpy(value: Any, tf_tensor_types: Tuple[type, ...] = _TF_TENSOR_TYPES) -> Any:
    if tf_tensor_types and isinstance(value, tf_tensor_types):
        value = value.numpy()

    if isinstance(value, Mapping):
        return {k: _tf_to_numpy(v, tf_tensor_types) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(_tf_to_numpy(v, tf_tensor_types) for v in value)
    if isinstance(value, list):
        return [_tf_to_numpy(v, tf_tensor_types) for v in value]

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return bytes(value)

    if isinstance(value, str):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U"}:
            return value.astype(str).tolist() if value.ndim > 0 else str(value.item())
        if value.dtype.kind == "O":
            return value.tolist()
        return np.ascontiguousarray(value)

    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.bytes_):
            item = value.item()
            return item.decode("utf-8") if isinstance(item, bytes) else str(item)
        if np.issubdtype(value.dtype, np.str_):
            return str(value.item())
        return value.item()

    if isinstance(value, (bool, int, float)):
        return value

    return value


def _flatten_leaves(tree: Any, prefix: Tuple[str, ...] = ()) -> Dict[Tuple[str, ...], Any]:
    if isinstance(tree, Mapping):
        flat: Dict[Tuple[str, ...], Any] = {}
        for key, val in tree.items():
            flat.update(_flatten_leaves(val, prefix + (str(key),)))
        return flat
    return {prefix: tree}


def _unflatten_leaves(flat: Dict[Tuple[str, ...], Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for path, value in flat.items():
        cur: Dict[str, Any] = out
        for key in path[:-1]:
            cur = cur.setdefault(key, {})
        cur[path[-1]] = value
    return out


def _path_to_stem(path: Tuple[str, ...]) -> str:
    return "__".join(path)


def _flatten_features(tree: Any, prefix: str = "") -> Dict[str, ModalitySpec]:
    flattened: Dict[str, ModalitySpec] = {}

    if isinstance(tree, Mapping):
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(_flatten_features(value, child_prefix))
        return flattened

    if hasattr(tree, "items") and callable(tree.items):
        try:
            for key, value in tree.items():
                if key == "steps" and hasattr(value, "feature"):
                    flattened.update(_flatten_features(value.feature, prefix))
                else:
                    child_prefix = f"{prefix}/{key}" if prefix else str(key)
                    flattened.update(_flatten_features(value, child_prefix))
            return flattened
        except Exception:
            pass

    if hasattr(tree, "feature"):
        return _flatten_features(tree.feature, prefix)

    shape: Optional[Tuple[int, ...]] = None
    dtype_str: Optional[str] = None

    if hasattr(tree, "shape"):
        try:
            raw = tree.shape
            if hasattr(raw, "as_list"):
                raw = raw.as_list()
            shape = tuple(-1 if dim is None else int(dim) for dim in raw) if raw else ()
        except Exception:
            pass

    if hasattr(tree, "dtype"):
        try:
            dt = tree.dtype
            dtype_str = dt.name if hasattr(dt, "name") else str(dt)
        except Exception:
            pass

    if dtype_str is None and isinstance(tree, np.ndarray):
        dtype_str = str(tree.dtype)
        shape = tuple(int(dim) for dim in tree.shape)

    path = prefix or "value"
    kind = infer_kind(path, tree)
    flattened[path] = ModalitySpec(
        path=path,
        kind=kind,
        dtype=dtype_str,
        shape=shape,
        source="metadata",
    )
    return flattened


def _episode_to_numpy_steps(
    episode: Any,
    episode_idx: int,
    tf_tensor_types: Tuple[type, ...] = _TF_TENSOR_TYPES,
) -> List[Dict[str, Any]]:
    if "steps" in episode:
        raw_steps = list(episode["steps"])
    else:
        raw_steps = [episode]

    steps = [_tf_to_numpy(step, tf_tensor_types) for step in raw_steps]
    total = len(steps)

    ted_steps: List[Dict[str, Any]] = []
    for step_idx, step in enumerate(steps):
        is_last = bool(step.get("is_last", step_idx == total - 1))
        is_terminal = bool(step.get("is_terminal", is_last))
        reward = np.asarray(float(step.get("reward", 0.0)), dtype=np.float32)

        next_step = steps[step_idx + 1] if step_idx < total - 1 else step
        obs = step.get("observation", {})
        next_obs = next_step.get("observation", {})

        item: Dict[str, Any] = {
            "observation": obs if isinstance(obs, Mapping) else {"obs": obs},
            "action": step.get("action", np.zeros(1, dtype=np.float32)),
            "done": np.asarray(is_last, dtype=np.bool_),
            "terminated": np.asarray(is_terminal, dtype=np.bool_),
            "next": {
                "observation": next_obs if isinstance(next_obs, Mapping) else {"obs": next_obs},
                "reward": reward,
                "done": np.asarray(is_last, dtype=np.bool_),
                "terminated": np.asarray(is_terminal, dtype=np.bool_),
            },
            "collector": {
                "episode_id": np.asarray(episode_idx, dtype=np.int64),
            },
        }

        for key, val in step.items():
            if key in {"observation", "action", "reward", "is_last", "is_terminal", "is_first", "discount", "steps"}:
                continue
            if isinstance(val, Mapping):
                continue
            item[key] = val

        ted_steps.append(item)

    return ted_steps


def _stack_leaf_values(values: List[Any]) -> Tuple[str, Any]:
    if all(_is_numeric_array(v) for v in values):
        arrs = [np.asarray(v) for v in values]
        return "numeric", np.stack(arrs, axis=0)
    return "text", list(values)


def is_episode_cached(episode_dir: Path) -> bool:
    return (episode_dir / _EPISODE_SENTINEL).exists()


def _build_one_episode(
    episode: Any,
    global_idx: int,
    episode_dir: Path,
    tf_tensor_types: Tuple[type, ...],
) -> int:
    steps = _episode_to_numpy_steps(episode, global_idx, tf_tensor_types)
    if not steps:
        return 0

    episode_dir.mkdir(parents=True, exist_ok=True)
    data_dir = episode_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    flattened_per_step = [_flatten_leaves(step) for step in steps]
    all_paths: List[Tuple[str, ...]] = sorted(flattened_per_step[0].keys())

    leaves_meta: List[Dict[str, Any]] = []
    for path in all_paths:
        values = [flat[path] for flat in flattened_per_step]
        leaf_kind, payload = _stack_leaf_values(values)
        stem = _path_to_stem(path)
        if leaf_kind == "numeric":
            file_name = f"{stem}.npy"
            np.save(data_dir / file_name, payload)
            leaves_meta.append(
                {
                    "path": list(path),
                    "storage_kind": "numeric",
                    "file": file_name,
                    "dtype": str(payload.dtype),
                    "shape": [int(dim) for dim in payload.shape],
                }
            )
        else:
            file_name = f"{stem}.pkl"
            with open(data_dir / file_name, "wb") as handle:
                pickle.dump(payload, handle)
            leaves_meta.append(
                {
                    "path": list(path),
                    "storage_kind": "text",
                    "file": file_name,
                    "dtype": None,
                    "shape": [len(payload)],
                }
            )

    n_steps = len(steps)
    meta = {"n_steps": n_steps, "leaves": leaves_meta}
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps(meta))
    return n_steps


def build_missing_episodes(
    builder: Any,
    split: str,
    episodes_dir: Path,
    missing: List[int],
    tf_tensor_types: Tuple[type, ...],
) -> None:
    if not missing:
        return

    missing_set = set(missing)
    max_idx = max(missing_set)
    dataset = builder.as_dataset(split=split, shuffle_files=False)

    pbar = (
        _tqdm_cls(total=len(missing_set), desc="Converting episodes to NumPy", unit="ep", dynamic_ncols=True)
        if _tqdm_cls is not None
        else None
    )

    for global_idx, episode in enumerate(dataset):
        if global_idx > max_idx:
            break
        if global_idx not in missing_set:
            continue
        episode_dir = episodes_dir / str(global_idx)
        if not is_episode_cached(episode_dir):
            _build_one_episode(episode, global_idx, episode_dir, tf_tensor_types)
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()


def combined_dir_key(selected: List[int]) -> str:
    payload = ",".join(str(i) for i in sorted(selected))
    return hashlib.md5(payload.encode()).hexdigest()[:16]


def is_combined_complete(combined_dir: Path) -> bool:
    return (combined_dir / _COMBINED_SENTINEL).exists()


def _read_episode_meta(episode_dir: Path) -> Dict[str, Any]:
    return json.loads((episode_dir / _EPISODE_SENTINEL).read_text())


def build_combined_storage(
    selected: List[int],
    episodes_dir: Path,
    combined_dir: Path,
) -> None:
    if not selected:
        raise ValueError("selected episode list cannot be empty")

    episode_meta = {
        idx: _read_episode_meta(episodes_dir / str(idx))
        for idx in selected
    }
    n_steps_map = {idx: int(meta["n_steps"]) for idx, meta in episode_meta.items()}
    total_steps = sum(n_steps_map.values())

    data_dir = combined_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    first_leaves = episode_meta[selected[0]]["leaves"]
    numeric_mmaps: Dict[Tuple[str, ...], np.memmap] = {}
    text_accumulators: Dict[Tuple[str, ...], List[Any]] = {}
    combined_leaves: List[Dict[str, Any]] = []

    for leaf in first_leaves:
        path = tuple(leaf["path"])
        file_name = leaf["file"]
        storage_kind = leaf["storage_kind"]

        if storage_kind == "numeric":
            dtype = np.dtype(leaf["dtype"])
            shape = tuple(int(dim) for dim in leaf["shape"])
            if not shape:
                raise RuntimeError(f"Invalid numeric shape for {'/'.join(path)}")
            combined_shape = (total_steps,) + shape[1:]
            mmap = np.lib.format.open_memmap(
                data_dir / file_name,
                mode="w+",
                dtype=dtype,
                shape=combined_shape,
            )
            numeric_mmaps[path] = mmap
            combined_leaves.append(
                {
                    "path": list(path),
                    "storage_kind": "numeric",
                    "file": file_name,
                    "dtype": str(dtype),
                    "shape": [int(dim) for dim in combined_shape],
                }
            )
        else:
            text_accumulators[path] = [None] * total_steps
            combined_leaves.append(
                {
                    "path": list(path),
                    "storage_kind": "text",
                    "file": file_name,
                    "dtype": None,
                    "shape": [total_steps],
                }
            )

    pbar = (
        _tqdm_cls(total=len(selected), desc="Assembling JAX training buffer", unit="ep", dynamic_ncols=True)
        if _tqdm_cls is not None
        else None
    )

    offset = 0
    for idx in selected:
        episode_dir = episodes_dir / str(idx)
        n_steps = n_steps_map[idx]
        meta = episode_meta[idx]
        by_path = {tuple(leaf["path"]): leaf for leaf in meta["leaves"]}

        for path, mmap in numeric_mmaps.items():
            leaf = by_path[path]
            ep_arr = np.load(episode_dir / "data" / leaf["file"], mmap_mode="r")
            mmap[offset: offset + n_steps] = ep_arr

        for path, values in text_accumulators.items():
            leaf = by_path[path]
            with open(episode_dir / "data" / leaf["file"], "rb") as handle:
                ep_values = pickle.load(handle)
            values[offset: offset + n_steps] = ep_values

        offset += n_steps
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    for mmap in numeric_mmaps.values():
        mmap.flush()

    for path, values in text_accumulators.items():
        file_name = f"{_path_to_stem(path)}.pkl"
        with open(data_dir / file_name, "wb") as handle:
            pickle.dump(values, handle)

    combined_meta = {
        "n_steps": total_steps,
        "episodes": sorted(selected),
        "leaves": combined_leaves,
    }
    (combined_dir / _COMBINED_META).write_text(json.dumps(combined_meta))
    (combined_dir / _COMBINED_SENTINEL).write_text(
        json.dumps({"n_steps": total_steps, "episodes": sorted(selected)})
    )


class CombinedNumpyStore:
    def __init__(self, combined_dir: Path) -> None:
        meta = json.loads((combined_dir / _COMBINED_META).read_text())
        self.n_steps = int(meta["n_steps"])
        self.episodes = list(meta.get("episodes", []))
        self.leaves: Dict[Tuple[str, ...], Any] = {}

        data_dir = combined_dir / "data"
        for leaf in meta["leaves"]:
            path = tuple(leaf["path"])
            kind = leaf["storage_kind"]
            if kind == "numeric":
                self.leaves[path] = np.load(data_dir / leaf["file"], mmap_mode="r")
            else:
                with open(data_dir / leaf["file"], "rb") as handle:
                    self.leaves[path] = pickle.load(handle)

    def __len__(self) -> int:
        return self.n_steps


class JAXTemporalSampler:
    def __init__(
        self,
        delta_timestamps: Dict[str, List[float]],
        control_frequency: float = 10.0,
        image_keys: Sequence[Tuple[str, ...]] = (),
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.control_frequency = control_frequency
        self.image_keys = set(image_keys)
        self._offsets: Dict[Tuple[str, ...], List[int]] = {
            tuple(key.split("/")): [round(delta * control_frequency) for delta in deltas]
            for key, deltas in delta_timestamps.items()
        }
        self._next_offsets: Dict[Tuple[str, ...], List[int]] = {
            key: sorted(-off for off in offsets)
            for key, offsets in self._offsets.items()
        }

    @staticmethod
    def build_episode_index(episode_ids: np.ndarray) -> Tuple[Dict[int, int], Dict[int, int]]:
        starts: Dict[int, int] = {}
        lengths: Dict[int, int] = {}
        if len(episode_ids) == 0:
            return starts, lengths

        cur_id = int(episode_ids[0])
        cur_start = 0
        for idx in range(1, len(episode_ids) + 1):
            boundary = idx == len(episode_ids) or int(episode_ids[idx]) != cur_id
            if boundary:
                starts[cur_id] = cur_start
                lengths[cur_id] = idx - cur_start
                if idx < len(episode_ids):
                    cur_id = int(episode_ids[idx])
                    cur_start = idx
        return starts, lengths

    def _build_flat_indices(
        self,
        anchor_indices: np.ndarray,
        offsets_map: Dict[Tuple[str, ...], List[int]],
        episode_ids: np.ndarray,
        episode_starts: Dict[int, int],
        episode_lengths: Dict[int, int],
    ) -> Dict[Tuple[str, ...], np.ndarray]:
        key_flat: Dict[Tuple[str, ...], np.ndarray] = {}
        batch_size = int(anchor_indices.shape[0])
        for key_tuple, offsets in offsets_map.items():
            idx = np.zeros((batch_size, len(offsets)), dtype=np.int64)
            for b, anchor in enumerate(anchor_indices.tolist()):
                ep_id = int(episode_ids[anchor])
                ep_start = episode_starts[ep_id]
                ep_len = episode_lengths[ep_id]
                step = anchor - ep_start
                for t_idx, off in enumerate(offsets):
                    clamped = max(0, min(ep_len - 1, step + off))
                    idx[b, t_idx] = ep_start + clamped
            key_flat[key_tuple] = idx
        return key_flat

    def _apply_image_permutation(self, data: np.ndarray, key: Tuple[str, ...]) -> np.ndarray:
        if key in self.image_keys and data.ndim >= 4:
            perm = (0, 1, data.ndim - 1) + tuple(range(2, data.ndim - 1))
            return np.transpose(data, perm)
        return data

    def sample(
        self,
        store: CombinedNumpyStore,
        episode_starts: Dict[int, int],
        episode_lengths: Dict[int, int],
        batch_size: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        total_steps = len(store)
        if total_steps <= 0:
            raise RuntimeError("Cannot sample from an empty combined store")

        generator = rng if rng is not None else np.random.default_rng()
        anchor_indices = generator.integers(0, total_steps, size=batch_size, endpoint=False)

        episode_ids = np.asarray(store.leaves[("collector", "episode_id")])
        obs_flat = self._build_flat_indices(
            anchor_indices, self._offsets, episode_ids, episode_starts, episode_lengths
        )
        next_flat = self._build_flat_indices(
            anchor_indices, self._next_offsets, episode_ids, episode_starts, episode_lengths
        )

        batch_flat: Dict[Tuple[str, ...], Any] = {}

        for path, payload in store.leaves.items():
            if isinstance(payload, np.ndarray):
                batch_flat[path] = payload[anchor_indices]
            else:
                batch_flat[path] = [payload[int(i)] for i in anchor_indices]

        for key_tuple, idx in obs_flat.items():
            src = np.asarray(store.leaves[key_tuple])
            temporal = self._apply_image_permutation(src[idx], key_tuple)
            batch_flat[key_tuple] = temporal

        for key_tuple, idx in next_flat.items():
            src = np.asarray(store.leaves[key_tuple])
            temporal = self._apply_image_permutation(src[idx], key_tuple)
            batch_flat[("next",) + key_tuple] = temporal

        return _to_jax_tree(_unflatten_leaves(batch_flat))


def _to_jax_tree(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _to_jax_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jax_tree(v) for v in value]
    if isinstance(value, np.ndarray):
        if jnp is None:
            return value
        return jnp.asarray(value)
    return value


def _safe_gfile_listdir(tf_module: Optional[object], path: str) -> List[str]:
    if tf_module is None:
        return []
    try:
        return tf_module.io.gfile.listdir(path)
    except Exception:
        return []


def _safe_gfile_isdir(tf_module: Optional[object], path: str) -> bool:
    if tf_module is None:
        return False
    try:
        return tf_module.io.gfile.isdir(path)
    except Exception:
        return False


def _get_dataset_map(refresh: bool = False, dataset_name: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    global _DATASET_CACHE

    if not refresh and _DATASET_CACHE is not None and (
        dataset_name is None or dataset_name in _DATASET_CACHE
    ):
        return _DATASET_CACHE

    if _DATASET_CACHE is None:
        _DATASET_CACHE = {}

    if dataset_name is not None and (refresh or dataset_name not in _DATASET_CACHE):
        direct = discover_dataset_versions(tf, OXE_BUCKET_URL, dataset_name)
        if direct:
            _DATASET_CACHE[dataset_name] = direct
            return _DATASET_CACHE

    if refresh or not _DATASET_CACHE:
        _DATASET_CACHE = discover_datasets_from_bucket(tf, OXE_BUCKET_URL)

    return _DATASET_CACHE


def list_datasets(refresh: bool = False) -> Dict[str, List[str]]:
    return {
        name: sorted(versions.keys(), key=normalize_version_key, reverse=True)
        for name, versions in _get_dataset_map(refresh).items()
    }


def validate_dataset_name(dataset_name: str, version: Optional[str] = None) -> bool:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        return False
    if version is not None:
        return version in datasets[dataset_name]
    return True


def dataset2path(dataset_name: str, version: Optional[str] = None) -> str:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Use list_datasets() to see available datasets."
        )
    versions = datasets[dataset_name]
    if version is not None:
        if version not in versions:
            available = ", ".join(sorted(versions.keys(), key=normalize_version_key, reverse=True))
            raise ValueError(
                f"Version '{version}' not available for '{dataset_name}'. Available: {available}"
            )
        return versions[version]
    return versions[latest_version(list(versions.keys()))]


def _gcs_walk_files(src: str, dst: str) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    src = src.rstrip("/")
    dst = dst.rstrip("/")
    for entry in tf.io.gfile.listdir(src):
        name = entry.strip("/")
        s = f"{src}/{name}"
        d = f"{dst}/{name}"
        if tf.io.gfile.isdir(s):
            pairs.extend(_gcs_walk_files(s, d))
        else:
            pairs.append((s, d))
    return pairs


def _needs_download(dst_file: str) -> bool:
    if not tf.io.gfile.exists(dst_file):
        return True
    try:
        return tf.io.gfile.stat(dst_file).length == 0
    except Exception:
        return True


def _is_data_shard(filename: str) -> bool:
    return any(ext in filename for ext in (".tfrecord", ".riegeli", ".array_record"))


def _copy_tree(src: str, dst: str) -> None:
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    for src_file, dst_file in _progress(
        pairs, desc="Downloading dataset files", unit="file", total=len(pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


def _copy_metadata_only(src: str, dst: str) -> None:
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    metadata_pairs = [(s, d) for s, d in pairs if not _is_data_shard(s)]
    for src_file, dst_file in _progress(
        metadata_pairs, desc="Downloading dataset metadata", unit="file", total=len(metadata_pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


class OXEJAXDataset:
    def __init__(
        self,
        dataset_name: str = "droid",
        split: str = "train",
        version: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 10.0,
    ) -> None:
        _require_tf_stack()
        _require_jax()

        dataset_name = dataset_name.strip("/")
        if not validate_dataset_name(dataset_name, version):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'"
                + (f" version '{version}'" if version else "")
                + f". Available datasets: {', '.join(sorted(list_datasets().keys()))}"
            )

        self.dataset_name = dataset_name
        self.split = split
        self.version = version
        self.episodes: Optional[List[int]] = list(episodes) if episodes is not None else None
        self.dataset_path = dataset2path(dataset_name, version=version)
        self.root = _get_cache_dir(root)
        self.batch_size = batch_size

        local_dir = self._local_tfds_dir()
        info_file = local_dir / "dataset_info.json"
        if not info_file.exists() or info_file.stat().st_size == 0:
            _copy_metadata_only(self.dataset_path, str(local_dir))

        try:
            self.builder = tfds.builder_from_directory(builder_dir=str(local_dir))
            self.info = getattr(self.builder, "info", None)
            self._modalities = self._infer_modalities()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from '{local_dir}'. {error}"
            ) from error

        episodes_dir = self._episodes_dir()
        if self.episodes is not None:
            selected = sorted(self.episodes)
        else:
            selected = list(range(self._get_total_episodes()))

        missing = [idx for idx in selected if not is_episode_cached(episodes_dir / str(idx))]

        if missing:
            if self.episodes is not None:
                builder_for_data = tfds.builder_from_directory(builder_dir=self.dataset_path)
            else:
                _copy_tree(self.dataset_path, str(local_dir))
                builder_for_data = self.builder

            build_missing_episodes(
                builder=builder_for_data,
                split=split,
                episodes_dir=episodes_dir,
                missing=missing,
                tf_tensor_types=_TF_TENSOR_TYPES,
            )

        self._loaded_indices = selected
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)

        self._store = CombinedNumpyStore(combined_dir)

        default_dt: Dict[str, List[float]] = {
            path: [0.0]
            for path, spec in self._modalities.items()
            if spec.get("dtype") is not None and spec.get("kind") != "text"
        }
        effective_dt = {**default_dt, **(delta_timestamps or {})}

        episode_ids = np.asarray(self._store.leaves[("collector", "episode_id")])
        self._episode_starts, self._episode_lengths = JAXTemporalSampler.build_episode_index(episode_ids)
        self._temporal_sampler = JAXTemporalSampler(
            delta_timestamps=effective_dt,
            control_frequency=control_frequency,
            image_keys=self.image_keys,
        )

    def _local_tfds_dir(self) -> Path:
        path = self.root / "oxe" / self.dataset_name
        if self.version:
            path = path / self.version
        return path

    def _episodes_dir(self) -> Path:
        return self._local_tfds_dir() / "episodes" / self.split

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        return self._local_tfds_dir() / "combined_jax" / self.split / key

    def _get_total_episodes(self) -> int:
        if self.info is not None:
            splits = getattr(self.info, "splits", {})
            split_info = splits.get(self.split)
            if split_info is not None:
                count = getattr(split_info, "num_examples", None)
                if count is not None:
                    return int(count)
        raise RuntimeError(
            f"Cannot determine episode count for '{self.dataset_name}/{self.split}' from builder.info. "
            "Pass an explicit episodes list."
        )

    def _infer_modalities(self) -> Dict[str, Dict[str, Any]]:
        features = None
        if self.info is not None:
            features = getattr(self.info, "features", None)
        if features is None and hasattr(self.builder, "meta"):
            features = getattr(self.builder, "meta", None)
        if features is None:
            return {}
        flattened = _flatten_features(features)
        return {path: asdict(spec) for path, spec in flattened.items()}

    @property
    def num_episodes(self) -> int:
        return len(self._loaded_indices)

    @property
    def image_keys(self) -> List[Tuple[str, ...]]:
        return sorted(
            tuple(path.split("/"))
            for path, spec in self._modalities.items()
            if spec.get("kind") == "image"
        )

    @property
    def modalities(self) -> Dict[str, List[str]]:
        kinds = {"image", "text", "state", "action"}
        grouped: Dict[str, List[str]] = {}
        for path, spec in self._modalities.items():
            kind = spec.get("kind", "generic")
            if kind in kinds:
                grouped.setdefault(kind, []).append(path)
        return {kind: sorted(paths) for kind, paths in grouped.items()}

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        return dict(self._modalities)

    def set_sampler(self, sampler: JAXTemporalSampler) -> None:
        self._temporal_sampler = sampler

    def sample(
        self,
        batch_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        size = int(batch_size if batch_size is not None else self.batch_size)
        return self._temporal_sampler.sample(
            self._store,
            self._episode_starts,
            self._episode_lengths,
            size,
            rng=rng,
        )

    def get_batch(
        self,
        batch_size: Optional[int] = None,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        return self.sample(batch_size=batch_size, rng=rng)

    def __len__(self) -> int:
        return len(self._store)

    @property
    def data_path(self) -> Path:
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        return self._local_tfds_dir()

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(idx)) for idx in self._loaded_indices)


__all__ = [
    "OXEJAXDataset",
    "JAXTemporalSampler",
    "list_datasets",
    "validate_dataset_name",
    "dataset2path",
]
