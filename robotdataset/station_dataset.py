"""Reduced-functionality loader for a local folder of Station (norma-core) parquet episodes.

Station's ``dataset-generator`` writes one ``.parquet`` file per episode (see
elrobot's ``inference-client`` repo, ``DATA_FORMAT.md``): each row is one
frame, with a ``joints`` struct list (``position_norm`` / ``goal_norm`` per
joint), an ``images`` struct list (one JPEG per camera) and a constant
``task`` string for the episode. There is no HuggingFace Hub id and no video
files to decode -- just a directory of files on disk -- so this loader skips
the Hub-download and mp4-decode machinery the other loaders need and goes
straight from parquet rows to the same TED-memmap pipeline they end in
(:mod:`robotdataset.oxe.memmap_builder`, :mod:`robotdataset.oxe.temporal_sampler`,
:class:`robotdataset.dataset_stats.DatasetStatsMixin`).

Reduced relative to :class:`robotdataset.table30v2_dataset.Table30v2Dataset` /
:class:`robotdataset.lerobot_video_dataset.LeRobotVideoDataset`:

- No Hub download, no revision pinning, no mp4 decoding.
- One file == one episode, episode id == its sorted position in the
  directory listing (no episode-id column to key off).
- fps is measured from ``timestamp_ns_since_episode_start``, not read from a
  ``meta/info.json``.
- Camera names are not stored anywhere upstream, so they are assigned
  positionally (``cam0``, ``cam1``, ...) in ``images`` list order, matching
  the convention inference-client's bridge hardcodes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.dataset_stats import DatasetStatsMixin
from robotdataset.hf.loader import infer_modalities_from_storage
from robotdataset.oxe.memmap_builder import (
    _EPISODE_SENTINEL,
    build_combined_storage,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.temporal_sampler import TemporalSampler
from robotdataset.oxe.utils import episode_to_ted_steps

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:  # pragma: no cover
    _tqdm_cls = None


def _decode_jpeg(data: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(__import__("io").BytesIO(data)) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _parquet_episode_files(data_dir: Path) -> List[Path]:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No .parquet files found in {data_dir}")
    return files


def parquet_episode_to_ted_dict(
    path: Path, camera_names: Optional[Sequence[str]] = None
) -> Dict[str, Any]:
    """One Station parquet file -> an OXE-style episode dict (see ``episode_to_ted_steps``).

    Every row becomes one step with ``observation/images/<cam>`` (uint8 HWC),
    ``observation/state`` (float32, joint ``position_norm``, Frame order),
    ``observation/language_instruction`` (the episode's ``task`` string) and
    top-level ``action`` (float32, joint ``goal_norm``, same order/width as
    state). Raises if ``task`` is not constant within the file, since GR00T
    takes one instruction per episode.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    cols = table.to_pydict()
    n_rows = len(cols["joints"])
    if n_rows == 0:
        return {"steps": []}

    has_task = "task" in cols
    if has_task:
        tasks = set(cols["task"])
        if len(tasks) > 1:
            raise ValueError(f"{path}: episode has {len(tasks)} distinct task strings, expected 1")
        task = next(iter(tasks))

    n_images = len(cols["images"][0])
    names = list(camera_names) if camera_names is not None else [f"cam{i}" for i in range(n_images)]
    if len(names) != n_images:
        raise ValueError(f"{path}: row has {n_images} images but {len(names)} camera_names given")

    steps: List[Dict[str, Any]] = []
    for i in range(n_rows):
        joints = cols["joints"][i]
        state = torch.tensor([float(j["position_norm"]) for j in joints], dtype=torch.float32)
        action = torch.tensor([float(j["goal_norm"]) for j in joints], dtype=torch.float32)
        images = {
            name: torch.from_numpy(_decode_jpeg(cols["images"][i][k]["jpeg"]))
            for k, name in enumerate(names)
        }
        observation: Dict[str, Any] = {"images": images, "state": state}
        if has_task:
            observation["language_instruction"] = task
        steps.append({"observation": observation, "action": action})
    return {"steps": steps}


def _episode_fps(path: Path) -> Optional[float]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["timestamp_ns_since_episode_start"])
    ts = np.asarray(table.column(0).to_pylist(), dtype=np.int64)
    if len(ts) < 2:
        return None
    dt = np.diff(np.sort(ts))
    dt = dt[dt > 0]
    if len(dt) == 0:
        return None
    return float(1e9 / np.median(dt))


def _build_one_episode(path: Path, episode_id: int, episode_dir: Path, camera_names) -> int:
    episode = parquet_episode_to_ted_dict(path, camera_names)
    steps = episode_to_ted_steps(episode, episode_id, tf_tensor_types=())
    if not steps:
        return 0
    episode_dir.mkdir(parents=True, exist_ok=True)
    td = torch.stack(steps)
    td.memmap_(str(episode_dir))
    n_steps = len(td)
    (episode_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
    return n_steps


class StationParquetDataset(DatasetStatsMixin, BaseDatasetExperienceReplay):
    """A folder of Station parquet episodes as a TorchRL ``BaseDatasetExperienceReplay``.

    One file per episode, converted once to a per-episode TED memmap and
    cached under ``root`` (same combined-memmap + statistics machinery as
    :class:`~robotdataset.table30v2_dataset.Table30v2Dataset`); a later run
    against the same ``data_dir``/``episodes`` selection skips conversion
    entirely.

    Usage::

        ds = StationParquetDataset(data_dir="/data/elrobot/cheese_episodes", batch_size=8)
        batch = ds.sample()
        ds.num_episodes

    Args:
        data_dir: Directory holding one ``.parquet`` file per episode.
        episodes: Episode indices (position in the sorted file listing) to
            load. Loads every file if ``None``.
        batch_size: Number of transitions returned by ``sample()``.
        root: Override cache root directory (see ``_get_cache_dir``).
        delta_timestamps: Per-modality time-delta lists for temporal
            sampling, as in the other loaders.
        control_frequency: Steps per second for converting time deltas to
            step offsets. Defaults to the fps measured from the first loaded
            episode's timestamps.
        camera_names: Names assigned to ``images`` list entries, in order.
            Defaults to ``cam0, cam1, ...``.
    """

    def __init__(
        self,
        data_dir: str,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: Optional[float] = None,
        camera_names: Optional[Sequence[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.root = _get_cache_dir(root)
        self.camera_names = list(camera_names) if camera_names is not None else None

        self._files = _parquet_episode_files(self.data_dir)
        all_episode_ids = list(range(len(self._files)))
        selected = sorted(episodes) if episodes is not None else all_episode_ids
        unknown = [e for e in selected if e not in all_episode_ids]
        if unknown:
            raise IndexError(
                f"Episodes {unknown} are not in this dataset "
                f"(it has {len(all_episode_ids)} files in {self.data_dir})"
            )
        self._loaded_indices: List[int] = selected

        episodes_dir = self._episodes_dir()
        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]
        pbar = (
            _tqdm_cls(total=len(missing), desc="Converting episodes to TED", unit="ep", dynamic_ncols=True)
            if missing and _tqdm_cls is not None
            else None
        )
        for i in missing:
            _build_one_episode(self._files[i], i, episodes_dir / str(i), self.camera_names)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()

        self._ensure_episode_stats()

        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        self.modalities = infer_modalities_from_storage(combined_td)

        _EXCLUDED_PREFIXES = ("next/", "collector/")
        _EXCLUDED_KEYS = {"done", "terminated"}
        default_dt: Dict[str, List[float]] = {
            path: [0.0]
            for path, spec in self.modalities.items()
            if spec.get("dtype") is not None
            and spec.get("kind") != "text"
            and not any(path.startswith(p) for p in _EXCLUDED_PREFIXES)
            and path not in _EXCLUDED_KEYS
        }
        effective_dt = {**default_dt, **(delta_timestamps or {})}
        self._fps = control_frequency if control_frequency is not None else _episode_fps(self._files[selected[0]])
        self.control_frequency = self._fps if self._fps is not None else 10.0

        self._episode_starts: Dict[int, int]
        self._episode_lengths: Dict[int, int]
        self._episode_starts, self._episode_lengths = TemporalSampler.build_episode_index(combined_td)
        self._temporal_sampler = TemporalSampler(
            delta_timestamps=effective_dt,
            control_frequency=self.control_frequency,
            image_keys=self.image_keys,
        )

        super().__init__(
            storage=storage,
            sampler=self._temporal_sampler,
            writer=ImmutableDatasetWriter(),
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # BaseDatasetExperienceReplay abstract interface
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        return self.root / "station" / self.data_dir.resolve().as_posix().strip("/").replace("/", "_")

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episodes_dir(self) -> Path:
        return self.data_path_root / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        return self.data_path_root / "combined" / combined_dir_key(selected)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        return len(self._loaded_indices)

    @property
    def fps(self) -> Optional[float]:
        return self._fps

    @property
    def image_keys(self) -> frozenset:
        return frozenset(
            tuple(path.split("/"))
            for path, spec in self.modalities.items()
            if spec.get("kind") == "image"
        )

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.modalities)

    def _sample(self, batch_size: int) -> Any:
        batch = self._temporal_sampler(
            self._storage._storage,
            self._episode_starts,
            self._episode_lengths,
            batch_size,
        )
        return batch, {}

    def set_sampler(self, sampler: TemporalSampler) -> None:
        self._temporal_sampler = sampler
        self._sampler = sampler
