from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.hf.loader import infer_modalities_from_storage
from robotdataset.oxe.memmap_builder import (
    build_combined_storage,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.temporal_sampler import TemporalSampler


HF_REPO_ID   = "amandlek/robomimic"
HF_SUBFOLDER = "v1.5"
FILE_SUFFIX  = "_v15"

VALID_TASKS  = frozenset({"lift", "can", "square", "transport", "tool_hang"})
VALID_TYPES  = frozenset({"ph", "mh", "mg"})
VALID_OBS    = frozenset({"low_dim", "image"})

MG_IMAGE_UNSUPPORTED = frozenset({"mg"})
PH_ONLY_TASKS        = frozenset({"tool_hang"})

_EPISODE_SENTINEL = "_steps.json"


def _hdf5_filename(task: str, dataset_type: str, obs_type: str) -> str:
    """Return the relative path of the HDF5 file within the HF repo."""
    return f"{HF_SUBFOLDER}/{task}/{dataset_type}/{obs_type}{FILE_SUFFIX}.hdf5"


def _validate_combo(task: str, dataset_type: str, obs_type: str) -> None:
    if task not in VALID_TASKS:
        raise ValueError(
            f"Unknown task {task!r}. Valid tasks: {sorted(VALID_TASKS)}"
        )
    if dataset_type not in VALID_TYPES:
        raise ValueError(
            f"Unknown dataset_type {dataset_type!r}. Valid types: {sorted(VALID_TYPES)}"
        )
    if obs_type not in VALID_OBS:
        raise ValueError(
            f"Unknown obs_type {obs_type!r}. Valid obs types: {sorted(VALID_OBS)}"
        )
    if dataset_type in MG_IMAGE_UNSUPPORTED and obs_type == "image":
        raise ValueError(
            f"obs_type='image' is not supported for dataset_type='mg' (machine-generated). "
            "Use obs_type='low_dim' instead."
        )
    if task in PH_ONLY_TASKS and dataset_type != "ph":
        raise ValueError(
            f"Task {task!r} only supports dataset_type='ph', got {dataset_type!r}."
        )


def _download_hdf5(task: str, dataset_type: str, obs_type: str, cache_dir: Path) -> Path:
    """Download the HDF5 file from HuggingFace and return its local path."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise RuntimeError(
            "RobomimicDataset requires 'huggingface_hub'. "
            "Install it with: pip install 'robotdataset[robomimic]'"
        )

    filename = _hdf5_filename(task, dataset_type, obs_type)
    local_path = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir / "robomimic" / "hf_cache"),
        )
    )
    return local_path


def _list_demos(hdf5_path: Path, filter_key: Optional[str] = None) -> List[str]:
    """Return sorted demo key strings, optionally filtered by a mask group."""
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            "RobomimicDataset requires 'h5py'. "
            "Install it with: pip install 'robotdataset[robomimic]'"
        )

    with h5py.File(hdf5_path, "r") as f:
        if filter_key is not None and "mask" in f and filter_key in f["mask"]:
            raw = f["mask"][filter_key][()]
            keys = [k.decode() if isinstance(k, bytes) else k for k in raw]
            return sorted(keys)
        return sorted(f["data"].keys())


def _load_obs_group(group: Any) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for key in group:
        arr = group[key][()]
        if arr.dtype == np.uint8 and arr.ndim == 4:
            # HWC uint8 (T, H, W, C) → CHW float32 (T, C, H, W) in [0, 1]
            tensors[key] = torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0
        else:
            tensors[key] = torch.from_numpy(arr.astype(np.float32))
    return tensors


def _demo_to_ted(h5file: Any, demo_key: str, episode_id: int) -> TensorDict:
    """Convert a single HDF5 demo group to a TED-format TensorDict of shape (T,)."""
    demo = h5file["data"][demo_key]
    T = int(demo.attrs["num_samples"])

    actions = torch.from_numpy(demo["actions"][()].astype(np.float32))                # (T, A)
    rewards = torch.from_numpy(demo["rewards"][()].astype(np.float32)).unsqueeze(1)   # (T, 1)
    dones   = torch.from_numpy(demo["dones"][()].astype(bool)).unsqueeze(1)           # (T, 1)

    obs_tensors      = _load_obs_group(demo["obs"])
    next_obs_tensors = _load_obs_group(demo["next_obs"])

    obs_td      = TensorDict(obs_tensors,      batch_size=[T])
    next_obs_td = TensorDict(next_obs_tensors, batch_size=[T])
    episode_id_t = torch.full((T,), episode_id, dtype=torch.int64)

    return TensorDict(
        {
            "observation": obs_td,
            "action":      actions,
            "done":        dones,
            "terminated":  dones.clone(),
            "next": TensorDict(
                {
                    "observation": next_obs_td,
                    "reward":      rewards,
                    "done":        dones.clone(),
                    "terminated":  dones.clone(),
                },
                batch_size=[T],
            ),
            "collector": TensorDict(
                {"episode_id": episode_id_t},
                batch_size=[T],
            ),
        },
        batch_size=[T],
    )


def _build_episodes(
    hdf5_path: Path,
    episodes_dir: Path,
    indices: List[int],
    demo_keys: List[str],
) -> None:
    """Convert and cache missing demos to per-episode TED memmaps."""
    try:
        import h5py
    except ImportError:
        raise RuntimeError(
            "RobomimicDataset requires 'h5py'. "
            "Install it with: pip install 'robotdataset[robomimic]'"
        )

    missing = [idx for idx in indices if not is_episode_cached(episodes_dir / str(idx))]
    if not missing:
        return

    try:
        from tqdm.auto import tqdm as _tqdm_cls
    except ImportError:
        _tqdm_cls = None  # type: ignore[assignment]

    pbar = (
        _tqdm_cls(
            total=len(missing),
            desc="Converting robomimic episodes",
            unit="ep",
            dynamic_ncols=True,
        )
        if _tqdm_cls is not None
        else None
    )

    with h5py.File(hdf5_path, "r") as f:
        for idx in missing:
            ep_dir = episodes_dir / str(idx)
            if is_episode_cached(ep_dir):
                if pbar is not None:
                    pbar.update(1)
                continue
            ted = _demo_to_ted(f, demo_keys[idx], idx)
            ep_dir.mkdir(parents=True, exist_ok=True)
            ted.memmap_(str(ep_dir))
            n_steps = ted.batch_size[0]
            (ep_dir / _EPISODE_SENTINEL).write_text(json.dumps({"n_steps": n_steps}))
            if pbar is not None:
                pbar.update(1)

    if pbar is not None:
        pbar.close()


class RobomimicDataset(BaseDatasetExperienceReplay):
    """Robomimic offline dataset from HuggingFace as a TorchRL ``BaseDatasetExperienceReplay``.

    Downloads robomimic HDF5 files from ``amandlek/robomimic`` on HuggingFace Hub.
    On first use, demos are converted to TED (Trajectory Episode Data) format and
    persisted as memory-mapped tensors for fast subsequent access.

    TED layout per step::

        TensorDict({
            "observation":  TensorDict({...}),   # step[t] observations
            "action":       Tensor(A,),
            "done":         Tensor(1, bool),
            "terminated":   Tensor(1, bool),
            "next": TensorDict({
                "observation": TensorDict({...}),
                "reward":      Tensor(1,),
                "done":        Tensor(1, bool),
                "terminated":  Tensor(1, bool),
            }),
            "collector": TensorDict({
                "episode_id": Tensor(int64),
            }),
        })

    Images are stored as ``float32`` tensors in ``(C, H, W)`` channel-first
    layout with values in ``[0, 1]``.  No further normalisation is applied by
    the temporal sampler.

    Usage::

        from robotdataset import RobomimicDataset

        # Proficient-human Lift, low-dim observations
        ds = RobomimicDataset(task="lift", dataset_type="ph", batch_size=32)
        batch = ds.sample()
        print(batch["observation"]["robot0_eef_pos"].shape)  # (32, 3)
        print(ds.num_episodes)   # 200
        print(len(ds))           # total steps

        # Multi-human Can, image obs, train split only, 10 episodes
        ds = RobomimicDataset(
            task="can",
            dataset_type="mh",
            obs_type="image",
            filter_key="train",
            episodes=list(range(10)),
            batch_size=8,
            delta_timestamps={
                "observation/agentview_image":          [-0.1, 0.0],
                "observation/robot0_eye_in_hand_image": [-0.1, 0.0],
            },
            control_frequency=20.0,
        )
        batch = ds.sample()
        # (B, T, C, H, W) — channel-first, float32 in [0, 1]
        print(batch["observation"]["agentview_image"].shape)  # (8, 2, 3, 84, 84)

    Cache directory (priority order):
        1. ``root`` argument
        2. ``ROBOTDATASET_CACHE`` environment variable
        3. ``~/.cache/robotdataset``  (default)

    Cache layout::

        ~/.cache/robotdataset/robomimic/<task>/<dataset_type>/<obs_type>/
          <filter_key_or_all>/
            episodes/
              0/   ← per-demo TED memmap
              1/
              ...
            combined/<hash>/data/   ← flat combined memmap

    Args:
        task: Manipulation task — one of ``"lift"``, ``"can"``, ``"square"``,
            ``"transport"``, ``"tool_hang"``.
        dataset_type: Quality level — ``"ph"`` (proficient human), ``"mh"``
            (multi human), or ``"mg"`` (machine generated).
        obs_type: Observation modality — ``"low_dim"`` or ``"image"``.
            ``"mg"`` datasets only support ``"low_dim"``.
        episodes: List of episode indices (0-based into the filtered demo list)
            to include.  Uses all available demos when ``None``.
        filter_key: Optional mask group key for train/val splitting, e.g.
            ``"train"`` or ``"valid"``.  Reads from the ``mask/`` group in the
            HDF5 file.
        batch_size: Number of transitions returned by ``sample()``.
        root: Override cache root directory.
        delta_timestamps: Per-modality time-delta lists for temporal sampling.
            Keys are slash-separated modality paths; values are lists of seconds.
            e.g. ``{"observation/agentview_image": [-0.1, 0.0]}``
        control_frequency: Steps per second for converting time deltas to step
            offsets (default ``20.0`` Hz, the robosuite default).
    """

    def __init__(
        self,
        task: str = "lift",
        dataset_type: str = "ph",
        obs_type: str = "low_dim",
        episodes: Optional[List[int]] = None,
        filter_key: Optional[str] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 20.0,
    ) -> None:
        # 1. Validate task / dataset_type / obs_type combo
        _validate_combo(task, dataset_type, obs_type)

        self.task = task
        self.dataset_type = dataset_type
        self.obs_type = obs_type
        self.filter_key = filter_key
        self.root = _get_cache_dir(root)

        # 2. Download HDF5 file (cached after first download)
        hdf5_path = _download_hdf5(task, dataset_type, obs_type, self.root)

        # 3. List demos, resolve episodes list
        demo_keys = _list_demos(hdf5_path, filter_key)
        all_indices = list(range(len(demo_keys)))
        selected = sorted(episodes) if episodes is not None else all_indices
        self._loaded_indices: List[int] = selected

        # 4. Convert missing demos to TED memmaps
        episodes_dir = self._episodes_dir()
        _build_episodes(hdf5_path, episodes_dir, selected, demo_keys)

        # 5. Build (or reuse) combined memmap, then load it
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        # 6. Infer modalities from storage
        self.modalities = infer_modalities_from_storage(combined_td)

        # 7. Temporal sampler — always active.
        #    Default: every non-text modality at [0.0] (single anchor step).
        #    Images are already stored as float32 CHW — no permutation needed.
        default_dt: Dict[str, List[float]] = {
            path: [0.0]
            for path, spec in self.modalities.items()
            if spec.get("dtype") is not None and spec.get("kind") != "text"
        }
        effective_dt = {**default_dt, **(delta_timestamps or {})}

        self._episode_starts: Dict[int, int]
        self._episode_lengths: Dict[int, int]
        self._episode_starts, self._episode_lengths = (
            TemporalSampler.build_episode_index(combined_td)
        )
        self._temporal_sampler = TemporalSampler(
            delta_timestamps=effective_dt,
            control_frequency=control_frequency,
            image_keys=frozenset(),
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
        """Per-episode TED memmap directory."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root path for all cached data for this dataset variant."""
        return self.root / "robomimic" / self.task / self.dataset_type / self.obs_type

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episodes_dir(self) -> Path:
        fk = self.filter_key or "all"
        return self.data_path_root / fk / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        fk = self.filter_key or "all"
        key = combined_dir_key(selected)
        return self.data_path_root / fk / "combined" / key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of demos loaded into this dataset."""
        return len(self._loaded_indices)

    @property
    def image_keys(self) -> frozenset:
        """Tuple-path keys for image modalities (stored as float32 CHW in [0, 1]).

        Images in this dataset are already stored in channel-first format and
        do not require further permutation.  Pass these to a custom sampler only
        if you need to identify which modalities are images.
        """
        return frozenset(
            tuple(path.split("/"))
            for path, spec in self.modalities.items()
            if spec.get("kind") == "image"
        )

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        """Return a dict of modality specs inferred from the TED storage."""
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
        """Replace the temporal sampler without rebuilding the dataset."""
        self._temporal_sampler = sampler
        self._sampler = sampler

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"task={self.task!r}, "
            f"dataset_type={self.dataset_type!r}, "
            f"obs_type={self.obs_type!r}, "
            f"num_episodes={self.num_episodes}, "
            f"num_steps={len(self)}"
            f")"
        )
