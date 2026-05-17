from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.hf.loader import (
    filter_by_tasks,
    get_episode_ids,
    infer_modalities_from_storage,
    load_hf_dataset,
)
from robotdataset.hf.memmap_builder import build_missing_episodes
from robotdataset.oxe.memmap_builder import (
    build_combined_storage,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.temporal_sampler import TemporalSampler


class Table30v2Dataset(BaseDatasetExperienceReplay):
    """Table30v2 offline dataset from HuggingFace as a TorchRL ``BaseDatasetExperienceReplay``.

    Loads ``RoboChallenge/Table30v2`` from HuggingFace Hub.  On first use,
    episodes are downloaded via the ``datasets`` library and converted to TED
    (Trajectory Episode Data) format, then persisted as memory-mapped tensors
    for fast subsequent access.

    TED layout per step::

        TensorDict({
            "observation":  TensorDict({...}),   # step[t] modalities
            "action":       Tensor,
            "done":         Tensor([1], bool),
            "terminated":   Tensor([1], bool),
            "next": TensorDict({
                "observation": TensorDict({...}),
                "reward":      Tensor([1]),
                "done":        Tensor([1], bool),
                "terminated":  Tensor([1], bool),
            }),
            "collector": TensorDict({
                "episode_id": Tensor(int64),
            }),
        })

    Usage::

        ds = Table30v2Dataset(episodes=[0, 1, 2], batch_size=32)
        batch = ds.sample()
        print(ds.num_episodes)   # 3
        print(len(ds))           # total steps

    Cache directory (priority order):
        1. ``root`` argument
        2. ``ROBOTDATASET_CACHE`` environment variable
        3. ``~/.cache/robotdataset``  (default)

    Args:
        split: Dataset split, e.g. ``"train"``.
        tasks: List of task IDs to include.  Only episodes belonging to these
            tasks are loaded.  Uses all tasks if None.
        episodes: List of episode indices to include within the selected tasks.
            Uses all episodes from the selected tasks if None.
        batch_size: Number of transitions returned by ``sample()``.
        root: Override cache root directory.
        delta_timestamps: Per-modality time-delta lists for temporal sampling.
            Keys are slash-separated modality paths; values are lists of seconds.
            e.g. ``{"observation/image": [-0.1, 0.0, 0.2]}``
        control_frequency: Steps per second for converting time deltas to step
            offsets (default 10.0).
    """

    HF_DATASET = "RoboChallenge/Table30v2"

    def __init__(
        self,
        split: str = "train",
        tasks: Optional[List[int]] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 10.0,
    ) -> None:
        self.split = split
        self.tasks = tasks
        self.root = _get_cache_dir(root)

        # ------------------------------------------------------------------
        # 1. Load HuggingFace dataset (HF handles download + caching)
        # ------------------------------------------------------------------
        hf_dataset = load_hf_dataset(self.HF_DATASET, split, self.root)
        if tasks is not None:
            hf_dataset = filter_by_tasks(hf_dataset, tasks)

        # ------------------------------------------------------------------
        # 2. Determine which episodes to load
        # ------------------------------------------------------------------
        all_episode_ids = get_episode_ids(hf_dataset)
        selected = sorted(episodes) if episodes is not None else all_episode_ids
        self._loaded_indices: List[int] = selected

        # ------------------------------------------------------------------
        # 3. Convert and cache missing episodes as TED memmaps
        # ------------------------------------------------------------------
        episodes_dir = self._episodes_dir()
        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]
        if missing:
            build_missing_episodes(hf_dataset, episodes_dir, missing)

        # ------------------------------------------------------------------
        # 4. Build (or reuse) combined memmap, then load it lazily
        # ------------------------------------------------------------------
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        # ------------------------------------------------------------------
        # 5. Infer modalities from the built storage
        # ------------------------------------------------------------------
        self.modalities = infer_modalities_from_storage(combined_td)

        # ------------------------------------------------------------------
        # 6. Temporal sampler — always active.
        #    Default: {every_tensor_modality: [0.0]} (T=1 anchor-only).
        #    Images are auto-permuted from on-disk HWC to CHW.
        # ------------------------------------------------------------------
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
        """Per-episode TED memmap directory for the current split."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root path for all cached data for this dataset."""
        return self.root / "hf" / "table30v2"

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episodes_dir(self) -> Path:
        return self.root / "hf" / "table30v2" / self.split / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        return self.root / "hf" / "table30v2" / self.split / "combined" / key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of episodes loaded into this dataset."""
        return len(self._loaded_indices)

    @property
    def image_keys(self) -> frozenset:
        """Tuple-path keys for image modalities stored as HWC on disk.

        These are automatically applied to the default :class:`TemporalSampler`
        for HWC→CHW permutation.  Pass to a custom sampler when using
        :meth:`set_sampler`::

            sampler = TemporalSampler(..., image_keys=dataset.image_keys)
            dataset.set_sampler(sampler)
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
        """Replace the temporal sampler without rebuilding the dataset.

        Args:
            sampler: A :class:`TemporalSampler` with the desired configuration.
        """
        self._temporal_sampler = sampler
        self._sampler = sampler
