from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.agibot.loader import (
    DEFAULT_CAMERAS,
    build_episode_mapping,
    get_repo_files,
    load_episode_map,
    load_task_info,
    save_episode_map,
)
from robotdataset.agibot.memmap_builder import build_missing_agibot_episodes
from robotdataset.hf.loader import infer_modalities_from_storage
from robotdataset.oxe.memmap_builder import (
    build_combined_storage,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.temporal_sampler import TemporalSampler


class AgiBotWorldBetaDataset(BaseDatasetExperienceReplay):
    """AgiBotWorld-Beta offline dataset from HuggingFace as a TorchRL ``BaseDatasetExperienceReplay``.

    Loads ``agibot-world/AgiBotWorld-Beta``.  The dataset is structured as
    TAR archives (``observations/{task_id}/*.tar``) containing MP4 video
    streams from 8 cameras.  Metadata (episode list, language labels) lives in
    ``task_info/task_{id}.json``.

    Only the ``tasks`` you specify are downloaded; only the ``episodes`` you
    request are decoded from video and converted to TED memmaps.

    TED layout per step::

        TensorDict({
            "observation": TensorDict({
                "head_color":       Tensor([H, W, 3], uint8),
                "hand_left_color":  Tensor([H, W, 3], uint8),
                "hand_right_color": Tensor([H, W, 3], uint8),
                "language_instruction": NonTensor(str),
            }),
            "action":     Tensor([1], float32),   # zero placeholder
            "done":       Tensor([1], bool),
            "terminated": Tensor([1], bool),
            "next": TensorDict({
                "observation": TensorDict({...}),
                "reward":      Tensor([1], float32),
                "done":        Tensor([1], bool),
                "terminated":  Tensor([1], bool),
            }),
            "collector": TensorDict({
                "episode_id": Tensor(int64),
            }),
        })

    Usage::

        from robotdataset import AgiBotWorldBetaDataset, list_agibot_tasks

        print(list_agibot_tasks()[:5])     # [327, 351, 352, ...] — no download
        ds = AgiBotWorldBetaDataset(
            tasks=[327],
            episodes=[0, 1],
            batch_size=16,
        )
        batch = ds.sample()
        print(ds.num_episodes)             # 2
        print(ds.get_modalities().keys())

    **Global episode IDs**: episodes are numbered globally across all tasks in
    sorted ``(task_id, episode_id)`` order.  Use ``ds.episode_map`` to
    inspect the mapping::

        {0: (327, 648649), 1: (327, 648709), ...}

    **Camera selection**: the default cameras are
    ``["head_color", "hand_left_color", "hand_right_color"]``.
    Pass ``cameras=[...]`` to override.  Use
    ``robotdataset.agibot.loader.ALL_CAMERAS`` for the full list.

    Cache directory (priority order):
        1. ``root`` argument
        2. ``ROBOTDATASET_CACHE`` environment variable
        3. ``~/.cache/robotdataset``  (default)

    Args:
        tasks: Integer task IDs to load, e.g. ``[327, 351]``.
            Use :func:`list_agibot_tasks` to discover available IDs.
        split: Kept for API consistency; AgiBot has a single pool of data.
        episodes: Global episode indices to include.  Uses all episodes from
            the selected tasks when ``None``.
        cameras: Camera streams to decode.  Defaults to
            ``["head_color", "hand_left_color", "hand_right_color"]``.
        batch_size: Transitions returned by ``sample()``.
        root: Override cache root directory.
        delta_timestamps: Per-modality time-delta lists (seconds) for temporal
            sampling.  Keys are slash-separated paths, e.g.
            ``{"observation/head_color": [-0.1, 0.0, 0.1]}``.
        control_frequency: Frames per second of the source video (default
            30.0, matching AgiBotWorld-Beta recording rate).
    """

    HF_DATASET = "agibot-world/AgiBotWorld-Beta"

    def __init__(
        self,
        tasks: List[int],
        split: str = "train",
        episodes: Optional[List[int]] = None,
        cameras: Optional[List[str]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 30.0,
    ) -> None:
        self.tasks = list(tasks)
        self.split = split
        self.cameras: List[str] = list(cameras) if cameras is not None else list(DEFAULT_CAMERAS)
        self.root = _get_cache_dir(root)

        # ------------------------------------------------------------------
        # 1. Download task metadata (task_info JSON) for each requested task.
        #    These are small JSON files — fast even on first call.
        # ------------------------------------------------------------------
        task_infos: Dict[int, List[Dict]] = {
            t: load_task_info(t, self.root) for t in self.tasks
        }

        # ------------------------------------------------------------------
        # 2. Build / load the global episode ID mapping.
        #    Persisted to JSON so subsequent inits skip re-reading task_info.
        # ------------------------------------------------------------------
        map_path = self._episodes_dir() / "episode_map.json"
        if map_path.exists():
            full_map: Dict[int, Tuple[int, int]] = load_episode_map(map_path)
            # Add new tasks if they were not present when the map was saved.
            known_task_ids = {tid for _, (tid, _) in full_map.items()}
            new_task_infos = {t: task_infos[t] for t in self.tasks if t not in known_task_ids}
            if new_task_infos:
                extension = build_episode_mapping(new_task_infos)
                offset = max(full_map) + 1 if full_map else 0
                full_map.update({k + offset: v for k, v in extension.items()})
                save_episode_map(map_path, full_map)
        else:
            full_map = build_episode_mapping(task_infos)
            save_episode_map(map_path, full_map)

        self.episode_map: Dict[int, Tuple[int, int]] = full_map

        # ------------------------------------------------------------------
        # 3. Filter to global IDs that belong to the requested tasks.
        # ------------------------------------------------------------------
        requested_task_set = set(self.tasks)
        task_global_ids = [
            gid for gid, (tid, _) in full_map.items() if tid in requested_task_set
        ]
        selected = sorted(episodes) if episodes is not None else sorted(task_global_ids)
        self._loaded_indices: List[int] = selected

        # ------------------------------------------------------------------
        # 4. Download TAR files and decode only the missing episodes.
        # ------------------------------------------------------------------
        episodes_dir = self._episodes_dir()
        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]
        if missing:
            repo_files = get_repo_files()
            build_missing_agibot_episodes(
                task_infos=task_infos,
                episode_map=full_map,
                episodes_dir=episodes_dir,
                missing=missing,
                cameras=self.cameras,
                repo_files=repo_files,
                cache_dir=self.root,
            )

        # ------------------------------------------------------------------
        # 5. Build (or reuse) the combined memmap and load it lazily.
        # ------------------------------------------------------------------
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        # ------------------------------------------------------------------
        # 6. Infer modalities from the built storage.
        # ------------------------------------------------------------------
        self.modalities = infer_modalities_from_storage(combined_td)

        # ------------------------------------------------------------------
        # 7. Temporal sampler — always active.
        #    Default: T=1 (anchor only) for every tensor modality.
        #    Camera images are auto-permuted from HWC to CHW on sampling.
        # ------------------------------------------------------------------
        default_dt: Dict[str, List[float]] = {
            path: [0.0]
            for path, spec in self.modalities.items()
            if spec.get("dtype") is not None and spec.get("kind") != "text"
        }
        effective_dt = {**default_dt, **(delta_timestamps or {})}

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
        """Per-episode TED memmap directory."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root cache path for this dataset."""
        return self.root / "hf" / "agibotworld-beta"

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episodes_dir(self) -> Path:
        return self.root / "hf" / "agibotworld-beta" / self.split / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        return self.root / "hf" / "agibotworld-beta" / self.split / "combined" / key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of episodes currently loaded."""
        return len(self._loaded_indices)

    @property
    def image_keys(self) -> frozenset:
        """Tuple-path keys for image modalities (stored as HWC on disk).

        The default :class:`~robotdataset.oxe.temporal_sampler.TemporalSampler`
        uses this to permute images to channels-first ``(B, T, C, H, W)``
        format on sampling.  Pass this to a custom sampler::

            sampler = TemporalSampler(..., image_keys=dataset.image_keys)
            dataset.set_sampler(sampler)
        """
        return frozenset(
            tuple(path.split("/"))
            for path, spec in self.modalities.items()
            if spec.get("kind") == "image"
        )

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        """Return modality specs inferred from the TED storage."""
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
            sampler: A configured :class:`TemporalSampler`.
        """
        self._temporal_sampler = sampler
        self._sampler = sampler
