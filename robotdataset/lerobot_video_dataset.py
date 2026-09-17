from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.hf.loader import infer_modalities_from_storage
from robotdataset.lerobot.loader import LeRobotSource
from robotdataset.lerobot.memmap_builder import build_missing_episodes
from robotdataset.oxe.memmap_builder import (
    build_combined_storage,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.temporal_sampler import TemporalSampler


class LeRobotVideoDataset(BaseDatasetExperienceReplay):
    """LeRobot v2.1 dataset with external mp4 video as a TorchRL replay buffer.

    Works with any LeRobot dataset whose camera streams are separate mp4 files
    rather than image columns inside the parquet -- the layout
    ``LeRobotDataset.push_to_hub`` produces.  Episodes are converted to TED
    format once and persisted as memory-mapped tensors, so the second run of a
    given episode selection skips download and decode entirely.

    Two origins are supported:

    - ``local_path=`` -- a directory already holding the LeRobot tree.  No
      network access at all, which is what you want on a cluster node or when
      the data was staged by someone else.
    - ``repo_id=`` -- a HuggingFace Hub dataset id.  Only the meta files plus
      the parquet and mp4 assets of the requested episodes are fetched, so
      ``episodes=[0, 1]`` on a 50-episode repo transfers two episodes.

    TED layout per step::

        TensorDict({
            "observation":  TensorDict({
                "images":   TensorDict({<camera>: uint8 (H, W, C)}),
                "state":    TensorDict({<slice>: float32 (D,)}),
                "language_instruction": str,
            }),
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

    A LeRobot dataset that ships both ``observation.state`` and
    ``observation.state.<slice>`` columns cannot represent both under one name
    in a nested TensorDict; the concatenated vector is stored at
    ``observation/state/all`` and the slices keep their own names.

    Usage::

        ds = LeRobotVideoDataset(
            repo_id="LSY-lab/stack_without_ft_tact_v4",
            episodes=[0, 1],
            batch_size=8,
            delta_timestamps={"observation/images/primary": [-0.2, -0.1, 0.0]},
        )
        batch = ds.sample()
        batch["observation/images/primary"].shape  # (8, 3, 3, 256, 256) B,T,C,H,W
        ds.num_episodes                            # 2

    Cache directory (priority order):
        1. ``root`` argument
        2. ``ROBOTDATASET_CACHE`` environment variable
        3. ``~/.cache/robotdataset``  (default)

    Args:
        repo_id: HuggingFace Hub dataset id.  Mutually exclusive with
            ``local_path``.
        local_path: Directory holding an already-downloaded LeRobot tree.
            Mutually exclusive with ``repo_id``.
        episodes: Episode indices to load.  Loads every episode if None.
        batch_size: Number of transitions returned by ``sample()``.
        root: Override cache root directory.
        delta_timestamps: Per-modality time-delta lists for temporal sampling.
            Keys are slash-separated modality paths; values are lists of
            seconds, e.g. ``{"observation/images/wrist": [-0.1, 0.0]}``.
        control_frequency: Steps per second used to convert time deltas to
            step offsets.  Defaults to the dataset's own ``meta/info.json``
            fps rather than a fixed value.
        video_keys: Camera streams to decode.  Defaults to every video feature
            in the dataset; naming a subset skips downloading and decoding the
            others.
        revision: Hub revision (branch, tag or commit sha) when using
            ``repo_id``.  Pin a commit sha for reproducibility across machines.
    """

    def __init__(
        self,
        repo_id: Optional[str] = None,
        local_path: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: Optional[float] = None,
        video_keys: Optional[List[str]] = None,
        revision: Optional[str] = None,
    ) -> None:
        self.root = _get_cache_dir(root)

        # ------------------------------------------------------------------
        # 1. Resolve the source (parses meta/, fetches nothing else yet)
        # ------------------------------------------------------------------
        self.source = LeRobotSource(
            repo_id=repo_id,
            local_path=local_path,
            cache_dir=str(self.root),
            revision=revision,
        )
        self.video_keys_loaded: List[str] = list(
            self.source.video_keys if video_keys is None else video_keys
        )

        # ------------------------------------------------------------------
        # 2. Determine which episodes to load
        # ------------------------------------------------------------------
        all_episode_ids = self.source.get_episode_ids()
        selected = sorted(episodes) if episodes is not None else all_episode_ids
        unknown = [e for e in selected if e not in self.source.episodes_meta]
        if unknown:
            raise IndexError(
                f"Episodes {unknown} are not in this dataset "
                f"(it has {len(all_episode_ids)}: {all_episode_ids[:3]}...)"
            )
        self._loaded_indices: List[int] = selected

        # ------------------------------------------------------------------
        # 3. Fetch, decode and cache missing episodes as TED memmaps
        # ------------------------------------------------------------------
        episodes_dir = self._episodes_dir()
        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]
        if missing:
            build_missing_episodes(
                self.source, episodes_dir, missing, self.video_keys_loaded
            )

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
        self.control_frequency = (
            self.source.fps if control_frequency is None else float(control_frequency)
        )

        self._episode_starts: Dict[int, int]
        self._episode_lengths: Dict[int, int]
        self._episode_starts, self._episode_lengths = (
            TemporalSampler.build_episode_index(combined_td)
        )
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
        """Per-episode TED memmap directory for this dataset."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root path for all cached data for this dataset."""
        return self.root / "hf" / "lerobot_video" / self.source.slug

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _episodes_dir(self) -> Path:
        return self.data_path_root / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        return self.data_path_root / "combined" / key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of episodes loaded into this dataset."""
        return len(self._loaded_indices)

    @property
    def fps(self) -> float:
        """The dataset's own control frequency, from ``meta/info.json``."""
        return self.source.fps

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
