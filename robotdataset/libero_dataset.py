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


class LiberoDataset(BaseDatasetExperienceReplay):
    """LIBERO offline dataset from HuggingFace as a TorchRL dataset."""

    HF_DATASET = "openvla/modified_libero_rlds"

    def __init__(
        self,
        split: str = "train",
        tasks: Optional[List[int]] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 10.0,
        config_name: Optional[str] = None,
    ) -> None:
        self.split = split
        self.tasks = tasks
        self.config_name = config_name
        self.root = _get_cache_dir(root)

        hf_dataset = load_hf_dataset(
            self.HF_DATASET,
            split,
            self.root,
            config_name=config_name,
        )
        if tasks is not None:
            hf_dataset = filter_by_tasks(hf_dataset, tasks)

        all_episode_ids = get_episode_ids(hf_dataset)
        selected = sorted(episodes) if episodes is not None else all_episode_ids
        self._loaded_indices: List[int] = selected

        episodes_dir = self._episodes_dir()
        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]
        if missing:
            build_missing_episodes(hf_dataset, episodes_dir, missing)

        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        storage = TensorStorage(combined_td)

        self.modalities = infer_modalities_from_storage(combined_td)

        default_delta_timestamps: Dict[str, List[float]] = {
            path: [0.0]
            for path, spec in self.modalities.items()
            if spec.get("dtype") is not None and spec.get("kind") != "text"
        }
        effective_dt = {**default_delta_timestamps, **(delta_timestamps or {})}

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

    @property
    def data_path(self) -> Path:
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        return self.root / "hf" / "libero"

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    def _episodes_dir(self) -> Path:
        base = self.root / "hf" / "libero" / self.split
        if self.config_name:
            base = base / self.config_name
        return base / "episodes"

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        base = self.root / "hf" / "libero" / self.split
        if self.config_name:
            base = base / self.config_name
        return base / "combined" / key

    @property
    def num_episodes(self) -> int:
        return len(self._loaded_indices)

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
