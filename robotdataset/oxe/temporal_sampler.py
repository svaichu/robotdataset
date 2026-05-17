from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Tuple

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import Storage


class TemporalSampler(Sampler):
    """Samples temporally-structured batches from a flat TED storage.

    Every modality listed in ``delta_timestamps`` is gathered at the specified
    time offsets (clamped to episode boundaries) and returned with shape
    ``(B, T, ...)``.  Image modalities (identified by ``image_keys``) are
    additionally permuted from the on-disk HWC layout to channel-first CHW,
    producing ``(B, T, C, H, W)``.

    Args:
        delta_timestamps: Mapping from slash-separated modality path to a list
            of time deltas **in seconds**.
            e.g. ``{"observation/image": [-0.1, 0.0, 0.2], "action": [0.0, 0.1]}``
        control_frequency: Steps per second.  Converts time deltas to integer
            step offsets via ``round(dt * control_frequency)``.  Default: 10.
        image_keys: Tuple-path keys whose tensors are stored in HWC layout and
            must be permuted to CHW after temporal stacking.
            e.g. ``{("observation", "image")}``.
    """

    def __init__(
        self,
        delta_timestamps: Dict[str, List[float]],
        control_frequency: float = 10.0,
        image_keys: FrozenSet[Tuple[str, ...]] = frozenset(),
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.control_frequency = control_frequency
        self.image_keys = image_keys
        # Pre-compute integer step offsets keyed by tuple path
        self._offsets: Dict[Tuple[str, ...], List[int]] = {
            tuple(k.split("/")): [round(dt * control_frequency) for dt in deltas]
            for k, deltas in delta_timestamps.items()
        }

    # ------------------------------------------------------------------
    # Sampler ABC
    # ------------------------------------------------------------------

    @property
    def ran_out(self) -> bool:
        return False

    def sample(self, storage: Storage, batch_size: int) -> Tuple[TensorDict, dict]:
        """Temporal sample called by ReplayBuffer machinery.

        Builds the episode index from ``storage`` on-the-fly and returns the
        full temporal batch rather than flat indices.  ``OXEDataset._sample``
        calls ``__call__`` directly with a pre-cached index for efficiency;
        this method exists so that a ``TemporalSampler`` can also be dropped
        into a plain ``ReplayBuffer``.
        """
        storage_td: TensorDict = getattr(storage, "_storage", storage)
        starts, lengths = self.build_episode_index(storage_td)
        return self(storage_td, starts, lengths, batch_size), {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "delta_timestamps": self.delta_timestamps,
            "control_frequency": self.control_frequency,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass

    def _empty(self) -> None:
        pass

    def dumps(self, path: Any) -> None:
        pass

    def loads(self, path: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # Episode index helpers
    # ------------------------------------------------------------------

    @staticmethod
    def build_episode_index(
        storage_td: TensorDict,
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Build per-episode start/length maps from ``collector/episode_id``.

        Assumes episodes are stored contiguously (guaranteed by
        ``build_combined_storage``).

        Returns:
            episode_starts:  {episode_id: first_flat_index}
            episode_lengths: {episode_id: number_of_steps}
        """
        episode_ids: torch.Tensor = storage_td["collector", "episode_id"]
        unique_ids, counts = torch.unique_consecutive(episode_ids, return_counts=True)
        starts = torch.zeros_like(counts)
        starts[1:] = counts[:-1].cumsum(0)
        episode_starts = {int(tid): int(s) for tid, s in zip(unique_ids, starts)}
        episode_lengths = {int(tid): int(c) for tid, c in zip(unique_ids, counts)}
        return episode_starts, episode_lengths

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def __call__(
        self,
        storage_td: TensorDict,
        episode_starts: Dict[int, int],
        episode_lengths: Dict[int, int],
        batch_size: int,
    ) -> TensorDict:
        """Return a temporally-structured batch of size ``batch_size``.

        All modalities in ``delta_timestamps`` are returned with shape
        ``(B, T, ...)``.  Image modalities in ``image_keys`` are permuted to
        ``(B, T, C, H, W)`` (channels first).  Other modalities keep their
        ``(B, ...)`` shape at the anchor step.

        Boundary handling: offsets outside an episode are clamped to the first /
        last frame of that episode (repeat-pad).

        Args:
            storage_td: Flat TED TensorDict of shape ``(total_steps,)``.
            episode_starts: {episode_id: first_flat_index}
            episode_lengths: {episode_id: number_of_steps}
            batch_size: Number of anchor steps to sample.

        Returns:
            TensorDict with ``batch_size=[batch_size]``.
        """
        total_steps = storage_td.batch_size[0]
        episode_ids = storage_td["collector", "episode_id"]

        # Sample B anchor flat indices uniformly
        anchor_indices = torch.randint(0, total_steps, (batch_size,))

        # Build (B, T) flat-index tensors for each temporal key
        key_flat_indices: Dict[Tuple[str, ...], torch.Tensor] = {}
        for key_tuple, offsets in self._offsets.items():
            T = len(offsets)
            idx = torch.zeros(batch_size, T, dtype=torch.long)
            for b, anchor in enumerate(anchor_indices.tolist()):
                tid = int(episode_ids[anchor].item())
                ep_start = episode_starts[tid]
                ep_len = episode_lengths[tid]
                step = anchor - ep_start
                for t_i, off in enumerate(offsets):
                    clamped = max(0, min(ep_len - 1, step + off))
                    idx[b, t_i] = ep_start + clamped
            key_flat_indices[key_tuple] = idx

        # Anchor batch — non-temporal modalities keep shape (B, ...)
        batch = storage_td[anchor_indices]

        # Override temporal modalities with (B, T, ...) tensors
        for key_tuple, idx_tensor in key_flat_indices.items():
            try:
                node = storage_td
                for k in key_tuple[:-1]:
                    node = node[k]
                leaf_tensor = node[key_tuple[-1]]  # (total_steps, ...)
            except KeyError:
                raise KeyError(
                    f"delta_timestamps key {'/'.join(key_tuple)!r} not found in storage. "
                    f"Top-level keys: {sorted(storage_td.keys())}"
                )
            temporal_data = leaf_tensor[idx_tensor]  # (B, T, ...)

            # Images are stored as (H, W, C); permute to (C, H, W) → (B, T, C, H, W)
            if key_tuple in self.image_keys and temporal_data.ndim >= 4:
                # move last dim (C) to position 2: (B, T, H, W, C) → (B, T, C, H, W)
                perm = (0, 1, temporal_data.ndim - 1) + tuple(range(2, temporal_data.ndim - 1))
                temporal_data = temporal_data.permute(*perm)

            batch[key_tuple] = temporal_data

        return batch
