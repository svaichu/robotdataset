from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Dict, List, Tuple

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

    The ``next`` field mirrors the observation window across the current step:
    if observation deltas are ``[-0.2, -0.1, 0.0]``, the next-field deltas are
    ``[0.0, 0.1, 0.2]``.  Both share the anchor step at offset 0.  The
    next-field values are read from the same storage leaf as the corresponding
    observation (not from the TED ``next/…`` field), using positive step offsets
    clamped to the episode boundary.

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
        image_keys: Collection[Tuple[str, ...]] = (),
    ) -> None:
        self.delta_timestamps = delta_timestamps
        self.control_frequency = control_frequency
        # Accept "/" strings or tuples; store as tuples for nested storage indexing
        self.image_keys = {
            tuple(k.split("/")) if isinstance(k, str) else k for k in image_keys
        }
        # Pre-compute integer step offsets keyed by tuple path
        self._offsets: Dict[Tuple[str, ...], List[int]] = {
            tuple(k.split("/")): [round(dt * control_frequency) for dt in deltas]
            for k, deltas in delta_timestamps.items()
        }
        # Mirror offsets for the next field: negate and sort ascending so that
        # obs offsets [-2, -1, 0] → next offsets [0, 1, 2].
        self._next_offsets: Dict[Tuple[str, ...], List[int]] = {
            key: sorted(-off for off in offsets)
            for key, offsets in self._offsets.items()
        }

    # ------------------------------------------------------------------
    # Sampler ABC
    # ------------------------------------------------------------------

    @property
    def ran_out(self) -> bool:
        return False

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Dict[str, Any], dict]:
        """Temporal sample called by ReplayBuffer machinery."""
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
    # Helpers
    # ------------------------------------------------------------------

    def _build_flat_indices(
        self,
        anchor_indices: torch.Tensor,
        offsets_map: Dict[Tuple[str, ...], List[int]],
        episode_ids: torch.Tensor,
        episode_starts: Dict[int, int],
        episode_lengths: Dict[int, int],
        batch_size: int,
    ) -> Dict[Tuple[str, ...], torch.Tensor]:
        """Build ``(B, T)`` flat-index tensors for each key/offsets pair.

        Offsets outside the episode are clamped to the first/last frame
        (repeat-pad).
        """
        key_flat_indices: Dict[Tuple[str, ...], torch.Tensor] = {}
        for key_tuple, offsets in offsets_map.items():
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
        return key_flat_indices

    @staticmethod
    def _read_leaf(storage_td: TensorDict, key_tuple: Tuple[str, ...]) -> torch.Tensor:
        """Navigate to a leaf tensor in storage, raising KeyError with context on miss."""
        try:
            node = storage_td
            for k in key_tuple[:-1]:
                node = node[k]
            return node[key_tuple[-1]]
        except KeyError:
            raise KeyError(
                f"delta_timestamps key {'/'.join(key_tuple)!r} not found in storage. "
                f"Top-level keys: {sorted(storage_td.keys())}"
            )

    def _apply_image_permutation(
        self, data: torch.Tensor, key_tuple: Tuple[str, ...]
    ) -> torch.Tensor:
        """Permute HWC → CHW for image tensors: ``(B, T, H, W, C) → (B, T, C, H, W)``."""
        if key_tuple in self.image_keys and data.ndim >= 4:
            perm = (0, 1, data.ndim - 1) + tuple(range(2, data.ndim - 1))
            return data.permute(*perm).contiguous()
        return data

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
        ``(B, T, ...)``.  The corresponding ``next`` field is populated with the
        mirrored temporal window (positive offsets from the anchor).  Image
        modalities in ``image_keys`` are permuted to ``(B, T, C, H, W)``
        (channels first) in both observation and next fields.

        Boundary handling: offsets outside an episode are clamped to the first /
        last frame of that episode (repeat-pad).

        Args:
            storage_td: Flat TED TensorDict of shape ``(total_steps,)``.
            episode_starts: {episode_id: first_flat_index}
            episode_lengths: {episode_id: number_of_steps}
            batch_size: Number of anchor steps to sample.

        Returns:
            Flat TensorDict with ``"/"``-separated string keys and temporal
            modalities having shape ``(B, T, ...)``.
        """
        total_steps = storage_td.batch_size[0]
        episode_ids = storage_td["collector", "episode_id"]

        # Sample B anchor flat indices uniformly
        anchor_indices = torch.randint(0, total_steps, (batch_size,))

        # (B, T) flat-index tensors for obs and next-field
        obs_flat = self._build_flat_indices(
            anchor_indices, self._offsets, episode_ids,
            episode_starts, episode_lengths, batch_size,
        )
        next_flat = self._build_flat_indices(
            anchor_indices, self._next_offsets, episode_ids,
            episode_starts, episode_lengths, batch_size,
        )

        # Anchor batch — non-temporal modalities keep shape (B, ...)
        batch = storage_td[anchor_indices]

        # Override obs temporal modalities with (B, T, ...) tensors
        for key_tuple, idx_tensor in obs_flat.items():
            leaf = self._read_leaf(storage_td, key_tuple)
            temporal = self._apply_image_permutation(leaf[idx_tensor], key_tuple)
            batch[key_tuple] = temporal

        # Populate next-field temporal modalities with mirrored (B, T, ...) tensors.
        # Source: same storage leaf as obs (e.g. observation/image read with +offsets),
        # destination: ("next", *key_tuple) in the batch.
        for key_tuple, idx_tensor in next_flat.items():
            leaf = self._read_leaf(storage_td, key_tuple)
            temporal = self._apply_image_permutation(leaf[idx_tensor], key_tuple)
            batch[("next",) + key_tuple] = temporal

        return batch.flatten_keys("/")
