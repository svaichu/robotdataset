from __future__ import annotations

from typing import Any, Collection, Dict, List, Tuple

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import Storage

from robotdataset.oxe.temporal_sampler import TemporalSampler


class EpisodeTubeletSampler(Sampler):
    """Samples a fixed grid of tubelets from one randomly chosen episode.

    Divides the episode into ``batch_size`` evenly-spaced clips.  Each clip
    contains ``tubelet_size`` frames sampled every ``n`` seconds (stride =
    ``round(n * control_frequency)`` steps).  The output shape is
    ``(batch_size, tubelet_size, *data_dims)``.

    Clip anchor placement:
    - ``batch_size == 1``: single clip anchored at the **end** of the episode.
    - ``batch_size >= 2``: first clip at episode start, last clip at episode end,
      remaining clips evenly spaced in between.
    - When the episode is shorter than a full clip the frame indices are clamped
      to the episode boundary (repeat-pad at end).

    Args:
        batch_size: Number of clips (T_ep axis).
        tubelet_size: Frames per clip (T_clip axis).
        n: Seconds between consecutive frames within a clip.
        control_frequency: Steps per second used to convert ``n`` to a step
            stride.  Default: 10.
        image_keys: Tuple-path keys stored in HWC layout that should be
            permuted to CHW after gathering, yielding
            ``(batch_size, tubelet_size, C, H, W)``.
    """

    def __init__(
        self,
        batch_size: int,
        tubelet_size: int,
        n: float,
        control_frequency: float = 10.0,
        image_keys: Collection[Tuple[str, ...]] = (),
    ) -> None:
        self.batch_size = batch_size
        self.tubelet_size = tubelet_size
        self.n = n
        self.control_frequency = control_frequency
        # Accept "/" strings or tuples; store as tuples for nested storage indexing
        self.image_keys = {
            tuple(k.split("/")) if isinstance(k, str) else k for k in image_keys
        }
        self._stride: int = max(1, round(n * control_frequency))

    # ------------------------------------------------------------------
    # Sampler ABC
    # ------------------------------------------------------------------

    @property
    def ran_out(self) -> bool:
        return False

    def sample(self, storage: Storage, batch_size: int) -> Tuple[Dict[str, Any], dict]:
        """Called by ReplayBuffer machinery; ``batch_size`` is ignored."""
        storage_td: TensorDict = getattr(storage, "_storage", storage)
        starts, lengths = TemporalSampler.build_episode_index(storage_td)
        return self(storage_td, starts, lengths), {}

    def state_dict(self) -> Dict[str, Any]:
        return {
            "batch_size": self.batch_size,
            "tubelet_size": self.tubelet_size,
            "n": self.n,
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
    # Sampling
    # ------------------------------------------------------------------

    def __call__(
        self,
        storage_td: TensorDict,
        episode_starts: Dict[int, int],
        episode_lengths: Dict[int, int],
        batch_size: int = 0,
    ) -> TensorDict:
        """Return a ``(batch_size, tubelet_size, *data_dims)`` flat TensorDict.

        Args:
            storage_td: Flat TED TensorDict of shape ``(total_steps,)``.
            episode_starts: {episode_id: first_flat_index}
            episode_lengths: {episode_id: number_of_steps}
            batch_size: Accepted for interface compatibility; ignored — clip
                count is controlled by the ``batch_size`` init parameter.
        """
        # Pick one episode at random
        ep_keys = list(episode_starts.keys())
        chosen_id = ep_keys[int(torch.randint(0, len(ep_keys), (1,)).item())]
        ep_start = episode_starts[chosen_id]
        ep_len = episode_lengths[chosen_id]

        B = self.batch_size
        T = self.tubelet_size
        stride = self._stride

        # Rightmost valid anchor: last frame of clip (anchor + (T-1)*stride)
        # must not exceed ep_len - 1.
        a_max = max(0, ep_len - 1 - (T - 1) * stride)

        # Anchor positions (local to episode start)
        if B == 1:
            anchors = [a_max]
        else:
            anchors = [round(i * a_max / (B - 1)) for i in range(B)]

        # Build (B, T) flat index tensor
        flat_indices = torch.zeros(B, T, dtype=torch.long)
        for b, anchor in enumerate(anchors):
            for t in range(T):
                local = min(ep_len - 1, anchor + t * stride)
                flat_indices[b, t] = ep_start + local

        # Index storage → (B, T, *data_dims)
        batch = storage_td[flat_indices]

        # Permute image tensors HWC → CHW: (B, T, H, W, C) → (B, T, C, H, W)
        for key_tuple in self.image_keys:
            try:
                data = batch[key_tuple]
            except KeyError:
                continue
            if data.ndim >= 4:
                perm = (0, 1, data.ndim - 1) + tuple(range(2, data.ndim - 1))
                batch[key_tuple] = data.permute(*perm).contiguous()

        return batch.flatten_keys("/")
