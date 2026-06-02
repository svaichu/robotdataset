from __future__ import annotations

from pathlib import Path
from typing import Any, Collection, Dict, List, Tuple

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers.samplers import Sampler
from torchrl.data.replay_buffers.storages import Storage

from robotdataset.oxe.temporal_sampler import TemporalSampler


class EpisodeNthSecondSampler(Sampler):
    """Samples one full episode, returning a data point for every nth second.

    Given an episode picked uniformly at random, iterates from the first step
    and selects every step whose index is a multiple of ``round(n *
    control_frequency)``.  The result is a single TensorDict with
    ``batch_size=[T_sub]`` where ``T_sub`` is the number of selected steps.

    Image modalities (identified by ``image_keys``) are permuted from the
    on-disk HWC layout to channel-first CHW.

    Args:
        n: Interval in seconds between sampled steps.
        control_frequency: Steps per second used to convert ``n`` to a step
            stride.  Default: 10.
        image_keys: Tuple-path keys whose tensors are stored in HWC layout and
            must be permuted to CHW after selection.
    """

    def __init__(
        self,
        n: float,
        control_frequency: float = 10.0,
        image_keys: Collection[Tuple[str, ...]] = (),
    ) -> None:
        self.n = n
        self.control_frequency = control_frequency
        self.image_keys = set(image_keys)
        # stride in steps (minimum 1)
        self._stride: int = max(1, round(n * control_frequency))

    # ------------------------------------------------------------------
    # Sampler ABC
    # ------------------------------------------------------------------

    @property
    def ran_out(self) -> bool:
        return False

    def sample(self, storage: Storage, batch_size: int) -> Tuple[TensorDict, dict]:
        """Pick one random episode and return its nth-second sub-sampled steps.

        ``batch_size`` is accepted for API compatibility but ignored — the
        returned batch size equals the number of selected steps in the episode.
        """
        storage_td: TensorDict = getattr(storage, "_storage", storage)
        starts, lengths = TemporalSampler.build_episode_index(storage_td)
        return self(storage_td, starts, lengths), {}

    def state_dict(self) -> Dict[str, Any]:
        return {
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
        """Return every nth-second step from one randomly chosen episode.

        Args:
            storage_td: Flat TED TensorDict of shape ``(total_steps,)``.
            episode_starts: {episode_id: first_flat_index}
            episode_lengths: {episode_id: number_of_steps}
            batch_size: Accepted for interface compatibility with
                :class:`TemporalSampler`; ignored here since the returned
                batch size is determined by the episode length and stride.

        Returns:
            TensorDict with ``batch_size=[T_sub]`` containing the selected steps.
        """
        episode_id = int(torch.randint(0, len(episode_starts), (1,)).item())
        ep_keys = list(episode_starts.keys())
        chosen_id = ep_keys[episode_id]

        ep_start = episode_starts[chosen_id]
        ep_len = episode_lengths[chosen_id]

        # Indices within the episode at every nth second
        local_indices = list(range(0, ep_len, self._stride))
        flat_indices = torch.tensor(
            [ep_start + i for i in local_indices], dtype=torch.long
        )

        batch = storage_td[flat_indices]

        # Permute image tensors from HWC → CHW
        for key_tuple in self.image_keys:
            try:
                data = batch[key_tuple]
            except KeyError:
                continue
            if data.ndim >= 3:
                # (T, H, W, C) → (T, C, H, W)
                perm = (0, data.ndim - 1) + tuple(range(1, data.ndim - 1))
                batch[key_tuple] = data.permute(*perm).contiguous()

        return batch
