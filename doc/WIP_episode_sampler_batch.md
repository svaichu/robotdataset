# WIP: EpisodeNthSecondSampler — Batch Size & Episode-Length Trajectory Design

## Problem

`EpisodeNthSecondSampler` returns `floor(ep_len / stride)` steps per call — variable by
episode. The `ReplayBuffer` machinery expects every sample to return exactly `batch_size`
elements. Currently `batch_size` is accepted in `__call__` but silently ignored.

## Options considered

| Option | Description | Pro | Con |
|--------|-------------|-----|-----|
| Truncate to `batch_size` | Keep first `batch_size` nth-second indices | Fits ReplayBuffer | Loses tail; arbitrary cutoff defeats coarse-sweep intent |
| `max_steps` cap | New sampler param, ignore `batch_size` | User controls seq len explicitly | Still variable if episode < max_steps |
| Pad + mask | Pad to `max_steps`, return mask tensor | Fixed size; works for transformers | TED has no canonical mask field; model must handle masking |
| `batch_size` = num episodes | Sample N episodes, stack with padding | Matches DT/RT-2 style | Same masking problem; breaks ReplayBuffer contract |

## Decision (pending)

- [ ] Add `max_steps: int | None = None` parameter to `EpisodeNthSecondSampler`
  - If set, truncate nth-second indices to the first `max_steps` frames
  - Document that this sampler is best used for evaluation/inspection, not i.i.d. training
- [ ] Keep `batch_size` ignored; document it clearly in docstring
- [ ] For whole-episode training: recommend `Dataset` + `DataLoader` + `collate_fn` pattern
  instead of `ReplayBuffer` (see below)

## Recommended pattern for whole-episode training

```python
class EpisodeDataset(torch.utils.data.Dataset):
    """One episode per __getitem__, sub-sampled at nth-second intervals."""
    ...

def collate_episodes(batch):
    # pad to max_len in this batch, return mask
    max_len = max(ep.batch_size[0] for ep in batch)
    ...

loader = DataLoader(EpisodeDataset(...), batch_size=8, collate_fn=collate_episodes)
```

This is the right abstraction for transformer-style models (Decision Transformer, RT-2,
OpenVLA) that consume full episode context.

## Files affected

- `robotdataset/oxe/episode_nth_second_sampler.py` — add `max_steps`, update docstring
- `doc/OXE.md` — note sampler limitations and recommended use
- (new) `robotdataset/oxe/episode_dataset.py` — optional whole-episode Dataset wrapper
