# Design: relative-action statistics (spike, not implemented)

GR00T can train on actions expressed relative to the current state. Its normalization
statistics are then for those differences and cannot be derived from absolute-action
statistics. Without them `StateActionProcessor` raises `ValueError` when
`use_relative_action=True` and a relative action key has no `relative_action` stats.

## GR00T's definition (from `state_action_processor.py` and `stats.py`, main branch)

- **Reference state.** `state[state_key][-1]` where `state_key` is the action config's
  `state_key`, else the action key itself. In the stats loader, for chunk start `i`
  the reference is the state at `state_delta_indices[-1] + i`; it is asserted that this
  index equals the action's first delta index (reference state and first action are the
  same time step).
- **Horizon.** Each chunk takes action rows at `action_delta_indices + i`; all rows of a
  chunk are converted against the same reference frame.
- **Edges.** Start positions number `len(episode) - action_delta_indices[-1]`; there is
  no padding, so chunks that would run past the episode end are dropped.
- **Transform.** Joint actions: `JointActionChunk.relative_chunking` against
  `JointPose(reference)`. EEF actions: relative to an `EndEffectorPose` frame.
  (The subtraction itself lives in `action_chunking.py`; verify before implementing.)
- **Aggregation.** All chunks of all episodes are concatenated; mean/std/min/max/q01/q99
  are taken with `axis=0`, giving statistics per horizon step and dimension
  (shape `(horizon, D)`), not per dimension only. The `export` layout must therefore
  allow a per-horizon vector or a documented reduction.
- GR00T caches `relative_stats.json` with a per-key fingerprint of the modality config.

## Proposal

**API.** Explicit arguments rather than reusing `delta_timestamps` (the sampler's
windows are for observations and clamp at episode ends, which GR00T's statistics do not):

```python
rel = ds.generate_relative_action_stats(
    groups={"arm": (0, 7)},
    state_keys={"arm": "observation/state/joints"},
    horizon=16,
    representation="joint",     # "eef" deferred
)
to_gr00t_statistics(stats, ..., relative_action=rel)
```

**Per-episode record.** Stored next to the absolute record, keyed by a fingerprint of
`(horizon, group ranges, state key mapping, representation)`, e.g.
`stats_rel_<fingerprint>.json/.npz`. Unlike absolute records, a row here is a
*chunk start*, so `num_rows = len(episode) - horizon + 1` (matching GR00T's edge rule)
and windows never cross episode boundaries by construction (they are built per episode).

**Merge.** Same scheme: exact moments per `(horizon step, dim)` over chunk starts, and a
weighted sample of chunk starts (weight `n_i / k_i`) for q01/q99. Sampling chunks, not
independent rows, keeps the horizon axis consistent.

## Open points before implementation

1. Verify `relative_chunking` arithmetic (and rotation handling for EEF) in `action_chunking.py`.
2. Decide the export reduction for the `(horizon, D)` statistics vs. GR00T's expected
   `[floats]` (check how `_compute_normalization_parameters` indexes them).
3. Confirm the horizon convention against the sampler's `control_frequency`.

## Follow-up issues

- Implement `generate_relative_action_stats` for joint actions (records, merge, tests).
- Wire `relative_action=` in `to_gr00t_statistics` (placeholder exists).
- EEF representation.
