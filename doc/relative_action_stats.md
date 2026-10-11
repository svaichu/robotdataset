# Relative-action statistics

A policy that predicts actions relative to the current state (GR00T with
`use_relative_action=True`) normalizes those differences, so it needs statistics of the
*relative* chunks. They cannot be derived from the absolute-action statistics.
`generate_relative_action_stats()` computes them from the loaded episodes; the
definition follows GR00T's `gr00t/data/stats.py` (`RelativeActionLoader`) without
importing it.

```python
rel = ds.generate_relative_action_stats(
    groups={"eef": (0, 6)},                              # columns of `action`
    state_keys={"eef": "observation/state/cartesian"},   # reference state, same width
    horizon=16,                                          # action chunk length
    representation="xyz_rotvec",                         # or "joint"; str or {group: str}
    exact=True,                                          # exact q01/q99 (recommended for training)
)
# {"eef": {"mean", "std", "min", "max", "q01", "q99": (16, 6) nested lists, "count": n_chunks}}

g = to_gr00t_statistics(ds.generate_stats(exact=True), "new_embodiment",
                        modality_meta=modality_json, relative_action=rel)
processor.set_statistics(g, override=True)
```

## Definition

- **Chunks.** An episode of `n` rows has `n - horizon + 1` chunks (start `i = 0 .. n - horizon`).
  Chunks never run past the end of the episode and are never padded; episodes shorter than
  the horizon contribute nothing. Chunk `i` holds action rows `i .. i + horizon - 1`.
- **Reference.** The state at the chunk's first step, `state[i]` (GR00T asserts that the
  state's last delta index equals the action's first one, i.e. reference state and first
  action are the same time step).
- **`"joint"`.** `action - reference`, per dimension.
- **`"xyz_rotvec"`.** Six columns, translation + rotation vector. Each action pose is
  expressed in the reference frame, `T_ref^-1 @ T_action`, and returned as translation
  `R_ref^T (p - p_ref)` + rotation vector of `R_ref^T R` (angle in `[0, pi]`).
  Implemented in numpy (`rotvec_to_matrix`, `matrix_to_rotvec`); no scipy needed.
- **Aggregation.** `mean`, `std` (population), `min`, `max`, `q01`, `q99` over all chunks of
  all loaded episodes, separately for each horizon step and dimension, so each statistic has
  shape `(horizon, D)`; `to_gr00t_statistics(relative_action=...)` keeps that shape.

Not supported: chunks with arbitrary `delta_indices` (the horizon is `0 .. horizon - 1`),
`rot6d`/matrix poses, and delta (step-to-step) actions.

## Per-episode records

Like the absolute statistics, each episode keeps a record next to its memmap, so it is
computed once and only for new episodes when the selection grows:

```
<episode>/stats_rel_<fingerprint>.json
<episode>/stats_rel_<fingerprint>_sample.npz
```

The fingerprint covers `(group, columns, state_key, representation, horizon, action_key)`, so
changing any of them writes a new record instead of reusing a stale one. A record is a
`robotdataset.stats.EpisodeStats` with a single signal `relative_action/<group>` of
`horizon * D` columns whose `num_rows` is the number of chunks: exact moments (merged with
the same Chan update) plus a seeded uniform sample of whole chunks (`stats_sample_size`, default 256).
Sampling whole chunks keeps the horizon axis consistent.

`mean`/`std`/`min`/`max` are exact. `q01`/`q99` are estimates from the samples;
`exact=True` rebuilds every chunk and takes numpy's `linear` quantiles. It writes nothing
and needs about `chunks * horizon * D * 8` bytes (about 9 MB for 11k chunks of 16x6).

## Validation (`stack_without_ft_tact_v4`, 51 episodes, 11,479 chunks of 16x6)

Reference: float64 statistics from GR00T's own `EndEffectorPose` / `EndEffectorActionChunk`
applied to the parquet data. Errors are the maximum over the 96 entries, divided by the
reference std.

| source | mean | std | min / max | q01 / q99 |
|---|---|---|---|---|
| `generate_relative_action_stats(exact=True)` | 2e-15 | 5e-16 | 4e-15 | 4e-15 |
| `generate_relative_action_stats()` (sampled) | 5e-14 | 5e-15 | 4e-15 | 4.1e-2 / 2.8e-2 |
| GR00T `meta/relative_stats.json` | 3.9e-5 | 3.8e-6 | 5e-7 | 3.5e-7 / 2.0e-7 |

GR00T's file is the float32 pipeline (chunks are cast to float32 before the statistics).
After percentile normalization (range 2.0) the sampled quantiles move a value by at most
0.028, the exact ones by 1e-14 and GR00T's file by 2e-6. Computing the records takes about
1 s for the 51 episodes, so use `exact=True` for training.
