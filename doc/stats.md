# Dataset statistics

`generate_stats()` returns per-dimension normalization statistics (`mean`, `std`, `min`,
`max`, `q01`, `q99`, `count`) over the episodes a dataset loaded. Available on
`OXEDataset`, `Table30v2Dataset`, `AgiBotWorldBetaDataset` and `LeRobotVideoDataset`
(not `OXEJAXDataset`).

## Concept

- **Per-episode records.** Right after an episode is built (or found in the cache), a
  statistics record is computed from its memmap and stored next to it. Records are
  merged on demand for exactly the episodes you loaded (`episodes=`).
- **Exact moments.** `count`, `mean`, `std` (population std, like `np.std`), `min`, `max`
  come from streaming moments (count / mean / M2) that merge exactly. Accumulation is float64.
- **Estimated quantiles.** `q01` / `q99` come from a seeded uniform sample of rows per
  episode (`stats_sample_size`, default 256, seed = episode id). Sampled rows of
  episode *i* weigh `n_i / k_i`; the merge takes a weighted quantile of the pooled
  samples. With equal weights it equals numpy's `linear` quantile, so when the sample
  covers every row the result is exact. Use `exact=True` for exact quantiles otherwise.
- **Excluded.** `next/`, `collector/`, flags (`done`, `terminated`, `truncated`,
  `reward`, `is_first`, `is_last`, `is_terminal`), image-like paths and non-float
  leaves. Trailing dimensions are flattened (`D = prod(shape[1:])`).
- **Non-finite data.** The episode gets no record and a warning is emitted; dataset
  construction still succeeds, and `generate_stats()` raises naming the episodes.

## API

```python
ds = LeRobotVideoDataset(repo_id="...", episodes=train_ids, batch_size=8)

stats = ds.generate_stats()                    # {"action": {"mean": [...], ..., "count": N}, ...}
stats = ds.generate_stats(keys=["action"])     # subset; unknown key -> ValueError listing valid keys
stats = ds.generate_stats(force=True)          # recompute the per-episode records
stats = ds.generate_stats(exact=True)          # exact q01/q99, rescans the combined memmap
ds.statistics                                  # same as generate_stats()
```

`exact=True` writes nothing. Mean/std/min/max are one streaming pass; quantiles are
computed one dimension at a time (`np.quantile(..., method="linear")`), so peak memory
is `rows * 8` bytes plus one chunk (65 536 rows) of the feature being scanned.

Lower-level functions live in `robotdataset.stats`: `compute_episode_stats`,
`merge_episode_stats`, `EpisodeStats`.

## Progressive growth and train-only statistics

Statistics follow the loading model. Statistics over the train split only:

```python
ds = LeRobotVideoDataset(repo_id="...", episodes=train_ids, batch_size=8)
```

Adding episodes is a longer `episodes` list; only the new episodes are computed:

```python
ds = LeRobotVideoDataset(repo_id="...", episodes=train_ids + more_ids, batch_size=8)
```

Episodes cached before the feature existed get records on first load, without rebuilding
or re-downloading anything.

## On disk

Inside each episode directory (`<cache>/.../episodes/<id>/`): `stats.json` (version,
row count, schema, moments) and `stats_sample.npz` (sampled rows). Both are written
atomically. A file with an unknown version, or one whose row count/schema no longer
matches the episode, is ignored and recomputed. To force a recompute call
`generate_stats(force=True)` or delete those two files. If the cache is read-only,
statistics are kept in memory with a warning.

## GR00T export

`robotdataset.gr00t.to_gr00t_statistics` reshapes the result into
`{embodiment_tag: {"state": {group: {stat: [...]}}, "action": {...}}}`:

```python
from robotdataset.gr00t import to_gr00t_statistics

g = to_gr00t_statistics(
    ds.generate_stats(),
    embodiment_tag="new_embodiment",
    action_groups={"arm": (0, 7), "gripper": (7, 8)},   # or modality_meta=<meta/modality.json>
)
processor.set_statistics(g, override=True)
```

State groups default to every leaf under `observation/state/` (`all` is dropped when
slices exist). `count` is dropped. Use `override=True`: with `override=False` the
processor silently keeps any statistics already stored for that embodiment tag.
`check_against(g, modality_configs)` compares group names to a GR00T modality config.
Relative-action statistics are not computed yet; see
[relative_action_stats.md](relative_action_stats.md).

## Sample-size benchmark (synthetic)

100 synthetic episodes of 200-1200 rows, 8 dims, heavy-tailed (Student-t, df=4) plus drift.
Error is the max over dimensions of |q - q_exact| / std. Hook cost is ~0.3-0.4 ms per
episode at this size; merging takes 0.002 s (10 episodes), 0.02 s (100), 0.4 s (1000).

| `stats_sample_size` | 64 | 128 | 256 | 512 | 1024 |
|---|---|---|---|---|---|
| max q01/q99 error / std | 0.163 | 0.136 | 0.075 | 0.022 | 0.013 |

256 stays the default. These are synthetic numbers; validation on a real LeRobot dataset
(comparison with `exact=True` and the dataset's own `meta/stats.json`) is still open.
