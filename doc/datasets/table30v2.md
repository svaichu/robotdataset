# Table30 v2 — `Table30v2Dataset`

> **Status: alpha.** Functional end-to-end; API may change without notice.

**Requires:** `datasets` and `huggingface_hub` (install with `pip install "robotdataset[hf]"`).

Loads the [`RoboChallenge/Table30v2`](https://huggingface.co/datasets/RoboChallenge/Table30v2)
dataset from HuggingFace Hub and exposes it as a TorchRL
`BaseDatasetExperienceReplay` in TED format. The HuggingFace `datasets` library
handles download and HF-side caching; episodes are then converted to per-episode
TED memmaps and combined exactly as in the [OXE pipeline](../overview.md#cache-directory).

## Usage

```python
from robotdataset import Table30v2Dataset

# Load three episodes
ds = Table30v2Dataset(episodes=[0, 1, 2], batch_size=32)

# Or restrict by task: only episodes belonging to the listed task IDs
ds = Table30v2Dataset(tasks=[3, 7], batch_size=32)

batch = ds.sample()
print(ds.num_episodes)   # 3
print(len(ds))           # total steps
```

### Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `split` | `"train"` | Dataset split |
| `tasks` | all | Task IDs to include; episodes are filtered to these tasks |
| `episodes` | all (within tasks) | Episode indices to include |
| `batch_size` | `32` | Transitions per `sample()` |
| `root` | cache default | Override cache root |
| `delta_timestamps` | `{key: [0.0]}` | Temporal window per modality, in seconds |
| `control_frequency` | `10.0` | Steps per second for delta→offset conversion |

## Sampling and inspection

The sampling API is shared with all Torch-path datasets — a `TemporalSampler` is
always active, defaulting to anchor-only (`T=1`) windows; see
[Samplers](../samplers.md).

```python
ds.modalities            # {path: spec} inferred from the built TED storage
ds.image_keys            # frozenset of tuple-paths for HWC image modalities
ds.get_modalities()      # same as ds.modalities, as a plain dict
ds.set_sampler(sampler)  # swap temporal sampler without rebuilding
```

Note: unlike OXE (whose modalities come from TFDS metadata), Table30v2 modalities
are inferred **from the built storage**, so `ds.modalities` here maps
`path → spec dict` rather than grouping by kind.

Cache location: `~/.cache/robotdataset/hf/table30v2/<split>/{episodes,combined}/`.

Example notebook: [`example/example_table30v2.ipynb`](../../example/example_table30v2.ipynb)
