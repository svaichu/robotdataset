# Overview & TED Format

## Scope of the library

`robotdataset` solves a recurring problem in robot-learning research: every dataset
collection ships in a different on-disk format (RLDS/TFDS shards on GCS, parquet on
HuggingFace, TAR archives of MP4 videos, …), yet a training pipeline wants a single,
uniform way to sample batches.

The library normalizes all sources into one target representation:

> **Every dataset becomes a flat sequence of per-step transitions in TorchRL TED
> format, persisted as memory-mapped tensors on local disk.**

Each supported source gets its own loader class, but they all share the same
pipeline and the same sampling API:

```
 source (GCS / HF Hub / MP4 in TAR)
        │  download only what is requested
        ▼
 per-episode TED memmaps        ~/.cache/robotdataset/.../episodes/<idx>/
        │  concatenate selected episodes
        ▼
 combined TED memmap            ~/.cache/robotdataset/.../combined/<hash>/
        │  loaded lazily (TensorDict.load_memmap)
        ▼
 TemporalSampler → batch        TensorDict with batch_size=[B]
```

Because both stages are cached and keyed by content, subsequent runs skip download
and conversion entirely. Peak RAM during conversion is one episode; during training
it is one batch — the combined storage stays on disk and is read through memmaps.

## The TED format

TED (TorchRL Episode Data) is TorchRL's canonical layout for offline RL data: a flat
`TensorDict` of shape `(total_steps,)` where each row is one transition. All loaders
in this library produce exactly this structure:

```
TensorDict({
    "observation":  TensorDict({...}),   # step[t] modalities (nested, e.g. image, state)
    "action":       Tensor,
    "done":         Tensor(bool),
    "terminated":   Tensor(bool),
    "next": TensorDict({
        "observation": TensorDict({...}), # step[t+1] observation (copy of t for last step)
        "reward":      Tensor,
        "done":        Tensor(bool),
        "terminated":  Tensor(bool),
    }),
    "collector": TensorDict({
        "episode_id": Tensor(int64),      # which episode each row belongs to
    }),
})
```

Key properties:

- **Flat step storage** — episodes are concatenated back-to-back; episode membership
  is recoverable from `collector/episode_id`. Episodes are always stored
  contiguously, which is what lets samplers rebuild episode boundaries cheaply.
- **`next` semantics** — for the last step of an episode, `next/observation` is a
  copy of the current observation and `done` is `True`.
- **Text fields** (e.g. `language_instruction`) are carried as non-tensor data and
  appear in batches as `NonTensorStack` of Python strings.
- **Images are stored HWC** (`H, W, C`, uint8) on disk; samplers can permute to
  channel-first CHW at sampling time (see [Samplers](samplers.md)).

Because the result is a standard TED `TensorDict`, batches plug directly into
TorchRL losses, transforms, and replay-buffer tooling, the same way D4RL or Minari
datasets do.

## Common dataset API

> **Only `OXEDataset` is usable today** (alpha, not stable). `Table30v2Dataset`,
> `AgiBotWorldBetaDataset`, and `OXEJAXDataset` are in development.

The Torch-path dataset classes (`OXEDataset`, `Table30v2Dataset`,
`AgiBotWorldBetaDataset`) all inherit from TorchRL's
`BaseDatasetExperienceReplay`, which provides:

- **Immutability** — an `ImmutableDatasetWriter` prevents accidental writes.
- **`sample()`** — returns a `TensorDict` batch of `batch_size` transitions.
- **Iteration** — `next(iter(dataset))` yields batches.
- **`preprocess()`** — parallelized transform pipeline saving to a new memmap.
- **`delete()`** — clears the cached memmap from disk.
- **`data_path` / `data_path_root`** — standardized cache-path interface.

Shared members across all dataset classes:

| Member | Description |
|---|---|
| `num_episodes` | Number of episodes loaded |
| `len(dataset)` | Total steps across loaded episodes |
| `image_keys` | Tuple-paths of image modalities (for HWC→CHW permutation) |
| `modalities` / `get_modalities()` | Modality discovery (paths, kinds, dtypes, shapes) |
| `set_sampler(sampler)` | Swap the temporal sampler without rebuilding the dataset |

Every dataset has a `TemporalSampler` active by default, configured as
`{modality: [0.0]}` — i.e. each sample is just the anchor step (`T = 1`). Pass
`delta_timestamps` to the constructor, or call `set_sampler()`, to get temporal
windows.

## Modality inference

Each leaf field in the data is classified into a *kind* based on its path name and
value type:

| Kind | Matched by path tokens |
|---|---|
| `image` | image, camera, rgb, video, frame, color, fisheye, depth |
| `text` | language, instruction, text, caption, prompt |
| `action` | action, policy, torque, velocity |
| `state` | observation, state, proprio, joint, pose, ee |

`dataset.modalities` groups paths by kind; `dataset.image_keys` derives the keys
that need HWC→CHW handling; text kinds are excluded from temporal sampling.

```python
dataset.modalities
# {'action': ['action'],
#  'image':  ['observation/image', 'observation/finger_vision_1'],
#  'state':  ['observation/state'],
#  'text':   ['language_instruction', 'language_embedding']}
```

## Cache directory

All loaders resolve their cache root in the same priority order:

1. the `root` constructor argument,
2. the `ROBOTDATASET_CACHE` environment variable,
3. `~/.cache/robotdataset` (default).

Layout under the root:

```
~/.cache/robotdataset/
├── oxe/<dataset_name>[/<version>]/
│   ├── dataset_info.json …            # TFDS metadata
│   ├── episodes/<split>/<idx>/        # per-episode TED memmaps
│   ├── combined/<split>/<hash>/       # combined memmap (Torch path)
│   └── combined_jax/<split>/<hash>/   # combined NumPy store (JAX path)
└── hf/
    ├── table30v2/<split>/{episodes,combined}/
    └── agibotworld-beta/<split>/{episodes,combined}/
```

The `<hash>` is an MD5 digest of the sorted selected-episode list, so each distinct
episode selection gets its own combined storage while per-episode conversions are
shared between selections.
