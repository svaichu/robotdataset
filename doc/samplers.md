# Samplers

Samplers turn the flat TED storage (shape `(total_steps,)`) into structured training
batches. Three samplers are provided:

| Sampler | Output shape | Backend | Use case |
|---|---|---|---|
| [`TemporalSampler`](#temporalsampler) | `(B, T, ...)` per modality | torch / TensorDict | History/future windows around random anchor steps |
| [`EpisodeTubeletSampler`](#episodetubeletsampler) | `(B, T, ...)` from one episode | torch / TensorDict | Evenly-spaced clips ("tubelets") across a single episode |
| [`JAXTemporalSampler`](#jaxtemporalsampler) | `(B, T, ...)` per modality | NumPy → JAX | Same as `TemporalSampler` for the JAX path |

All three implement the TorchRL `Sampler` interface (the JAX one mirrors it), so
they can be passed directly as a replay-buffer sampler. All of them respect episode
boundaries: offsets that run past the start or end of an episode are **clamped to
the boundary frame** (repeat-pad) — windows never bleed into a neighbouring episode.

A `TemporalSampler` is always active on every dataset. The default configuration is
`{modality: [0.0]}` for every tensor (non-text) modality, i.e. anchor-only, `T = 1`.
Swap it at any time without rebuilding the dataset:

```python
dataset.set_sampler(my_sampler)
```

---

## TemporalSampler

```python
from robotdataset import TemporalSampler
```

Samples `batch_size` random anchor steps uniformly across the storage, then gathers
each modality listed in `delta_timestamps` at the requested time offsets.

### Arguments

| Argument | Default | Description |
|---|---|---|
| `delta_timestamps` | required | `{slash-separated path: [offsets in seconds]}`, e.g. `{"observation/image": [-0.2, -0.1, 0.0]}` |
| `control_frequency` | `10.0` | Steps per second. Offsets are converted to integer step offsets via `round(dt * control_frequency)` |
| `image_keys` | `()` | Tuple-paths stored as HWC that should be permuted to CHW after stacking |

### Example

```python
import numpy as np
from robotdataset import TemporalSampler

sampler = TemporalSampler(
    delta_timestamps={
        "observation/image": np.arange(-1.0, 0.1, 0.1).tolist(),  # 10-frame history
        "observation/state": [-0.1, 0.0],
        "action":            [0.0, 0.1, 0.2],                     # action chunk
    },
    control_frequency=10,
    image_keys=dataset.image_keys,   # optional: HWC → CHW
)
dataset.set_sampler(sampler)

batch = dataset.sample()
batch["observation"]["image"].shape   # (B, 10, C, H, W)  — CHW because image_keys set
batch["observation"]["state"].shape   # (B, 2, state_dim)
batch["action"].shape                 # (B, 3, action_dim)
```

### The mirrored `next` window

For every key in `delta_timestamps`, the `next` field is populated with the
observation window **mirrored across the anchor step**: offsets are negated and
sorted, so observation deltas `[-0.2, -0.1, 0.0]` produce next-field deltas
`[0.0, 0.1, 0.2]`. Both windows share the anchor at offset 0. The values are read
from the same storage leaf as the observation (using positive step offsets), not
from the stored TED `next/…` field.

```python
batch["next"]["observation"]["image"].shape   # same (B, T, ...) shape, future window
```

Modalities *not* listed in `delta_timestamps` keep their plain anchor shape
`(B, ...)`, and non-tensor fields (language strings) come along untouched.

---

## EpisodeTubeletSampler

```python
from robotdataset import EpisodeTubeletSampler
```

Picks **one episode at random** and divides it into `batch_size` evenly-spaced
clips. Each clip contains `tubelet_size` frames sampled every `n` seconds
(stride = `round(n * control_frequency)` steps). Output is a TensorDict of shape
`(batch_size, tubelet_size, *data_dims)`. Useful for video-model training (e.g.
V-JEPA-style tubelets) where you want coverage of a whole episode rather than
random transitions.

### Arguments

| Argument | Description |
|---|---|
| `batch_size` | Number of clips per sample (the clip-grid axis) |
| `tubelet_size` | Frames per clip |
| `n` | Seconds between consecutive frames within a clip |
| `control_frequency` | Steps per second (default `10.0`) |
| `image_keys` | Tuple-paths to permute HWC → CHW |

### Clip placement

- `batch_size == 1`: a single clip anchored at the **end** of the episode.
- `batch_size >= 2`: first clip at episode start, last clip at episode end, the
  rest evenly spaced between.
- Episodes shorter than a full clip are repeat-padded at the end.

### Example

```python
sampler = EpisodeTubeletSampler(
    batch_size=8,        # 8 clips spanning the episode
    tubelet_size=16,     # 16 frames per clip
    n=0.2,               # one frame every 0.2 s (stride 2 at 10 Hz)
    control_frequency=10,
    image_keys=dataset.image_keys,
)
dataset.set_sampler(sampler)

batch = dataset.sample()                # batch_size arg of the dataset is ignored
batch["observation"]["image"].shape     # (8, 16, C, H, W)
```

Note: the clip count is fixed by the sampler's own `batch_size`; the replay buffer's
`batch_size` argument is ignored when this sampler is active.

---

## JAXTemporalSampler

```python
from robotdataset import JAXTemporalSampler
```

The NumPy/JAX counterpart of `TemporalSampler`, used by
[`OXEJAXDataset`](datasets/oxe_jax.md). Identical constructor
(`delta_timestamps`, `control_frequency`, `image_keys`) and identical semantics:
uniform anchors, boundary clamping, mirrored `next` window, HWC→CHW permutation.

Differences:

- It samples from a `CombinedNumpyStore` and returns a **nested dict of
  `jax.numpy` arrays** (text leaves as Python lists).
- `sample(...)` accepts an explicit `rng: np.random.Generator` for reproducibility.

```python
import numpy as np
from robotdataset import JAXTemporalSampler

sampler = JAXTemporalSampler(
    delta_timestamps={"observation/image": [-0.2, -0.1, 0.0]},
    control_frequency=10,
    image_keys=jax_dataset.image_keys,
)
jax_dataset.set_sampler(sampler)

batch = jax_dataset.sample(rng=np.random.default_rng(0))
batch["observation"]["image"].shape   # (B, 3, C, H, W) — jax array
```
