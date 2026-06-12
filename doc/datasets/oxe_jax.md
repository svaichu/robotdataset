# OXE Datasets, JAX Path — `OXEJAXDataset`

> **Status: in development — not ready for use.**

`OXEJAXDataset` mirrors [`OXEDataset`](oxe.md) but produces **JAX arrays** in plain
nested dicts instead of TorchRL `TensorDict`s. Use it when your training loop is
written in JAX/Flax.

**Requires:** `tensorflow`, `tensorflow-datasets`, and `jax`. The import is
optional — `from robotdataset import OXEJAXDataset` yields `None` when JAX is
missing.

## Differences from the Torch path

| | `OXEDataset` | `OXEJAXDataset` |
|---|---|---|
| Batch type | `TensorDict` (torch tensors) | nested `dict` of `jax.numpy` arrays |
| Storage | TorchRL `TensorStorage` over a TED memmap | `CombinedNumpyStore` (NumPy memmaps + pickled text) |
| Sampler | `TemporalSampler` | `JAXTemporalSampler` |
| Combined cache dir | `combined/<split>/<hash>` | `combined_jax/<split>/<hash>` |
| Base class | `BaseDatasetExperienceReplay` | standalone class |
| RNG | torch global RNG | explicit `numpy.random.Generator` (optional) |

The download/conversion pipeline, per-episode caching, modality inference, cache
directory resolution, and the TED step layout are identical — both paths share the
same per-episode cache, so converting episodes once serves both.

## Usage

```python
import numpy as np
from robotdataset import OXEJAXDataset, JAXTemporalSampler

dataset = OXEJAXDataset(
    dataset_name="cmu_playing_with_food",
    episodes=[0, 2],
    batch_size=8,
    delta_timestamps={"observation/image": [-0.2, -0.1, 0.0]},
    control_frequency=10,
)

# Reproducible sampling with an explicit RNG
rng = np.random.default_rng(seed=0)
batch = dataset.sample(rng=rng)           # or dataset.get_batch(...)

batch["observation"]["image"].shape       # (8, 3, C, H, W) — jax array, CHW
batch["next"]["observation"]["image"].shape  # mirrored future window
batch["language_instruction"]             # list[str]
```

`sample(batch_size=None, rng=None)` overrides the constructor batch size per call;
`get_batch` is an alias. Text leaves come back as Python lists; everything numeric
is converted to `jnp` arrays.

### Constructor arguments

Same as `OXEDataset` (`dataset_name`, `split`, `version`, `episodes`, `batch_size`,
`root`, `delta_timestamps`, `control_frequency`), except there is no
`load_str_fields` flag — string fields are always carried.

## Inspecting

`num_episodes`, `len(dataset)`, `modalities`, `image_keys`, `get_modalities()`,
`data_path`, `data_path_root`, and `set_sampler(JAXTemporalSampler)` behave exactly
as in the Torch path.

The module also re-exports `list_datasets`, `validate_dataset_name`, and
`dataset2path` so the JAX path can be used without importing the Torch one.

Example notebook: [`example/example_oxe_jax.ipynb`](../../example/example_oxe_jax.ipynb)
