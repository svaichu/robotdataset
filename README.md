# robotdataset

[![Publish to PyPI](https://github.com/svaichu/robotdataset/actions/workflows/publish.yml/badge.svg)](https://github.com/svaichu/robotdataset/actions/workflows/publish.yml)

`robotdataset` is a Python library for loading robot learning datasets from multiple
sources (Google Cloud Storage, HuggingFace Hub) into the [TorchRL](https://github.com/pytorch/rl)
**TED format** (TorchRL Episode Data), backed by memory-mapped tensors for efficient
use in deep-learning training pipelines.

## Documentation

| Page | Covers |
|---|---|
| [Overview & TED format](doc/overview.md) | Library scope, architecture, the TED format, caching |
| [OXE datasets](doc/datasets/oxe.md) | `OXEDataset`, `list_datasets`, `validate_dataset_name`, `dataset2path` |
| [OXE JAX datasets](doc/datasets/oxe_jax.md) | `OXEJAXDataset` — NumPy/JAX batch path |
| [Table30v2](doc/datasets/table30v2.md) | `Table30v2Dataset` (RoboChallenge/Table30v2, HuggingFace) |
| [AgiBotWorld-Beta](doc/datasets/agibot.md) | `AgiBotWorldBetaDataset`, `list_agibot_tasks` |
| [Samplers](doc/samplers.md) | `TemporalSampler`, `EpisodeTubeletSampler`, `JAXTemporalSampler` |
| [Visualization](doc/visualization.md) | `batchViz`, `itemViz` |

## Quick start

```bash
pip install "robotdataset[oxe]"
```

```python
from robotdataset import OXEDataset, TemporalSampler, batchViz

# Load two episodes of an OXE dataset (streams from GCS, caches locally)
dataset = OXEDataset(
    dataset_name="cmu_playing_with_food",
    episodes=[0, 2],
    batch_size=6,
)

# Attach a temporal sampler: each sample carries a 10-frame image history
sampler = TemporalSampler(
    delta_timestamps={"observation/image": [-0.9, -0.8, -0.7, -0.6, -0.5,
                                            -0.4, -0.3, -0.2, -0.1, 0.0]},
    control_frequency=10,
)
dataset.set_sampler(sampler)

batch = dataset.sample()
batch["observation/image"].shape   # (6, 10, 480, 640, 3) — (B, T, H, W, C)

# Visualize the batch as an animated mosaic
batchViz(batch, key="observation/image", fps=8)
```

![batchViz output — 6 items from cmu_playing_with_food, observation/image, 10-frame window at 8 fps](doc/batchviz_example.gif)

## Dataset status

> **Only `OXEDataset` is usable today.** It is in alpha — APIs may change without
> notice. All other loaders are under active development and not yet ready for use.

| Dataset | Class | Source | Status |
|---|---|---|---|
| Open X-Embodiment (OXE) | `OXEDataset` | `gs://gresearch/robotics` | **Alpha (usable, not stable)** |
| Open X-Embodiment (JAX path) | `OXEJAXDataset` | `gs://gresearch/robotics` | In development |
| Table30 v2 | `Table30v2Dataset` | `RoboChallenge/Table30v2` (HF) | In development |
| AgiBotWorld-Beta | `AgiBotWorldBetaDataset` | `agibot-world/AgiBotWorld-Beta` (HF) | In development |

## Public API

Everything below is importable directly from the top-level package:

```python
from robotdataset import (
    # Dataset classes
    OXEDataset, OXEJAXDataset, Table30v2Dataset, AgiBotWorldBetaDataset,
    # Samplers
    TemporalSampler, EpisodeTubeletSampler, JAXTemporalSampler,
    # Discovery helpers
    list_datasets, validate_dataset_name, dataset2path, list_agibot_tasks,
    # Visualization
    batchViz, itemViz,
)
```

`OXEJAXDataset` and `JAXTemporalSampler` are `None` when JAX is not installed —
they are optional-dependency imports.

## License

MIT — see [LICENSE](LICENSE).
