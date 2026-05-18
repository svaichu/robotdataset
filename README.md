# robotdataset

A Python package for loading robot learning datasets into [TorchRL](https://github.com/pytorch/rl) TED format.

> **Status:** This repository is a collection for multiple robot datasets. Currently **only the OXE loader is functional**. All other dataset loaders (`Table30v2Dataset`, `AgiBotWorldBetaDataset`) are under active development.
>
> **OXE is in alpha.** APIs may change without notice.

---

## Supported datasets

| Dataset | Class | Status |
|---|---|---|
| Open X-Embodiment (OXE) | `OXEDataset` | Alpha |
| Table30 v2 | `Table30v2Dataset` | In development |
| AgiBotWorld-Beta | `AgiBotWorldBetaDataset` | In development |

---

## Installation

```bash
pip install "robotdataset[oxe]"
```

Or from source:

```bash
git clone https://github.com/robotics-action-group/robotdataset.git
cd robotdataset
pip install -e ".[oxe]"
```

**Requirements:** Python >= 3.7, PyTorch >= 1.13.1, TensorFlow >= 2.11.1, TensorFlow Datasets >= 4.8.2

---

## OXE Dataset

The OXE loader pulls from the [`gs://gresearch/robotics`](https://console.cloud.google.com/storage/browser/gresearch/robotics) GCS bucket and converts episodes into TorchRL TED format backed by memory-mapped tensors. Downloaded data is cached locally so subsequent runs skip the download.

### Discover available datasets

```python
from robotdataset import list_datasets

list_datasets()
# {'viola': ['0.1.0'], 'bridge_data_v2': ['0.0.1'], 'droid': ['1.0.1', '1.0.0'], ...}
```

Each key is a dataset name; values are the available version tags. `OXEDataset` always picks the highest version automatically.

### Load a dataset

```python
from robotdataset import OXEDataset

# Load the full train split (all episodes)
dataset = OXEDataset(
    dataset_name="viola",
    split="train",
    batch_size=16,
)

# Load only specific episodes (no full download — streams from GCS)
dataset = OXEDataset(
    dataset_name="cmu_playing_with_food",
    episodes=[0, 2],
    batch_size=6,
    control_frequency=5,   # Hz — used to convert delta_timestamps to step offsets
)
```

Downloaded shards and converted memmaps land in `~/.cache/robotdataset` by default. Override with the `ROBOTDATASET_CACHE` environment variable.

### Inspect modalities

```python
dataset.modalities
# {
#   'action': ['action'],
#   'image':  ['observation/finger_vision_1', 'observation/finger_vision_2', 'observation/image'],
#   'state':  ['observation/state'],
#   'text':   ['language_embedding', 'language_instruction'],
# }

dataset.image_keys
# [('observation', 'finger_vision_1'), ('observation', 'finger_vision_2'), ('observation', 'image')]

dataset.num_episodes   # 2  — episodes loaded
len(dataset)           # 244 — total steps across all loaded episodes
```

### Sampling without a temporal sampler

With no sampler set, each sample is a single time-step. Image tensors have shape `(B, H, W, C)`.

```python
batch = next(iter(dataset))
batch["observation"]["image"].shape   # (6, 480, 640, 3)
batch["action"].shape                 # (6, 8)
batch["language_instruction"]         # NonTensorStack(['Grasp the carrot slice.', ...])
```

The batch follows TorchRL TED layout:

```
observation/          ← step t
next/observation/     ← step t+1
next/reward
done, terminated
action
collector/episode_id  ← which episode each row belongs to
```

### Temporal sampling

`TemporalSampler` gathers a window of frames around each sampled step. `delta_timestamps` is a dict mapping a modality path to a list of time offsets **in seconds** relative to the current step.

```python
import numpy as np
from robotdataset import TemporalSampler

sampler = TemporalSampler(
    delta_timestamps={
        "observation/image":         np.arange(-1.0, 0.1, 0.1).tolist(),  # 10 frames
        "observation/finger_vision_1": np.arange(-0.5, 0.1, 0.1).tolist(),  # 5 frames
    },
    control_frequency=10,  # Hz
)
dataset.set_sampler(sampler)
```

With a temporal sampler active, image tensors pick up a time dimension `T`:

```python
batch = next(iter(dataset))
batch["observation"]["image"].shape          # (6, 10, 480, 640, 3) — (B, T, H, W, C)
batch["next"]["observation"]["image"].shape  # (6, 10, 480, 640, 3) — mirrored future window
batch["observation"]["state"].shape          # (6, 10, 6)           — (B, T, state_dim)
```

The `next` observation mirrors the window across the current step: if observation deltas are `[-0.2, -0.1, 0.0]`, the next-field deltas are `[0.0, 0.1, 0.2]`.

Pass `image_keys` to the sampler to automatically permute images from on-disk HWC to channel-first CHW `(B, T, C, H, W)`:

```python
sampler = TemporalSampler(
    delta_timestamps={"observation/image": [-0.2, -0.1, 0.0]},
    control_frequency=10,
    image_keys=dataset.image_keys,   # triggers HWC → CHW permutation
)
dataset.set_sampler(sampler)

batch = next(iter(dataset))
batch["observation"]["image"].shape   # (B, T, C, H, W)
```

### Visualisation

Both helpers work with a raw tensor **or** a TensorDict. `batchViz` renders all items in the batch side-by-side as a mosaic; `itemViz` renders a single item.

```python
from robotdataset import batchViz, itemViz

# Mosaic GIF of the whole batch — B=6 items arranged in a grid
batchViz(batch["observation"]["image"], fps=8)
```

![batchViz output — 6 items from cmu_playing_with_food, observation/image, 10-frame window at 8 fps](doc/batchviz_example.gif)

```python
# Mosaic MP4
batchViz(batch["observation"]["image"], fps=10, is_output_video=True)

# Single item as GIF
itemViz(batch["observation"]["image"], idx=0)

# Single item as MP4
itemViz(batch["observation"]["wrist_image"], idx=2, is_output_video=True)
```

Both functions return an IPython `Image` or `Video` for inline notebook display, or a file path string when `embed=False`. Supported tensor layouts are auto-detected: `(B, T, H, W, C)`, `(B, T, C, H, W)`, and `(B, C, T, H, W)`.

---

## License

MIT — see [LICENSE](LICENSE).
