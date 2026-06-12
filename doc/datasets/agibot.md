# AgiBotWorld-Beta — `AgiBotWorldBetaDataset`

> **Status: in development.** The API may change.

Loads the [`agibot-world/AgiBotWorld-Beta`](https://huggingface.co/datasets/agibot-world/AgiBotWorld-Beta)
dataset from HuggingFace Hub. The source data is structured as TAR archives
(`observations/{task_id}/*.tar`) containing MP4 video streams from 8 cameras, with
episode metadata and language labels in `task_info/task_{id}.json`.

Only the **tasks** you request are downloaded, and only the **episodes** you request
are decoded from video into TED memmaps.

## Discovering tasks

```python
from robotdataset import list_agibot_tasks

list_agibot_tasks()[:5]   # [327, 351, 352, ...]
```

This is a metadata-only call (it reads the repo file index) — nothing heavy is
downloaded.

## Usage

```python
from robotdataset import AgiBotWorldBetaDataset

ds = AgiBotWorldBetaDataset(
    tasks=[327],
    episodes=[0, 1],
    batch_size=16,
)

batch = ds.sample()
print(ds.num_episodes)            # 2
print(ds.get_modalities().keys())
```

### Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `tasks` | required | Integer task IDs, e.g. `[327, 351]` |
| `split` | `"train"` | Kept for API consistency; AgiBot has a single data pool |
| `episodes` | all (within tasks) | **Global** episode indices to include |
| `cameras` | head + both hands | Camera streams to decode (see below) |
| `batch_size` | `32` | Transitions per `sample()` |
| `root` | cache default | Override cache root |
| `delta_timestamps` | `{key: [0.0]}` | Temporal window per modality, in seconds |
| `control_frequency` | `30.0` | Source video frame rate (AgiBotWorld-Beta records at 30 fps) |

## Global episode IDs

Episodes are numbered globally across all tasks, in sorted `(task_id, episode_id)`
order. The mapping is persisted to `episode_map.json` in the cache and extended when
new tasks are added. Inspect it via:

```python
ds.episode_map
# {0: (327, 648649), 1: (327, 648709), ...}   global_id → (task_id, source_episode_id)
```

## Camera selection

Default cameras: `["head_color", "hand_left_color", "hand_right_color"]`.
The full set is available as a constant:

```python
from robotdataset.agibot.loader import ALL_CAMERAS, DEFAULT_CAMERAS

ALL_CAMERAS
# ['head_color', 'head_left_fisheye_color', 'head_right_fisheye_color',
#  'head_center_fisheye_color', 'back_left_fisheye_color',
#  'back_right_fisheye_color', 'hand_left_color', 'hand_right_color']

ds = AgiBotWorldBetaDataset(tasks=[327], cameras=["head_color"], batch_size=8)
```

## TED layout

Each decoded camera frame becomes an observation field; the action is currently a
zero placeholder (action labels are not yet wired in):

```
observation/head_color           Tensor([H, W, 3], uint8)
observation/hand_left_color      Tensor([H, W, 3], uint8)
observation/hand_right_color     Tensor([H, W, 3], uint8)
observation/language_instruction NonTensor(str)
action                           Tensor([1], float32)   # zero placeholder
done, terminated, next/…, collector/episode_id           # standard TED fields
```

Sampling, `image_keys`, `modalities`, and `set_sampler` work exactly as in the
other Torch-path datasets — see [Samplers](../samplers.md).

Cache location: `~/.cache/robotdataset/hf/agibotworld-beta/<split>/{episodes,combined}/`.

Example notebook: [`example/example_agibot.ipynb`](../../example/example_agibot.ipynb)
