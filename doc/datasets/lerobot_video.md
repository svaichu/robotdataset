# LeRobot datasets with external video

`LeRobotVideoDataset` loads [LeRobot](https://github.com/huggingface/lerobot) v2.1
datasets whose camera streams live in standalone `.mp4` files rather than as image
columns inside the parquet — the layout `LeRobotDataset.push_to_hub` produces and
what almost every LeRobot dataset on the Hub looks like.

Like the other loaders in this package, it normalizes to
[TED format](../overview.md) backed by memory-mapped tensors, so episode
selection, temporal sampling and visualization all work the same way they do for
OXE or Table30v2.

## Why a separate loader

`Table30v2Dataset` goes through `datasets.load_dataset()`, which only sees the
parquet. For a LeRobot v2.1 dataset that parquet contains state and action but no
pixels: the frames are in `videos/chunk-000/<video_key>/episode_NNNNNN.mp4`,
addressed by a template in `meta/info.json`. This loader reads that template,
fetches the assets for the episodes you asked for, and decodes the frames.

## Quick start

```python
from robotdataset import LeRobotVideoDataset, batchViz

# From the Hub — fetches only the two episodes requested
ds = LeRobotVideoDataset(
    repo_id="LSY-lab/stack_without_ft_tact_v4",
    episodes=[0, 1],
    batch_size=8,
    delta_timestamps={"observation/images/primary": [-0.2, -0.1, 0.0]},
)

batch = ds.sample()
batch["observation/images/primary"].shape   # (8, 3, 3, 256, 256) — B, T, C, H, W
batch["observation/state/cartesian"].shape  # (8, 1, 6)
batch["action"].shape                       # (8, 1, 7)

batchViz(batch, key="observation/images/primary", fps=8)
```

From a directory already on disk — no network at all, which is what you want on a
cluster node or when someone else staged the data:

```python
ds = LeRobotVideoDataset(
    local_path="/hpcwork/$USER/gr00t-ft/data/stack_without_ft_tact_v4",
    episodes=[0, 1],
    batch_size=8,
)
```

## Constructor

| Argument | Default | Meaning |
|---|---|---|
| `repo_id` | `None` | HuggingFace Hub dataset id. Mutually exclusive with `local_path`. |
| `local_path` | `None` | Directory holding an already-materialised LeRobot tree. Mutually exclusive with `repo_id`. |
| `episodes` | `None` | Episode indices to load. All episodes if `None`. |
| `batch_size` | `32` | Transitions returned by `sample()`. |
| `root` | `None` | Cache root override. |
| `delta_timestamps` | `None` | Per-modality time deltas in **seconds**, keyed by `"/"`-separated path. |
| `control_frequency` | `None` | Steps per second for converting deltas to step offsets. Defaults to the dataset's own `meta/info.json` fps. |
| `video_keys` | `None` | Camera streams to decode. A subset skips downloading and decoding the rest. |
| `revision` | `None` | Hub branch, tag or commit sha. Pin a sha for cross-machine reproducibility. |

Shared members — `num_episodes`, `len(dataset)`, `image_keys`, `modalities` /
`get_modalities()`, `set_sampler()`, `data_path` / `data_path_root` — behave as
described in [Overview](../overview.md). Two extras: `fps` (the dataset's own
control frequency) and `source` (the underlying `LeRobotSource`).

## Loading only some episodes

Both origins are lazy per episode. With `repo_id`, only `meta/` plus the parquet
and mp4 files of the selected episodes are fetched, so two episodes of a
fifty-episode repo transfer two episodes' worth of bytes. With `local_path`, only
the selected episodes are read and decoded.

Per-episode TED memmaps are shared across selections; the combined storage is keyed
by an MD5 of the sorted episode list. Loading `[0, 1]` and later `[0, 1, 2]` reuses
the conversions of 0 and 1 and only converts 2.

```
$ROBOTDATASET_CACHE/hf/lerobot_video/<slug>/
├── hf_cache/                  # Hub downloads (repo_id origin only)
├── episodes/<episode_index>/  # per-episode TED memmaps
└── combined/<hash>/           # combined memmap for one episode selection
```

`<slug>` is the repo id with `/` replaced by `--`, or the directory name for a
local origin.

## Key layout

LeRobot's dot-separated column names become nested TED paths:

| LeRobot | TED path | Kind |
|---|---|---|
| `observation.images.primary` (mp4) | `observation/images/primary` | image |
| `observation.state.cartesian` | `observation/state/cartesian` | state |
| `observation.state` | `observation/state/all` | state |
| `action` | `action` | action |
| task string from `meta/` | `observation/language_instruction` | text |

The `observation/state/all` row is the one non-obvious mapping. Many LeRobot
datasets ship both `observation.state` (the concatenated vector) and
`observation.state.<slice>` columns. A nested TensorDict cannot hold a tensor and
a sub-dict under the same name, so the aggregate moves to `all` and the slices keep
their own names. Nothing is dropped.

Images are stored HWC uint8 on disk and permuted to CHW at sampling time, as
everywhere else in this package.

## Frame rate and codec

Two things about LeRobot v2.1 exports are worth knowing, because both have bitten
this dataset:

**The fps field is not always singular.** `meta/info.json` can carry a top-level
`fps`, an `features[<key>].info["video.fps"]`, and an
`features[<key>].video_info["video.fps"]` that disagree — on
`LSY-lab/stack_without_ft_tact_v4` the last one says 30 while the stream really is
15. The loader uses the top-level `fps`, and the builder asserts that every
camera's decoded frame count equals the parquet row count. A mismatch raises
`FrameCountMismatch` at build time rather than producing a policy trained on
frames shifted against their states.

**The codec is usually AV1.** LeRobot's default encoder emits AV1, which `decord`
cannot decode — it tends to return blank frames rather than raising. This loader
decodes through `imageio` / `imageio-ffmpeg`, whose bundled ffmpeg 7.x reads AV1
via libdav1d. That is already a core dependency of this package, so nothing extra
is needed.

## Building blocks

The submodule is usable without the replay buffer when you want raw assets:

```python
from robotdataset.lerobot.loader import LeRobotSource
from robotdataset.lerobot.video_decoder import decode_mp4

src = LeRobotSource(repo_id="LSY-lab/stack_without_ft_tact_v4")
src.fps                      # 15.0
src.video_keys               # ['observation.images.primary', 'observation.images.wrist']
src.get_episode_length(0)    # 278
frames = decode_mp4(src.video_path(0, "observation.images.primary"))
frames.shape                 # (278, 256, 256, 3) uint8
```

`decode_mp4(path, indices=[...])` decodes a chosen subset of frames — still
sequentially, because inter-frame codecs make random seeks unreliable, but it stops
reading once the requested set is complete.
