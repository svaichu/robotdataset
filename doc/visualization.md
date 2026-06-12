# Visualization — `batchViz`, `itemViz`, and `episodeViz`

Three helpers for rendering video tensors as animated GIFs or MP4s, designed
for quick inspection inside notebooks.

```python
from robotdataset import batchViz, itemViz, episodeViz
```

Both helpers:

- accept either a **raw tensor** or a **TensorDict** batch (with `key` selecting the
  modality, default `"observation/image"`);
- auto-detect the tensor layout — `(B, T, H, W, C)`, `(B, T, C, H, W)`, and
  `(B, C, T, H, W)` are all supported, so output works regardless of whether
  `image_keys` permutation was active;
- normalize pixel values robustly using the 1st/99th percentiles, so float and
  uint8 inputs both render correctly;
- return an IPython `Image`/`Video` for inline display when `embed=True` (default),
  or the output file path string when `embed=False`.

## `batchViz` — whole-batch mosaic

Arranges all `B` items of the batch in a square-ish grid and animates them over the
`T` time dimension.

```python
batch = dataset.sample()

# Animated GIF mosaic of the whole batch
batchViz(batch, key="observation/image", fps=8)

# Or pass the tensor directly (flat key since the batch is a flat TensorDict)
batchViz(batch["observation/image"], fps=8)

# MP4 instead of GIF, saved to my_batch.mp4
batchViz(batch, fps=10, is_output_video=True, file_name="my_batch")

# Just get the file path (e.g. for scripts / CI)
path = batchViz(batch, embed=False)   # "batch.gif"
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `batch` | required | `TensorDict` from `dataset.sample()` or a 5-D tensor |
| `key` | `"observation/image"` | Slash-separated path into the TensorDict (ignored for raw tensors) |
| `fps` | `4` | Animation frame rate |
| `is_output_video` | `False` | `True` → MP4, `False` → GIF |
| `embed` | `True` | Return IPython display object vs. file path |
| `file_name` | `"batch"` | Output file stem (no extension) |

## `itemViz` — single batch item

Renders the temporal frames of one item (`idx`) in the batch.

```python
# First item as a GIF
itemViz(batch, idx=0)

# Third item's wrist camera as an MP4
itemViz(batch, idx=2, key="observation/wrist_image", is_output_video=True)
```

`itemViz` takes the same arguments as `batchViz` plus `idx` (default `0`,
raises `IndexError` when out of range); its default `file_name` is `"item"`.

## `episodeViz` — full episode playback

Renders **all steps** of one episode directly from the dataset, bypassing the
temporal sampler entirely.  Useful for sanity-checking data quality, inspecting
episode length, and comparing multiple camera angles at once.

```python
dataset = OXEDataset("viola", episodes=[0, 1], batch_size=8)

# All cameras as a side-by-side strip, animated over every step
episodeViz(dataset, episode_idx=0)

# Single camera
episodeViz(dataset, episode_idx=1, key="observation/image")

# Multiple specific cameras
episodeViz(dataset, episode_idx=0, key=["observation/image", "observation/wrist_image"])

# MP4 at 10 fps, saved to ep0.mp4
episodeViz(dataset, episode_idx=0, fps=10, is_output_video=True, file_name="ep0")

# Just get the file path (e.g. for scripts)
path = episodeViz(dataset, episode_idx=0, embed=False)   # "episode.gif"
```

### Arguments

| Argument | Default | Description |
|---|---|---|
| `dataset` | required | Loaded `OXEDataset` instance |
| `episode_idx` | `0` | 0-based index into the loaded episode list |
| `key` | `None` | Key(s) to render. `None` → all `dataset.image_keys`. String or list of strings |
| `fps` | `4` | Animation frame rate |
| `is_output_video` | `False` | `True` → MP4, `False` → GIF |
| `embed` | `True` | Return IPython display object vs. file path |
| `file_name` | `"episode"` | Output file stem (no extension) |

**Multi-camera layout:** when more than one key is rendered, each animation
frame is a horizontal strip of all cameras in key order.  If cameras have
different heights, shorter images are bottom-padded with black.

## Notes

- `batchViz` / `itemViz` input must be 5-D `(B, T, ...)` — sample with a
  temporal window (`T > 1`) for a meaningful animation; with the default
  anchor-only sampler the animation has a single frame.
- `episodeViz` reads from raw storage (not the sampler), so it always shows
  the full episode regardless of `delta_timestamps`.
- Grayscale (`C=1`) tensors are repeated to RGB automatically.
- MP4 export requires an imageio ffmpeg backend; if unavailable the error is
  printed and the call returns without writing.
