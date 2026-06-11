# Visualization — `batchViz` and `itemViz`

Two helpers for rendering sampled video tensors as animated GIFs or MP4s, designed
for quick inspection inside notebooks.

```python
from robotdataset import batchViz, itemViz
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

## Notes

- Input must be 5-D `(B, T, ...)` — sample with a temporal window (`T > 1`) for a
  meaningful animation; with the default anchor-only sampler the animation has a
  single frame.
- Grayscale (`C=1`) tensors are repeated to RGB automatically.
- MP4 export requires an imageio ffmpeg backend; if unavailable the error is
  printed and the call returns without writing.
