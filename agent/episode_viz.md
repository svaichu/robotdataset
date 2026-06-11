# episode_viz — decision log

## Task

Add `episodeViz`, a util that renders a **full episode** from a loaded
`OXEDataset` as an animated GIF or MP4.

---

## Key decisions

### 1. Where to add the function

Added to `robotdataset/utils/visualization.py` alongside `batchViz` and
`itemViz`.  Exported from `robotdataset/utils/__init__.py` and the top-level
`robotdataset/__init__.py` so users can do:

```python
from robotdataset import episodeViz
```

### 2. How to access the full episode without the sampler

`OXEDataset` stores all steps in a flat TensorDict via `TensorStorage`.  The
sampler picks random anchor steps and builds `(B, T, ...)` windows.  For a
full episode we need to bypass the sampler and read all `length` steps
contiguously.

Access path (mirrors what `TemporalSampler.__call__` already uses):

```python
storage_td = dataset._storage._storage      # raw TensorDict, shape (total_steps,)
episode_id  = dataset._loaded_indices[episode_idx]
start       = dataset._episode_starts[episode_id]
length      = dataset._episode_lengths[episode_id]
episode_td  = storage_td[start : start + length]  # shape (length,)
```

These private attributes are already used internally (`_sample`, temporal
sampler), so this is a stable internal interface, not a hack.

### 3. Which image keys to render

Default: `None` → `dataset.image_keys`, the full list inferred from TFDS
feature metadata.  Users can override with a single string or a list.  This
mirrors how `batchViz` defaults to `"observation/image"` but `episodeViz`
needs to handle multi-camera datasets naturally.

### 4. Multi-camera layout

When more than one key is requested each animation frame is a horizontal
strip: `np.hstack([frame_cam0, frame_cam1, ...])`.

Height normalisation: bottom-pad shorter images with zeros (black) to match
`max_h`.  This handles the common case where cameras differ only slightly in
height without requiring a resize dependency.

Alternative considered: resize to uniform height using `np.linspace` index
trick (nearest-neighbour).  Rejected because it changes pixel content and
adds noise to quality checks; padding is cheaper and honest.

### 5. Pixel normalisation

Same 1st/99th percentile clip as `batchViz`/`itemViz`, computed per key
over all `(T, H, W)` pixels.  Per-key normalisation lets each camera be
viewed at its own contrast level — important when wrist cams and overhead
cams have very different brightness distributions.

### 6. Return type

Same as `batchViz`/`itemViz`: IPython `Image`/`Video` when `embed=True`,
file path string when `embed=False`.  Keeps the API consistent.

### 7. Negative episode_idx

Supported via Python-style modulo (`episode_idx % n_episodes`), so
`episode_idx=-1` gives the last episode.

---

## Files changed

| File | Change |
|---|---|
| `robotdataset/utils/visualization.py` | Added `episodeViz`; added `List` to imports |
| `robotdataset/utils/__init__.py` | Export `episodeViz` |
| `robotdataset/__init__.py` | Export `episodeViz` in `__all__` |
| `doc/visualization.md` | Documented `episodeViz` with examples and argument table |
| `agent/episode_viz.md` | This file |
