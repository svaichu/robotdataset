from __future__ import annotations

from typing import List, Optional, Union

import imageio
import numpy as np
import torch
from IPython.display import Image, Video
from tensordict import TensorDict


def batchViz(
    batch: Union[TensorDict, torch.Tensor],
    key: str = "observation/image",
    fps: int = 4,
    is_output_video: bool = False,
    embed: bool = True,
    file_name: str = "batch",
) -> Optional[Union[Image, Video, str]]:
    """Render a batch of video frames as a mosaic GIF or MP4.

    Args:
        batch: Flat TensorDict from ``OXEDataset.sample()`` (keys are
            ``"/"``-separated strings) **or** a raw tensor.
        key: ``"/"``-separated key into the batch.  Ignored when ``batch``
            is already a tensor.
        fps: Frames per second for the output animation.
        is_output_video: ``True`` → write an MP4; ``False`` → write a GIF.
        embed: Return an IPython display object for inline notebook rendering.
        file_name: Output file stem (no extension).

    Supported tensor layouts (auto-detected):
        * ``(B, T, H, W, C)`` — HWC, default from OXEDataset storage
        * ``(B, T, C, H, W)`` — CHW, produced when ``image_keys`` is set
        * ``(B, C, T, H, W)`` — legacy layout
    """
    if isinstance(batch, torch.Tensor):
        video = batch
    else:
        video = batch[key]

    if video.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, T, *, *, *), got shape {tuple(video.shape)}")

    # --- normalise to (B, T, H, W, C) ---
    B, d1, d2, d3, d4 = video.shape
    if d4 in (1, 3):
        # Already (B, T, H, W, C) — HWC layout from OXEDataset storage
        video_bthwc = video
    elif d2 in (1, 3):
        # (B, T, C, H, W) — CHW layout (TemporalSampler with image_keys)
        video_bthwc = video.permute(0, 1, 3, 4, 2)
    elif d1 in (1, 3):
        # (B, C, T, H, W) — legacy layout
        video_bthwc = video.permute(0, 2, 3, 4, 1)
    else:
        raise ValueError(
            f"Cannot infer channel position from shape {tuple(video.shape)}. "
            "Expected C in (1, 3) at dim 1, 2, or 4."
        )

    grid = video_bthwc.detach().float().cpu().contiguous().numpy()
    if grid.shape[-1] == 1:
        grid = np.repeat(grid, 3, axis=-1)

    # Robust percentile normalisation across the whole batch+time
    lo = np.percentile(grid, 1.0, axis=(0, 1, 2, 3), keepdims=True)
    hi = np.percentile(grid, 99.0, axis=(0, 1, 2, 3), keepdims=True)
    denom = np.maximum(hi - lo, 1e-6)
    grid_u8 = (((grid - lo) / denom).clip(0.0, 1.0) * 255.0).astype(np.uint8)

    B, T, H, W, C = grid_u8.shape
    cols = int(np.ceil(np.sqrt(B)))
    rows = int(np.ceil(B / cols))
    pad = 4

    mosaic_frames = []
    for t in range(T):
        canvas_h = rows * H + (rows - 1) * pad
        canvas_w = cols * W + (cols - 1) * pad
        canvas = np.zeros((canvas_h, canvas_w, C), dtype=np.uint8)
        for b in range(B):
            r, c_idx = divmod(b, cols)
            y0 = r * (H + pad)
            x0 = c_idx * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = grid_u8[b, t]
        mosaic_frames.append(canvas)

    if is_output_video:
        out_path = f"{file_name}.mp4"
        try:
            imageio.mimsave(out_path, mosaic_frames, fps=fps)
        except Exception as e:
            print(f"MP4 export skipped: {e}")
        return Video(filename=out_path, embed=True) if embed else out_path
    else:
        out_path = f"{file_name}.gif"
        imageio.mimsave(out_path, mosaic_frames, fps=fps, loop=0)
        return Image(filename=out_path) if embed else out_path


def itemViz(
    batch: Union[TensorDict, torch.Tensor],
    idx: int = 0,
    key: str = "observation/image",
    fps: int = 4,
    is_output_video: bool = False,
    embed: bool = True,
    file_name: str = "item",
) -> Optional[Union[Image, Video, str]]:
    """Render the temporal frames of a single batch item as a GIF or MP4.

    Args:
        batch: Flat TensorDict from ``OXEDataset.sample()`` or a raw tensor.
        idx: Which item in the batch to visualise (default ``0``).
        key: ``"/"``-separated key into the batch.  Ignored when ``batch``
            is already a tensor.
        fps: Frames per second for the output animation.
        is_output_video: ``True`` → write an MP4; ``False`` → write a GIF.
        embed: Return an IPython display object for inline notebook rendering.
        file_name: Output file stem (no extension).
    """
    if isinstance(batch, torch.Tensor):
        video = batch
    else:
        video = batch[key]

    if video.ndim != 5:
        raise ValueError(f"Expected 5D tensor (B, T, *, *, *), got shape {tuple(video.shape)}")

    if idx >= video.shape[0]:
        raise IndexError(f"idx={idx} out of range for batch size {video.shape[0]}")

    # Select single item: (T, d1, d2, d3)
    item = video[idx]

    # Normalise to (T, H, W, C)
    T, d1, d2, d3 = item.shape
    if d3 in (1, 3):
        # (T, H, W, C) — HWC
        item_thwc = item
    elif d1 in (1, 3):
        # (T, C, H, W) — CHW
        item_thwc = item.permute(0, 2, 3, 1)
    else:
        raise ValueError(
            f"Cannot infer channel position from shape {tuple(item.shape)}. "
            "Expected C in (1, 3) at dim 1 or 3."
        )

    frames = item_thwc.detach().float().cpu().contiguous().numpy()
    if frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)

    lo = np.percentile(frames, 1.0, axis=(0, 1, 2), keepdims=True)
    hi = np.percentile(frames, 99.0, axis=(0, 1, 2), keepdims=True)
    denom = np.maximum(hi - lo, 1e-6)
    frames_u8 = (((frames - lo) / denom).clip(0.0, 1.0) * 255.0).astype(np.uint8)

    if is_output_video:
        out_path = f"{file_name}.mp4"
        try:
            imageio.mimsave(out_path, list(frames_u8), fps=fps)
        except Exception as e:
            print(f"MP4 export skipped: {e}")
        return Video(filename=out_path, embed=True) if embed else out_path
    else:
        out_path = f"{file_name}.gif"
        imageio.mimsave(out_path, list(frames_u8), fps=fps, loop=0)
        return Image(filename=out_path) if embed else out_path


def episodeViz(
    dataset: "OXEDataset",  # type: ignore[name-defined]
    episode_idx: int = 0,
    key: Optional[Union[str, List[str]]] = None,
    fps: int = 4,
    is_output_video: bool = False,
    embed: bool = True,
    file_name: str = "episode",
) -> Optional[Union[Image, Video, str]]:
    """Render every frame of one episode as an animated GIF or MP4.

    When multiple image keys are requested (or auto-detected from
    ``dataset.image_keys``), each animation frame is a horizontal strip of
    all cameras side-by-side in key order.

    Args:
        dataset: A loaded ``OXEDataset`` instance.
        episode_idx: 0-based index into ``dataset._loaded_indices``.
        key: Image key(s) to render.  ``None`` → all ``dataset.image_keys``.
            Pass a single ``"/"``-separated string or a list of strings to
            select specific cameras.
        fps: Frames per second for the output animation.
        is_output_video: ``True`` → write an MP4; ``False`` → write a GIF.
        embed: Return an IPython display object for inline notebook rendering.
        file_name: Output file stem (no extension).

    Returns:
        IPython ``Image`` or ``Video`` when ``embed=True``; output file path
        string when ``embed=False``.

    Raises:
        IndexError: ``episode_idx`` is out of range.
        ValueError: No image keys found or a key has an unexpected shape.
    """
    # ------------------------------------------------------------------ #
    # 1. Resolve which image keys to render
    # ------------------------------------------------------------------ #
    if key is None:
        keys: List[str] = list(dataset.image_keys)
        if not keys:
            raise ValueError(
                "dataset.image_keys is empty. Pass key= explicitly with the "
                "slash-separated path to the image modality."
            )
    elif isinstance(key, str):
        keys = [key]
    else:
        keys = list(key)

    # ------------------------------------------------------------------ #
    # 2. Locate the episode in flat storage
    # ------------------------------------------------------------------ #
    n_episodes = len(dataset._loaded_indices)
    if episode_idx >= n_episodes or episode_idx < -n_episodes:
        raise IndexError(
            f"episode_idx={episode_idx} out of range for {n_episodes} loaded episode(s)."
        )
    episode_id = dataset._loaded_indices[episode_idx % n_episodes]
    start = dataset._episode_starts[episode_id]
    length = dataset._episode_lengths[episode_id]

    # Raw flat TensorDict of shape (total_steps,)
    storage_td = dataset._storage._storage
    episode_td = storage_td[start : start + length]  # (T,)

    # ------------------------------------------------------------------ #
    # 3. Extract per-key frame arrays: (T, H, W, 3) uint8
    # ------------------------------------------------------------------ #
    per_key_frames: List[np.ndarray] = []
    for k in keys:
        parts = tuple(k.split("/"))
        node = episode_td
        try:
            for p in parts:
                node = node[p]
        except KeyError:
            raise KeyError(
                f"Key '{k}' not found in episode storage. "
                f"Available top-level keys: {sorted(episode_td.keys())}"
            )

        arr = node.detach().float().cpu().contiguous().numpy()  # (T, ...)
        if arr.ndim != 4:
            raise ValueError(
                f"Key '{k}' has shape {arr.shape}; expected 4-D (T, H, W, C) "
                "or (T, C, H, W)."
            )

        # Normalise to (T, H, W, C)
        T, d1, d2, d3 = arr.shape
        if d3 in (1, 3):
            pass  # already HWC
        elif d1 in (1, 3):
            arr = arr.transpose(0, 2, 3, 1)  # CHW → HWC
        else:
            raise ValueError(
                f"Cannot infer channel position from shape {arr.shape} for key '{k}'."
            )

        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        # Per-key percentile normalisation → uint8
        lo = np.percentile(arr, 1.0, axis=(0, 1, 2), keepdims=True)
        hi = np.percentile(arr, 99.0, axis=(0, 1, 2), keepdims=True)
        denom = np.maximum(hi - lo, 1e-6)
        arr = (((arr - lo) / denom).clip(0.0, 1.0) * 255.0).astype(np.uint8)

        per_key_frames.append(arr)

    # ------------------------------------------------------------------ #
    # 4. Assemble animation frames
    #    Single key  → (T, H, W, 3) directly.
    #    Multi-key   → horizontal strip; height-pad shorter cameras with zeros.
    # ------------------------------------------------------------------ #
    T = per_key_frames[0].shape[0]

    if len(per_key_frames) == 1:
        frames = [per_key_frames[0][t] for t in range(T)]
    else:
        max_h = max(arr.shape[1] for arr in per_key_frames)
        frames = []
        for t in range(T):
            strips = []
            for arr in per_key_frames:
                frame = arr[t]  # (H, W, 3)
                h, w = frame.shape[:2]
                if h < max_h:
                    pad = np.zeros((max_h - h, w, 3), dtype=np.uint8)
                    frame = np.vstack([frame, pad])
                strips.append(frame)
            frames.append(np.hstack(strips))

    # ------------------------------------------------------------------ #
    # 5. Write output
    # ------------------------------------------------------------------ #
    if is_output_video:
        out_path = f"{file_name}.mp4"
        try:
            imageio.mimsave(out_path, frames, fps=fps)
        except Exception as e:
            print(f"MP4 export skipped: {e}")
        return Video(filename=out_path, embed=True) if embed else out_path
    else:
        out_path = f"{file_name}.gif"
        imageio.mimsave(out_path, frames, fps=fps, loop=0)
        return Image(filename=out_path) if embed else out_path
