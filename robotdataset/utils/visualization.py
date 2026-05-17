from __future__ import annotations

from typing import Optional, Union

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

    Accepts either a raw tensor or a :class:`TensorDict` produced by
    :class:`~robotdataset.OXEDataset`.

    Args:
        batch: A ``TensorDict`` from ``OXEDataset.sample()`` **or** a raw
            tensor.  When a ``TensorDict`` is given, ``key`` selects which
            modality to visualise (slash-separated path, e.g.
            ``"observation/image"``).
        key: Slash-separated key into the batch TensorDict.  Ignored when
            ``batch`` is already a tensor.
        fps: Frames per second for the output animation.
        is_output_video: ``True`` → write an MP4; ``False`` → write a GIF.
        embed: Return an IPython display object (``Image`` / ``Video``) for
            inline notebook rendering.
        file_name: Output file stem (no extension).

    Returns:
        An IPython ``Image`` or ``Video`` when ``embed=True``, otherwise the
        output file path string.

    Supported tensor layouts (auto-detected):
        * ``(B, T, H, W, C)`` — HWC, default from OXEDataset storage
        * ``(B, T, C, H, W)`` — CHW, produced when ``image_keys`` is set
        * ``(B, C, T, H, W)`` — legacy layout
    """
    # --- extract tensor from TensorDict ---
    if isinstance(batch, TensorDict):
        key_tuple = tuple(key.split("/"))
        try:
            video = batch[key_tuple]
        except KeyError:
            raise KeyError(
                f"Key {key!r} not found in batch. "
                f"Available top-level keys: {sorted(batch.keys())}"
            )
    else:
        video = batch

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
        batch: A ``TensorDict`` from ``OXEDataset.sample()`` or a raw tensor.
        idx: Which item in the batch to visualise (default ``0``).
        key: Slash-separated key into the batch TensorDict.  Ignored when
            ``batch`` is already a tensor.
        fps: Frames per second for the output animation.
        is_output_video: ``True`` → write an MP4; ``False`` → write a GIF.
        embed: Return an IPython display object for inline notebook rendering.
        file_name: Output file stem (no extension).

    Returns:
        An IPython ``Image`` or ``Video`` when ``embed=True``, otherwise the
        output file path string.
    """
    if isinstance(batch, TensorDict):
        key_tuple = tuple(key.split("/"))
        try:
            video = batch[key_tuple]
        except KeyError:
            raise KeyError(
                f"Key {key!r} not found in batch. "
                f"Available top-level keys: {sorted(batch.keys())}"
            )
    else:
        video = batch

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
