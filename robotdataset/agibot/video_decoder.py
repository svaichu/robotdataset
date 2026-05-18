"""MP4 frame extraction from TAR archives for AgiBotWorld-Beta."""

from __future__ import annotations

import os
import tarfile
import tempfile
from pathlib import Path

import numpy as np


def decode_mp4_from_tar(tar_path: Path, member_name: str) -> np.ndarray:
    """Extract one MP4 from a TAR archive and decode all frames.

    Writes the MP4 to a temporary file (required by ffmpeg), reads all frames
    with imageio, then deletes the temp file.

    Args:
        tar_path: Path to the local ``.tar`` file.
        member_name: Path inside the TAR, e.g.
            ``"648649/videos/head_color.mp4"``.

    Returns:
        ``(T, H, W, 3)`` uint8 numpy array — one row per frame.

    Raises:
        KeyError: If ``member_name`` is not found in the TAR.
        RuntimeError: If imageio cannot decode the extracted file.
    """
    import imageio

    with tarfile.open(tar_path, "r") as tf:
        member = tf.getmember(member_name)
        mp4_bytes = tf.extractfile(member).read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name

    try:
        reader = imageio.get_reader(tmp_path)
        frames = [np.array(frame) for frame in reader]
        reader.close()
    finally:
        os.unlink(tmp_path)

    if not frames:
        raise RuntimeError(f"No frames decoded from {member_name!r} in {tar_path}")
    return np.stack(frames)  # (T, H, W, 3) uint8


def get_video_fps(tar_path: Path, member_name: str) -> float:
    """Return the FPS of a video inside a TAR without decoding all frames."""
    import imageio

    with tarfile.open(tar_path, "r") as tf:
        member = tf.getmember(member_name)
        mp4_bytes = tf.extractfile(member).read()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(mp4_bytes)
        tmp_path = tmp.name

    try:
        reader = imageio.get_reader(tmp_path)
        fps = reader.get_meta_data().get("fps", 30.0)
        reader.close()
    finally:
        os.unlink(tmp_path)

    return float(fps)
