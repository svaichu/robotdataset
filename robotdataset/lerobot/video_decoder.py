"""MP4 frame extraction for LeRobot video datasets.

LeRobot v2.1 stores each camera stream as a standalone mp4 beside the parquet
data.  Frames are decoded with imageio, whose bundled ffmpeg reads the AV1
streams LeRobot's default encoder produces -- a codec that decord, the other
common backend in this space, cannot decode (it fails silently with blank
frames rather than raising).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np


def decode_mp4(path: Path, indices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Decode an mp4 to a stacked uint8 array.

    Args:
        path: Path to the mp4 file.
        indices: Frame indices to keep, in the order they should be returned.
            ``None`` (default) decodes every frame.  Decoding is sequential
            either way -- inter-frame codecs make random seeks unreliable --
            but a bounded ``indices`` set stops reading once satisfied.

    Returns:
        ``(T, H, W, 3)`` uint8 array, ``T == len(indices)`` when given.

    Raises:
        IndexError: If any requested index is beyond the end of the stream.
        RuntimeError: If the file decodes to zero frames.
    """
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        if indices is None:
            frames = [np.asarray(frame) for frame in reader]
        else:
            wanted = {int(i) for i in indices}
            collected = {}
            for position, frame in enumerate(reader):
                if position in wanted:
                    collected[position] = np.asarray(frame)
                    if len(collected) == len(wanted):
                        break
            missing = sorted(wanted - set(collected))
            if missing:
                raise IndexError(
                    f"Frames {missing} are past the end of {Path(path).name}"
                )
            frames = [collected[int(i)] for i in indices]
    finally:
        reader.close()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return np.stack(frames)


def get_video_fps(path: Path) -> float:
    """Return the frame rate the container reports, or 0.0 if absent."""
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        return float(reader.get_meta_data().get("fps", 0.0))
    finally:
        reader.close()


def count_frames(path: Path) -> int:
    """Count frames by decoding the stream -- the only reliable way for AV1."""
    import imageio

    reader = imageio.get_reader(str(path))
    try:
        return sum(1 for _ in reader)
    finally:
        reader.close()
