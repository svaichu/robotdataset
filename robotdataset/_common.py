from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _get_cache_dir(override: Optional[str] = None) -> Path:
    """Return the root cache directory.

    Priority: override argument → ROBOTDATASET_CACHE env var → ~/.cache/robotdataset
    """
    if override is not None:
        return Path(override)
    env = os.environ.get("ROBOTDATASET_CACHE")
    return Path(env) if env else Path.home() / ".cache" / "robotdataset"
