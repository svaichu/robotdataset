from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

from robotdataset.oxe.utils import latest_version

# Parallelism for the full-bucket scan (I/O bound, so threads are fine).
_SCAN_WORKERS = 32


def _safe_gfile_listdir(tf_module: Optional[object], path: str) -> List[str]:
    if tf_module is None:
        return []
    try:
        return tf_module.io.gfile.listdir(path)
    except Exception as exc:
        warnings.warn(
            f"Could not list GCS path '{path}': {exc}. "
            "Dataset discovery will return no results.",
            stacklevel=3,
        )
        return []


def _safe_gfile_isdir(tf_module: Optional[object], path: str) -> bool:
    if tf_module is None:
        return False
    try:
        return tf_module.io.gfile.isdir(path)
    except Exception:
        return False


def _join(base: str, name: str) -> str:
    """Join a GCS base URL and an entry name, stripping any stray slashes."""
    return f"{base.rstrip('/')}/{name.strip('/')}"


def _discover_one_dataset(
    tf_module: object, bucket_url: str, raw_name: str
) -> Optional[tuple[str, Dict[str, str]]]:
    """Return (dataset_name, {version: path}) for a single bucket entry, or None."""
    dataset_name = raw_name.strip("/")
    dataset_root = _join(bucket_url, dataset_name)
    if not _safe_gfile_isdir(tf_module, dataset_root):
        return None

    entries = _safe_gfile_listdir(tf_module, dataset_root)

    def _is_version_dir(entry: str) -> bool:
        return _safe_gfile_isdir(tf_module, _join(dataset_root, entry))

    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
        flags = list(pool.map(_is_version_dir, entries))

    versions = [e.strip("/") for e, is_dir in zip(entries, flags) if is_dir]

    if versions:
        return dataset_name, {v: _join(dataset_root, v) for v in versions}
    return dataset_name, {"": dataset_root}


def discover_dataset_versions(
    tf_module: Optional[object], bucket_url: str, dataset_name: str
) -> Dict[str, str]:
    """Return {version: gcs_path} for a single dataset by direct path lookup."""
    if tf_module is None:
        return {}
    dataset_root = _join(bucket_url, dataset_name)
    entries = _safe_gfile_listdir(tf_module, dataset_root)

    def _is_dir(entry: str) -> bool:
        return _safe_gfile_isdir(tf_module, _join(dataset_root, entry))

    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
        flags = list(pool.map(_is_dir, entries))

    versions = [e.strip("/") for e, is_dir in zip(entries, flags) if is_dir]
    if versions:
        return {v: _join(dataset_root, v) for v in versions}
    if _safe_gfile_isdir(tf_module, dataset_root):
        return {"": dataset_root}
    return {}


def discover_datasets_from_bucket(
    tf_module: Optional[object], bucket_url: str
) -> Dict[str, Dict[str, str]]:
    """Return {dataset_name: {version: gcs_path}} for every dataset in the bucket.

    The per-dataset isdir/listdir calls are issued in parallel so the full scan
    completes in roughly one round-trip worth of latency rather than O(N).
    """
    if tf_module is None:
        return {}

    raw_names = _safe_gfile_listdir(tf_module, bucket_url)

    discovered: Dict[str, Dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=_SCAN_WORKERS) as pool:
        futures = {
            pool.submit(_discover_one_dataset, tf_module, bucket_url, name): name
            for name in raw_names
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                name, versions = result
                discovered[name] = versions

    return discovered
