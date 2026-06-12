from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import torch

try:
    from tqdm.auto import tqdm as _tqdm_cls
except ImportError:
    _tqdm_cls = None  # type: ignore[assignment]

from tensordict import TensorDict
from torchrl.data import ImmutableDatasetWriter, TensorStorage
from torchrl.data.datasets.common import BaseDatasetExperienceReplay

from robotdataset._common import _get_cache_dir
from robotdataset.oxe.episode_tubelet_sampler import EpisodeTubeletSampler
from robotdataset.oxe.temporal_sampler import TemporalSampler
from robotdataset.oxe.bucket import discover_dataset_versions, discover_datasets_from_bucket
from robotdataset.oxe.memmap_builder import (
    build_combined_storage,
    build_missing_episodes,
    combined_dir_key,
    is_combined_complete,
    is_episode_cached,
)
from robotdataset.oxe.utils import (
    ModalitySpec,
    flatten_structure,
    infer_kind,
    latest_version,
    normalize_version_key,
    tf_to_torch,
    dict_to_tensordict,
)

try:
    import tensorflow as tf
except ImportError:  # pragma: no cover - optional dependency
    tf = None

try:
    import tensorflow_datasets as tfds
except ImportError:  # pragma: no cover - optional dependency
    tfds = None


OXE_BUCKET_URL = "gs://gresearch/robotics"

_DATASET_CACHE: Optional[Dict[str, Dict[str, str]]] = None
_TF_TENSOR_TYPES = (tf.Tensor,) if tf is not None else tuple()


# ---------------------------------------------------------------------------
# Progress helpers (LeRobotDataset-style tqdm)
# ---------------------------------------------------------------------------

def _progress(
    iterable: Iterable,
    *,
    desc: str = "",
    unit: str = "it",
    total: Optional[int] = None,
    leave: bool = True,
) -> Iterable:
    if _tqdm_cls is not None:
        return _tqdm_cls(iterable, desc=desc, unit=unit, total=total, leave=leave, dynamic_ncols=True)
    return iterable


# ---------------------------------------------------------------------------
# TF stack guard
# ---------------------------------------------------------------------------

def _require_tf_stack() -> None:
    if tf is None or tfds is None:
        raise ImportError(
            "OXEDataset requires tensorflow and tensorflow-datasets. "
            "Install them with: pip install 'robotdataset[oxe]'"
        )


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------

def _get_dataset_map(refresh: bool = False, dataset_name: Optional[str] = None) -> Dict[str, Dict[str, str]]:
    global _DATASET_CACHE

    # Fast path: if a specific dataset is requested and already cached, return immediately.
    if not refresh and _DATASET_CACHE is not None and (
        dataset_name is None or dataset_name in _DATASET_CACHE
    ):
        return _DATASET_CACHE

    if _DATASET_CACHE is None:
        _DATASET_CACHE = {}

    # When a dataset name is known, try its direct GCS path first — avoids scanning
    # the entire bucket (hundreds of API calls) just to validate one dataset.
    if dataset_name is not None and (refresh or dataset_name not in _DATASET_CACHE):
        direct = discover_dataset_versions(tf, OXE_BUCKET_URL, dataset_name)
        if direct:
            _DATASET_CACHE[dataset_name] = direct
            return _DATASET_CACHE

    # Full bucket scan — used by list_datasets() or when direct lookup found nothing.
    if refresh or not _DATASET_CACHE:
        _DATASET_CACHE = discover_datasets_from_bucket(tf, OXE_BUCKET_URL)

    return _DATASET_CACHE


def list_datasets(refresh: bool = False) -> Dict[str, List[str]]:
    """Return {dataset_name: [versions]} sorted newest-first from the GCS bucket.

    Example: {'viola': ['0.1.0'], 'bridge': ['1.0.0'], ...}
    """
    return {
        name: sorted(versions.keys(), key=normalize_version_key, reverse=True)
        for name, versions in _get_dataset_map(refresh).items()
    }


def validate_dataset_name(dataset_name: str, version: Optional[str] = None) -> bool:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        return False
    if version is not None:
        return version in datasets[dataset_name]
    return True


def dataset2path(dataset_name: str, version: Optional[str] = None) -> str:
    datasets = _get_dataset_map(dataset_name=dataset_name)
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. Use list_datasets() to see available datasets."
        )
    versions = datasets[dataset_name]
    if version is not None:
        if version not in versions:
            available = ", ".join(sorted(versions.keys(), key=normalize_version_key, reverse=True))
            raise ValueError(
                f"Version '{version}' not available for '{dataset_name}'. Available: {available}"
            )
        return versions[version]
    return versions[latest_version(list(versions.keys()))]


# ---------------------------------------------------------------------------
# Tensor conversion helpers
# ---------------------------------------------------------------------------

def _tf_to_torch(value: Any) -> Any:
    return tf_to_torch(value, _TF_TENSOR_TYPES)


def _flatten_structure(tree: Any, prefix: str = "") -> Dict[str, ModalitySpec]:
    return flatten_structure(tree, _TF_TENSOR_TYPES, prefix)


def _flatten_tfds_features(tree: Any, prefix: str = "") -> Dict[str, ModalitySpec]:
    """Traverse a TFDS FeaturesDict spec tree and return a ModalitySpec per leaf.

    Unlike flatten_structure (which handles actual data values), this function
    understands TFDS feature schema objects:
      - FeaturesDict / any dict-like → recurse into items()
      - Sequence / Dataset wrapper   → unwrap via .feature (RLDS 'steps' key
                                       is stripped from the path automatically)
      - Image / Tensor / Text leaf   → create ModalitySpec with inferred kind
    """
    flattened: Dict[str, ModalitySpec] = {}

    # FeaturesDict and plain dicts both have items()
    if hasattr(tree, "items") and callable(tree.items):
        try:
            for key, value in tree.items():
                if key == "steps" and hasattr(value, "feature"):
                    # RLDS top-level Sequence wrapper — descend without 'steps/' prefix
                    flattened.update(_flatten_tfds_features(value.feature, prefix))
                else:
                    child = f"{prefix}/{key}" if prefix else str(key)
                    flattened.update(_flatten_tfds_features(value, child))
            return flattened
        except Exception:
            pass

    # Sequence / Dataset feature (not dict-like but wraps an inner feature spec)
    if hasattr(tree, "feature") and not hasattr(tree, "items"):
        return _flatten_tfds_features(tree.feature, prefix)

    # Leaf feature spec — extract shape and dtype
    shape: Optional[tuple] = None
    dtype_str: Optional[str] = None
    if hasattr(tree, "shape"):
        try:
            raw = tree.shape
            if hasattr(raw, "as_list"):
                raw = raw.as_list()
            shape = tuple(-1 if d is None else int(d) for d in raw) if raw else ()
        except Exception:
            pass
    if hasattr(tree, "dtype"):
        try:
            dt = tree.dtype
            dtype_str = dt.name if hasattr(dt, "name") else str(dt)
        except Exception:
            pass

    path = prefix or "value"
    type_name = type(tree).__name__.lower()

    if dtype_str in {"string", "tf.string"} or "text" in type_name or "string" in type_name:
        kind = "text"
    elif "image" in type_name or (shape and len(shape) == 3 and shape[-1] in (1, 3, 4)):
        kind = "image"
    else:
        kind = infer_kind(path)

    flattened[path] = ModalitySpec(
        path=path,
        kind=kind,
        dtype=dtype_str,
        shape=shape,
        source="metadata",
    )
    return flattened


def _to_tensordict(episode: Any) -> Any:
    """Convert a raw TF episode to a TensorDict."""
    return dict_to_tensordict(_tf_to_torch(episode))


# ---------------------------------------------------------------------------
# GCS → local copy helpers
# ---------------------------------------------------------------------------

def _gcs_walk_files(src: str, dst: str) -> List[tuple]:
    pairs: List[tuple] = []
    src = src.rstrip("/")
    dst = dst.rstrip("/")
    for entry in tf.io.gfile.listdir(src):
        name = entry.strip("/")
        s = f"{src}/{name}"
        d = f"{dst}/{name}"
        if tf.io.gfile.isdir(s):
            pairs.extend(_gcs_walk_files(s, d))
        else:
            pairs.append((s, d))
    return pairs


def _needs_download(dst_file: str) -> bool:
    """Return True if dst_file is missing or empty (i.e. a failed partial download)."""
    if not tf.io.gfile.exists(dst_file):
        return True
    try:
        return tf.io.gfile.stat(dst_file).length == 0
    except Exception:
        return True


def _is_data_shard(filename: str) -> bool:
    """Return True if filename is a TFDS data shard (not a metadata file)."""
    return any(ext in filename for ext in (".tfrecord", ".riegeli", ".array_record"))


def _copy_tree(src: str, dst: str) -> None:
    """Download all files (metadata + data shards) from GCS to local dir."""
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    for src_file, dst_file in _progress(
        pairs, desc="Downloading dataset files", unit="file", total=len(pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


def _copy_metadata_only(src: str, dst: str) -> None:
    """Download only TFDS metadata files (dataset_info.json etc.), skipping data shards.

    Used when specific episodes are requested — the heavy shard files stay on
    GCS and are streamed on-demand during TED conversion.
    """
    _require_tf_stack()
    pairs = _gcs_walk_files(src, dst)
    metadata_pairs = [(s, d) for s, d in pairs if not _is_data_shard(s)]
    for src_file, dst_file in _progress(
        metadata_pairs, desc="Downloading dataset metadata", unit="file", total=len(metadata_pairs)
    ):
        dst_dir = str(Path(dst_file).parent)
        if not tf.io.gfile.exists(dst_dir):
            tf.io.gfile.makedirs(dst_dir)
        if _needs_download(dst_file):
            tf.io.gfile.copy(src_file, dst_file, overwrite=True)


# ---------------------------------------------------------------------------
# OXEDataset
# ---------------------------------------------------------------------------

class OXEDataset(BaseDatasetExperienceReplay):
    """OXE offline dataset as a TorchRL ``BaseDatasetExperienceReplay``.

    On first use the TFDS dataset is downloaded from the OXE GCS bucket and
    converted to TED (Trajectory Episode Data) format step-by-step, then
    persisted as memory-mapped tensors so subsequent runs skip both steps.
    Inheriting from ``BaseDatasetExperienceReplay`` (rather than bare
    ``ReplayBuffer``) provides:

    * **Immutability** — ``ImmutableDatasetWriter`` prevents accidental writes.
    * **``preprocess()``** — parallelised transform pipeline to normalise
      observations or fuse modalities, saving results to a new memmap.
    * **``delete()``** — clears the cached memmap from disk.
    * **``data_path`` / ``data_path_root``** — standardised path interface.
    * **Ecosystem fit** — recognised by TorchRL tooling the same way D4RL /
      Minari datasets are.

    TED layout per step::

        TensorDict({
            "observation":  TensorDict({...}),   # step[t] modalities (nested)
            "action":       Tensor,
            "done":         Tensor([1], bool),
            "terminated":   Tensor([1], bool),
            "next": TensorDict({
                "observation": TensorDict({...}), # step[t+1] obs (copy for last)
                "reward":      Tensor([1]),
                "done":        Tensor([1], bool),
                "terminated":  Tensor([1], bool),
            }),
            "collector": TensorDict({
                "episode_id": Tensor(int64),      # episode index
            }),
        })

    Usage::

        ds = OXEDataset("droid", episodes=[0, 1, 2], batch_size=32)
        batch = ds.sample()          # TensorDict(batch_size=[32])
        print(ds.num_episodes)       # 3
        print(len(ds))               # total steps

    Cache directory (priority order):
        1. ``root`` argument
        2. ``ROBOTDATASET_CACHE`` environment variable
        3. ``~/.cache/robotdataset``  (default)

    Args:
        dataset_name: OXE dataset name (e.g. ``"droid"``).
        split: TFDS split, e.g. ``"train"``.
        version: Specific dataset version; auto-selects latest when omitted.
        episodes: List of episode indices to include.  Only those episodes are
            converted; the full dataset is otherwise used.
        batch_size: Number of transitions returned by ``sample()``.
        root: Override cache root directory.
        load_str_fields: Whether to include string leaves (e.g. language
            instructions) in cached/sampled TensorDicts.
    """

    def __init__(
        self,
        dataset_name: str = "droid",
        split: str = "train",
        version: Optional[str] = None,
        episodes: Optional[List[int]] = None,
        batch_size: int = 32,
        root: Optional[str] = None,
        delta_timestamps: Optional[Dict[str, List[float]]] = None,
        control_frequency: float = 10.0,
        load_str_fields: bool = True,
    ) -> None:
        dataset_name = dataset_name.strip("/")

        if not validate_dataset_name(dataset_name, version):
            raise ValueError(
                f"Unknown dataset '{dataset_name}'"
                + (f" version '{version}'" if version else "")
                + f". Available datasets: {', '.join(sorted(list_datasets().keys()))}"
            )

        _require_tf_stack()

        self.dataset_name = dataset_name
        self.split = split
        self.version = version
        self.episodes: Optional[List[int]] = list(episodes) if episodes is not None else None
        self.dataset_path = dataset2path(dataset_name, version=version)
        self.root = _get_cache_dir(root)
        self.load_str_fields = load_str_fields

        # ------------------------------------------------------------------
        # 1. Sync metadata JSON files from GCS if not already present
        # ------------------------------------------------------------------
        local_dir = self._local_tfds_dir()
        info_file = local_dir / "dataset_info.json"
        if not info_file.exists() or info_file.stat().st_size == 0:
            _copy_metadata_only(self.dataset_path, str(local_dir))

        try:
            self.builder = tfds.builder_from_directory(builder_dir=str(local_dir))
            self.info = getattr(self.builder, "info", None)
            self._modalities = self._infer_modalities()
        except Exception as error:
            raise RuntimeError(
                f"Failed to load dataset '{dataset_name}' from '{local_dir}'. {error}"
            ) from error

        # ------------------------------------------------------------------
        # 2. Determine which episodes to load; find what's missing from cache
        # ------------------------------------------------------------------
        episodes_dir = self._episodes_dir()

        if self.episodes is not None:
            selected = sorted(self.episodes)
        else:
            selected = list(range(self._get_total_episodes()))

        missing = [i for i in selected if not is_episode_cached(episodes_dir / str(i))]

        # ------------------------------------------------------------------
        # 3. Download and convert only missing episodes
        # ------------------------------------------------------------------
        if missing:
            if self.episodes is not None:
                # Episode-selective: stream missing episodes directly from GCS.
                # No shard files are written locally.
                builder_for_data = tfds.builder_from_directory(builder_dir=self.dataset_path)
            else:
                # Full dataset: download shards to local cache (idempotent),
                # then read locally for conversion.
                _copy_tree(self.dataset_path, str(local_dir))
                builder_for_data = self.builder

            build_missing_episodes(
                builder=builder_for_data,
                split=split,
                episodes_dir=episodes_dir,
                missing=missing,
                tf_tensor_types=_TF_TENSOR_TYPES,
                load_str_fields=self.load_str_fields,
            )

        # ------------------------------------------------------------------
        # 4. Build (or reuse) the combined memmap, then load it lazily.
        #    Peak RAM during build = one episode; during training = batch_size.
        # ------------------------------------------------------------------
        self._loaded_indices: List[int] = selected
        combined_dir = self._combined_dir(selected)
        if not is_combined_complete(combined_dir):
            build_combined_storage(selected, episodes_dir, combined_dir)
        combined_td = TensorDict.load_memmap(str(combined_dir / "data"))
        # Ensure nested key structure — some tensordict versions may load memmaps
        # with flat "a/b" keys instead of nested sub-TensorDicts.
        combined_td = combined_td.unflatten_keys("/")
        storage = TensorStorage(combined_td)
        _flat_combined = combined_td.flatten_keys("/")
        self._storage_keys: set = set(_flat_combined.keys())

        # ------------------------------------------------------------------
        # 5. Temporal sampler — always active (compulsory per spec).
        #    Built before super().__init__() so it can be passed as the
        #    buffer's sampler and satisfy the Sampler ABC contract.
        #
        # Effective delta_timestamps:
        #   • Start with {key: [0.0]} for every *tensor* modality (T=1, just
        #     the anchor step).  Non-tensor leaves (e.g. language_instruction
        #     stored as NonTensorStack) are excluded — they can't be indexed
        #     with a 2-D index tensor.
        #   • Caller-supplied delta_timestamps overrides per-key.
        # Image modalities are identified by kind="image" so the sampler can
        # permute them from on-disk HWC → CHW (channels first).
        # ------------------------------------------------------------------
        # Build defaults for "data" modalities only — exclude system/next-step
        # fields that should not have a temporal dimension.
        _EXCLUDED_PREFIXES = ("next/", "collector/")
        _EXCLUDED_KEYS = {"done", "terminated"}
        default_dt: Dict[str, List[float]] = {
            key: [0.0]
            for key in self._storage_keys
            if isinstance(_flat_combined.get(key), torch.Tensor)
            and not any(key.startswith(p) for p in _EXCLUDED_PREFIXES)
            and key not in _EXCLUDED_KEYS
        }
        # Caller-supplied values take precedence
        effective_dt = {**default_dt, **(delta_timestamps or {})}

        self._episode_starts: Dict[int, int]
        self._episode_lengths: Dict[int, int]
        self._episode_starts, self._episode_lengths = (
            TemporalSampler.build_episode_index(combined_td)
        )
        self._temporal_sampler = TemporalSampler(
            delta_timestamps=effective_dt,
            control_frequency=control_frequency,
            image_keys=self.image_keys,
        )

        super().__init__(
            storage=storage,
            sampler=self._temporal_sampler,
            writer=ImmutableDatasetWriter(),
            batch_size=batch_size,
        )

    # ------------------------------------------------------------------
    # BaseDatasetExperienceReplay abstract interface
    # ------------------------------------------------------------------

    @property
    def data_path(self) -> Path:
        """Per-episode TED memmap directory for the current split."""
        return self._episodes_dir()

    @property
    def data_path_root(self) -> Path:
        """Root path for all cached data for this dataset."""
        return self._local_tfds_dir()

    def _is_downloaded(self) -> bool:
        episodes_dir = self._episodes_dir()
        return all(is_episode_cached(episodes_dir / str(i)) for i in self._loaded_indices)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _local_tfds_dir(self) -> Path:
        d = self.root / "oxe" / self.dataset_name
        if self.version:
            d = d / self.version
        return d

    def _episodes_dir(self) -> Path:
        episodes_dir = self._local_tfds_dir() / "episodes" / self.split
        if not self.load_str_fields:
            episodes_dir = episodes_dir / "no_str_fields"
        return episodes_dir

    def _combined_dir(self, selected: List[int]) -> Path:
        key = combined_dir_key(selected)
        combined_dir = self._local_tfds_dir() / "combined" / self.split / key
        if not self.load_str_fields:
            combined_dir = combined_dir / "no_str_fields"
        return combined_dir

    def _get_total_episodes(self) -> int:
        """Return total episode count for the current split from builder.info."""
        if self.info is not None:
            splits = getattr(self.info, "splits", {})
            split_info = splits.get(self.split)
            if split_info is not None:
                count = getattr(split_info, "num_examples", None)
                if count is not None:
                    return int(count)
        raise RuntimeError(
            f"Cannot determine episode count for '{self.dataset_name}/{self.split}' "
            "from builder.info. Pass an explicit episodes list."
        )

    def _infer_modalities(self) -> Dict[str, Dict[str, Any]]:
        features = None
        if self.info is not None:
            features = getattr(self.info, "features", None)
        if features is None:
            return {}
        flattened = _flatten_tfds_features(features)
        return {path: asdict(spec) for path, spec in flattened.items()}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def num_episodes(self) -> int:
        """Number of episodes loaded into this dataset."""
        return len(self._loaded_indices)

    @property
    def image_keys(self) -> List[str]:
        """Slash-separated keys whose tensors are stored as HWC images, sorted for determinism."""
        return sorted(
            path
            for path, spec in self._modalities.items()
            if spec.get("kind") == "image" and path in self._storage_keys
        )

    @property
    def modalities(self) -> Dict[str, List[str]]:
        """Modality paths grouped by inferred kind.

        Returns a dict with keys from ``{"image", "text", "action", "state"}``
        (only kinds that are present in this dataset are included).
        Each value is a sorted list of slash-separated field paths, e.g.
        ``"observation/image"``.  System fields (reward, done, episode_id,
        etc.) are excluded — use ``get_modalities()`` for the raw spec.
        """
        _KINDS = {"image", "text", "state", "action"}
        grouped: Dict[str, List[str]] = {}
        for path, spec in self._modalities.items():
            kind = spec.get("kind", "generic")
            if kind not in _KINDS:
                continue
            grouped.setdefault(kind, []).append(path)
        return {k: sorted(v) for k, v in grouped.items()}

    def get_modalities(self) -> Dict[str, Dict[str, Any]]:
        """Raw per-path modality specs (path → ModalitySpec dict)."""
        return dict(self._modalities)

    def get_dataset_info(self) -> Dict[str, Any]:
        if self.info is None:
            return {}
        return {
            "description": getattr(self.info, "description", ""),
            "features": getattr(self.info, "features", {}),
            "splits": getattr(self.info, "splits", {}),
        }

    def _sample(self, batch_size: int) -> Any:
        batch = self._temporal_sampler(
            self._storage._storage,
            self._episode_starts,
            self._episode_lengths,
            batch_size,
        )
        return batch, {}

    def set_sampler(self, sampler: "TemporalSampler") -> None:
        """Replace the temporal sampler.

        Can be called after construction to change ``delta_timestamps`` or
        ``control_frequency`` without rebuilding the dataset.

        Args:
            sampler: A :class:`TemporalSampler` configured with the desired
                ``delta_timestamps`` and ``control_frequency``.
        """
        self._temporal_sampler = sampler
        self._sampler = sampler
