"""RLData - A Python package for robot learning dataset handling.

This package provides utilities for loading and handling robot learning datasets,
with support for the OXE (Open X-Embodiment) dataset collection from Google Cloud
and HuggingFace datasets.
"""

from robotdataset.configuration_system import Config, FieldSpec

try:
    from robotdataset.oxe_dataset import (
        OXEDataset,
        dataset2path,
        list_datasets,
        validate_dataset_name,
        EpisodeTubeletSampler,
        TemporalSampler,
    )
except Exception:  # pragma: no cover - heavy deps (torch/torchrl) not installed
    OXEDataset = None  # type: ignore[assignment]
    dataset2path = None  # type: ignore[assignment]
    list_datasets = None  # type: ignore[assignment]
    validate_dataset_name = None  # type: ignore[assignment]
    EpisodeTubeletSampler = None  # type: ignore[assignment]
    TemporalSampler = None  # type: ignore[assignment]
try:
    from robotdataset.oxe_jax_dataset import OXEJAXDataset, JAXTemporalSampler
except Exception:  # pragma: no cover - optional dependency
    OXEJAXDataset = None  # type: ignore[assignment]
    JAXTemporalSampler = None  # type: ignore[assignment]
try:
    from robotdataset.table30v2_dataset import Table30v2Dataset
    from robotdataset.agibot_dataset import AgiBotWorldBetaDataset
    from robotdataset.agibot.loader import list_agibot_tasks
    from robotdataset.utils import batchViz, episodeViz, itemViz
except Exception:  # pragma: no cover - heavy deps (torch/torchrl) not installed
    Table30v2Dataset = None  # type: ignore[assignment]
    AgiBotWorldBetaDataset = None  # type: ignore[assignment]
    list_agibot_tasks = None  # type: ignore[assignment]
    batchViz = None  # type: ignore[assignment]
    episodeViz = None  # type: ignore[assignment]
    itemViz = None  # type: ignore[assignment]

__all__ = [
    'Config',
    'FieldSpec',
    'OXEDataset',
    'OXEJAXDataset',
    'Table30v2Dataset',
    'AgiBotWorldBetaDataset',
    'dataset2path',
    'list_datasets',
    'list_agibot_tasks',
    'validate_dataset_name',
    'EpisodeTubeletSampler',
    'TemporalSampler',
    'JAXTemporalSampler',
    'batchViz',
    'itemViz',
    'episodeViz',
]

__version__ = '0.1.0'
__author__ = 'Robotics Action Group'
