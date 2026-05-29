"""RLData - A Python package for robot learning dataset handling.

This package provides utilities for loading and handling robot learning datasets,
with support for the OXE (Open X-Embodiment) dataset collection from Google Cloud
and HuggingFace datasets.
"""

from robotdataset.oxe_dataset import (
    OXEDataset,
    dataset2path,
    list_datasets,
    validate_dataset_name,
    TemporalSampler,
)
from robotdataset.table30v2_dataset import Table30v2Dataset
from robotdataset.agibot_dataset import AgiBotWorldBetaDataset
from robotdataset.agibot.loader import list_agibot_tasks
from robotdataset.utils import batchViz, itemViz

try:
    from robotdataset.robomimic_dataset import RobomimicDataset
except Exception:
    RobomimicDataset = None  # type: ignore[assignment,misc]

__all__ = [
    'OXEDataset',
    'Table30v2Dataset',
    'AgiBotWorldBetaDataset',
    'RobomimicDataset',
    'dataset2path',
    'list_datasets',
    'list_agibot_tasks',
    'validate_dataset_name',
    'TemporalSampler',
    'batchViz',
    'itemViz',
]

__version__ = '0.1.0'
__author__ = 'Robotics Action Group'
