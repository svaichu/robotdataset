"""Model adapters: turn a :class:`~robotdataset.info.DatasetInfo` into what one model consumes.

``robotdataset.info`` is model-agnostic.  An adapter is the model-specific second layer: it
reads the (edited) dataset-info file, selects and orders the modalities the model gets,
supplies the sampling windows, builds the loader, converts sampled TED batches into the
model's input structure and computes the statistics the model's preprocessing needs, all in
memory and from the loader itself.

Adapters are optional modules (their model package is imported lazily), so ``robotdataset``
never depends on a model library.  Available:

* :mod:`robotdataset.adapters.gr00t` -- NVIDIA Isaac-GR00T (``Gr00tAdapter``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

from robotdataset.info import DatasetInfo, build_loader


class ModelAdapter:
    """Base class: holds the dataset info and its model block, builds the loader with the windows.

    Subclasses set ``name`` (the YAML block that carries the model's settings), ``YAML_HELP``
    (comment text for that block) and implement :meth:`delta_timestamps`, :meth:`bind` and
    :meth:`with_defaults`.
    """

    name: str = ""
    YAML_HELP: str = ""

    def __init__(self, info: DatasetInfo) -> None:
        self.info = info
        self.block: Dict[str, Any] = dict(info.extra.get(self.name, {}) or {})

    @classmethod
    def from_info(cls, info: DatasetInfo) -> "ModelAdapter":
        return cls(info)

    @classmethod
    def from_yaml(cls, path: str) -> "ModelAdapter":
        """Load the edited dataset-info file."""
        return cls(DatasetInfo.from_yaml(path))

    @classmethod
    def with_defaults(cls, info: DatasetInfo) -> DatasetInfo:
        """``info`` with this adapter's default settings block added (written by ``python -m robotdataset.info``)."""
        raise NotImplementedError

    def delta_timestamps(self) -> Dict[str, List[float]]:
        """``{TED path: offsets}`` counted in steps (pair with ``control_frequency=1.0``)."""
        raise NotImplementedError

    def bind(self, ds: Any) -> None:
        """Check the selection against a built loader and remember what is needed to read its batches."""
        raise NotImplementedError

    def build_dataset(self, episodes: Optional[Sequence[int]], batch_size: int) -> Any:
        """Build the loader named in the file with this adapter's windows, then :meth:`bind` it.

        Windows are passed in steps with ``control_frequency=1.0`` (robotdataset converts offsets
        with ``round(dt * control_frequency)``), so the loader's own frame rate is irrelevant.
        """
        ds = build_loader(
            self.info.loader,
            self.info.kwargs,
            episodes=episodes,
            batch_size=batch_size,
            delta_timestamps=self.delta_timestamps(),
            control_frequency=1.0,
        )
        self.bind(ds)
        return ds
