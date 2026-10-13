# Model adapters

A model adapter is the model-specific second layer. It reads the edited [dataset-info](dataset_info.md) file,
selects and orders the modalities the model gets, supplies the sampling windows, builds the loader, converts
sampled TED batches into the model's input structure and computes the statistics the model's preprocessing
needs. Everything is in memory and comes from the loader; no `meta/*.json` file is read or written.

Adapters are optional modules and import their model package lazily, so `robotdataset` never depends on a
model library. To add one, subclass `robotdataset.adapters.ModelAdapter` (it holds the `DatasetInfo`, reads the
adapter's block of the YAML, builds the loader with the windows and calls `bind` to validate it) and register
its name in `robotdataset.info.ADAPTERS` so that `python -m robotdataset.info --adapter <name>` can write its
default block.

## Isaac-GR00T: `Gr00tAdapter`

```
python -m robotdataset.info --loader LeRobotVideoDataset --kwargs '{"local_path": "..."}' \
    --adapter gr00t -o modality.yaml          # then edit: use / order / name / split / gr00t block
```

```python
from robotdataset.adapters.gr00t import Gr00tAdapter

adapter = Gr00tAdapter.from_yaml("modality.yaml")
train_ds = adapter.build_dataset(episodes=range(46), batch_size=8)     # loader + windows + checks
processor = Gr00tN1d7Processor(modality_configs={tag: adapter.modality_configs()}, ...)
adapter.configure_processor(processor, train_ds, tag)                    # statistics, in memory
steps = adapter.to_steps(train_ds.sample(), embodiment_tag)              # list of VLAStepData
```

`to_steps` and `to_model_batch` take an `EmbodimentTag` or its name or value (`"NEW_EMBODIMENT"`, `"new_embodiment"`);
the statistics and the processor's `modality_configs` are keyed by the value.

### Selection

The selected modalities (`use: true`), in file order, become GR00T groups. The role comes from `kind`
(`image` -> video, `state` -> state, `action` -> action, a string `text` leaf -> language) unless `role:` says
otherwise. A group is named by `name` (or the last path component). A `split` cuts one vector leaf into several
groups. Image leaves cannot be split. A video group must be a uint8 HWC leaf (`kind: image`, or any uint8 HWC leaf
with an explicit `role: video`). State and action groups must be float vectors of known width. At most one text
leaf is selected; with none, `gr00t.language.default` supplies the instruction.

`adapter.index_map()` gives each group's column range in its leaf and its position in the assembled state / action
vector (groups concatenated in order); this is also how a predicted action splits back into groups.

### The `gr00t:` block

```yaml
gr00t:
  action_horizon: 16                 # action delta_indices = 0 .. horizon-1
  video_delta: [0]                   # step offsets relative to the anchor
  state_delta: [0]
  language:
    default: null                    # instruction used when no text modality is selected
    modality_key: annotation.human.task_description   # the name GR00T looks the text up under
  action:                            # per action group; a group left out is ABSOLUTE / NON_EEF / DEFAULT
    eef: {rep: RELATIVE, type: EEF, format: XYZ_ROTVEC, state_key: cartesian}
```

`rep` (`ABSOLUTE`, `RELATIVE`, `DELTA`), `type` (`EEF`, `NON_EEF`) and `format` (`DEFAULT`, `XYZ_ROT6D`, `XYZ_ROTVEC`)
are GR00T's own `ActionConfig` fields; `state_key` names a state group and is required for `RELATIVE`.

### Windows

GR00T's `delta_indices` count steps. `delta_timestamps()` returns them per TED leaf, and `build_dataset` passes them
to the loader with `control_frequency=1.0`: robotdataset converts offsets with `round(dt * control_frequency)`, so the
offsets arrive unchanged and no dataset's frame rate is needed. One leaf cannot be read at two different windows.

### Statistics

`adapter.statistics(ds, embodiment_tag)` returns what `Gr00tN1d7Processor.set_statistics` takes
(`{tag: {"state", "action", "relative_action"}}`); `configure_processor` passes it with `override=True`.

- Per-leaf statistics come from `ds.generate_stats` over exactly the episodes `ds` holds, so build the training loader
  from the training episodes only. Quantiles are exact by default (`exact=True`), which matters because GR00T
  normalises with q01/q99.
- The selection splits them into groups, in order, with the same layout code as `to_gr00t_statistics`.
- Every `RELATIVE` group gets `ds.generate_relative_action_stats` (one call per action leaf). Supported: `EEF` +
  `XYZ_ROTVEC` and `NON_EEF` + `DEFAULT`. The reference state must be a whole state leaf of the same width as the group
  (not a column slice), otherwise `NotImplementedError` explains why. `DELTA` groups are not supported.
- A group with a zero q01..q99 range in some column raises a warning: percentile normalisation cannot scale it.

### Checks

`build_dataset` (or `bind(ds)` for a loader built elsewhere) compares the file with a fresh listing of the loader
(paths, dtype and shape of the selected modalities) and the selection with a sampled batch (windows, shapes, column
bounds). Every problem is reported in one `ValueError` that says what to fix.

`robotdataset.gr00t` still exports `to_gr00t_statistics`, `check_against` and `GR00T_STAT_KEYS`; their code now lives in
`robotdataset.adapters.gr00t`.
