# Dataset info

`robotdataset.info.DatasetInfo` is the model-agnostic first layer: it tells you what a loader holds,
and its YAML form is the file you edit before handing the dataset to a [model adapter](adapters.md).

```
loader ──▶ DatasetInfo ──▶ modalities.yaml ──(you edit it)──▶ model adapter ──▶ model inputs + statistics
```

## Listing a dataset

```python
from robotdataset.info import DatasetInfo

info = DatasetInfo.from_loader(
    "LeRobotVideoDataset", {"local_path": "/data/stack_without_ft_tact_v4"}, episodes=[0], stats=True
)
print(info.table())
info.to_yaml("modalities.yaml")
```

or from a shell (loader arguments as JSON; `--episodes` takes `0:2` or `0,3,5`):

```
python -m robotdataset.info --loader OXEDataset --kwargs '{"dataset_name": "viola"}' \
    --episodes 0:2 --stats -o viola.yaml
```

`from_loader` builds the named loader (`LeRobotVideoDataset`, `Table30v2Dataset`, `OXEDataset`,
`AgiBotWorldBetaDataset`); a couple of episodes are enough to list the modalities. `from_dataset(ds)`
describes a loader you already built. The table shows one row per modality; with statistics it adds
`std min..max` and `const dims` columns.

## What is listed

- Every leaf the loader's sampler really returns, with `kind` (`image`, `state`, `action`, `text`,
  `generic`; robotdataset infers it from the path name), `dtype` and the per-step `shape` (images are HWC
  as stored; a scalar leaf has shape `[]`; a dimension that metadata leaves open, as on OXE, is read from a
  sampled batch).
- String leaves the sampler returns but the loader's metadata does not describe (for example
  `observation/language_instruction` on `LeRobotVideoDataset`) are listed as `kind: text`, `dtype: string`.
- Not listed: TED bookkeeping (`next/...`, `collector/...`, `done`, `terminated`, `truncated`, `reward`,
  `discount`, `is_first`, `is_last`, `is_terminal`, `episode_id`).
- The loader differences are normalised: OXE groups `ds.modalities` by kind and returns `image_keys` as strings,
  the other loaders return path-to-spec mappings and tuple keys.
- `episodes_loaded` / `steps_loaded` / `fps` describe the episodes the listing was built from, not the whole dataset.

## Statistics

`stats=True` (or `info.with_stats(ds, exact=..., keys=...)`) attaches the loader's per-leaf statistics
(`mean`, `std`, `min`, `max`, `q01`, `q99`, `count`) to every float, non-image leaf. They help with picking
(a leaf with constant columns is rarely useful). They are computed over exactly the episodes the loader holds,
kept in memory on `ModalityInfo.stats`, and never written to the YAML: a stored copy would not say which
episodes it covers.

## The YAML file

```yaml
dataset:
  loader: LeRobotVideoDataset
  kwargs: {local_path: /data/stack_without_ft_tact_v4}
  episodes_loaded: 1
  steps_loaded: 278
  fps: 15.0
modalities:                          # the order of this list is the order handed to the model
  - {path: observation/images/primary, kind: image, dtype: uint8, shape: [256, 256, 3], use: true, name: primary}
  - {path: observation/state/cartesian, kind: state, dtype: float32, shape: [6], use: true, name: cartesian}
  - {path: action, kind: action, dtype: float32, shape: [7], use: true, name: action, split: [{name: eef, columns: [0, 6]}, {name: gripper, columns: [6, 7]}]}
```

| field | meaning |
|---|---|
| `path`, `kind`, `dtype`, `shape` | facts about the leaf; checked against the loader when the file is loaded (`check_against`) |
| `use` | `true` on every modality the model should get. Everything starts as `false`: you pick explicitly |
| order | the order of the list is the order handed to the model |
| `name` | model-side group name; defaults to the shortest unique path suffix (`observation/state/all` -> `all`; `left/pose` and `right/pose` -> `left_pose`, `right_pose`) |
| `role` | optional override of the role implied by `kind` (`video`, `state`, `action`, `language`), for a leaf whose name robotdataset's heuristic misjudged |
| `split` | optional; cuts one vector leaf into named groups by column range `[start, end)` |

Unknown fields, duplicate paths, bad splits and unknown roles are rejected with the offending entry named. A
model adapter adds its own block below `modalities:` (for example `gr00t:`), which round-trips untouched.

Example notebook (dataset info -> GR00T N1.7 adapter on OXE `viola`, down to the processor inputs): [`example/example_gr00t_adapter.ipynb`](../example/example_gr00t_adapter.ipynb)
