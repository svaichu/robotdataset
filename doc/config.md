# Config

```python
from robotdataset import Config
```

`Config` is a fluent, self-learning config builder for training runs (dataset,
training hyperparameters, etc.). It groups fields under top-level categories
(e.g. `dataset`, `training`), learns each field's name/type/default from a
YAML/JSON file or an explicit `define()` call, and generates group methods
dynamically (`config.dataset(...)`, `config.training(...)`).

Unlike a plain dict or `argparse.Namespace`, a group or field must exist —
either loaded from a file or registered with `define()` — before it can be
set through the fluent API. Setting an unknown group or unknown field raises
a clear error instead of silently creating it.

## Quick start

```python
from robotdataset import Config

cfg = Config()
cfg.define("dataset", "name", default="oxe")
cfg.define("dataset", "batch_size", default=32)
cfg.define("training", "learning_rate", default=1e-4)

cfg.dataset(name="oxe", batch_size=64).training(learning_rate=3e-4)

cfg.dataset.name          # "oxe"
cfg.dataset.batch_size    # 64
cfg.groups()               # ["dataset", "training"]
cfg.fields("dataset")      # ["name", "batch_size"]
```

## Loading from a file

Loading a YAML/JSON file teaches the config its groups, field names, and each
field's `type`/`default`. A plain value (e.g. `batch_size: 32`) is enough —
`type` is inferred and `default` is set to the value itself:

```yaml
# config.yaml
dataset:
  name: oxe
  batch_size: 32
  shuffle: true

training:
  learning_rate: 1.0e-4
  optimizer: torch.optim.Adam
  num_epochs: 100
```

```python
cfg = Config.from_file("config.yaml")   # dispatches on extension (.yaml/.yml/.json)
# or explicitly:
cfg = Config.from_yaml("config.yaml")
cfg = Config.from_json("config.json")
cfg = Config.from_dict({"dataset": {"name": "oxe"}})
```

`from_yaml` requires `pyyaml` (`pip install robotdataset[config]` or
`pip install pyyaml`); it's imported lazily so plain dict/JSON usage needs no
extra dependency.

A field may also be spelled out explicitly as `{type: ..., default: ...}` in
the file, which is required if you want to declare `bounds`/`values` (see
below) directly in the file rather than attaching them afterward.

## Reading and updating values

```python
cfg.dataset.name              # attribute access
cfg.dataset["name"]           # item access
"name" in cfg.dataset         # membership check
cfg.dataset.to_dict()         # {"name": "oxe", "batch_size": 32, ...}

cfg.dataset(name="libero")    # update; returns the parent Config, so calls chain
cfg.dataset(name="libero").training(learning_rate=5e-4)
```

Setting an unregistered group or field raises:

```python
cfg.dataset(missing_field=1)
# KeyError: Unknown field 'dataset.missing_field'; define it with
# Config.define('dataset', 'missing_field', ...) or load it from a file first

cfg.unknown_group(x=1)
# AttributeError: Unknown group 'unknown_group'; define it with
# Config.define('unknown_group', ...) or load it from a file first
```

## Hyperparameter opt settings (bounds / values)

A field's file entry doesn't need to carry any search-space info. The usual
flow is to load the plain config, then attach bounds/values afterward — this
is what makes a field eligible for W&B sweep export:

```python
cfg.set_bounds("training", "learning_rate", min=1e-5, max=1e-2)
cfg.set_values("training", "optimizer", ["adam", "sgd"])

# equivalent group-level shorthand
cfg.training.set_bounds("learning_rate", min=1e-5, max=1e-2)
cfg.training.set_values("optimizer", ["adam", "sgd"])
```

## Exporting

```python
cfg.to_dict()             # nested dict; sweepable fields export as {type, default, bounds/values}
cfg.save("out.yaml")      # or "out.json"
```

## W&B sweep export

Fields with `bounds` become continuous ranges, fields with `values` become
discrete/categorical choices, and any other field is exported as a fixed
`value` from the current config:

```python
sweep = cfg.to_sweep(
    method="bayes",
    metric={"name": "loss", "goal": "minimize"},
    groups=["training"],   # optional: restrict to specific groups
)
cfg.to_sweep_file("sweep.yaml", method="bayes", metric={"name": "loss", "goal": "minimize"})
```

```python
sweep["parameters"]["training.learning_rate"]  # {"min": 1e-5, "max": 1e-2}
sweep["parameters"]["training.optimizer"]      # {"values": ["adam", "sgd"]}
sweep["parameters"]["training.num_epochs"]     # {"value": 100}  (fixed, not swept)
```

## API summary

| Member | Description |
|---|---|
| `Config()` | Empty config; groups/fields must be `define()`d or loaded before use |
| `define(group, field, default=None, type=None, **extra)` | Register a field, creating its group if needed |
| `groups()` / `fields(group)` | List known group / field names |
| `schema(group, field)` | Return the field's `FieldSpec` (`type`, `default`, `bounds`, `values`) |
| `set_bounds(group, field, min=None, max=None, **extra)` | Attach a continuous search range to an existing field |
| `set_values(group, field, values)` | Attach a discrete/categorical value set to an existing field |
| `Config.from_dict(data)` / `from_yaml(path)` / `from_json(path)` / `from_file(path)` | Load groups/fields from a dict or file |
| `to_dict()` | Export the current config as a nested dict |
| `save(path)` | Write to `.yaml`/`.yml`/`.json` |
| `to_sweep(method="bayes", metric=None, groups=None)` / `to_sweep_file(path, ...)` | Build/write a W&B-compatible sweep config |

`config.<group>(**fields)` (e.g. `config.dataset(...)`) updates a group and
returns the parent `Config` for chaining; `config.<group>.<field>` reads the
current value.
