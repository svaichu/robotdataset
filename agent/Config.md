# Flexible Config System for WorldModel and VLA Training

AGENT.md is the main entrypoint for agent.
Use robotdataset/configuration_system dir.

## Overview
A **fluent builder pattern** for configuring downstream tasks like worldmodel and VLA training. The system:
- Uses **groups** (e.g., `dataset`, `training`) as top-level categories.
- Supports **fields** (e.g., `batch_size`, `learning_rate`) as key-value pairs within groups.
- Allows **methods** (e.g., `dataset(...)`, `training(...)`) to update groups dynamically, where the method name matches the group name.
- Supports loading configurations from YAML/JSON files.

## Requirements
- **Dynamic Group/Field Learning**: After loading a YAML/JSON file, the `Config` object must:
  - **Learn group names** (e.g., `dataset`, `training`) dynamically from the file.
  - **Learn field names** (e.g., `batch_size`, `learning_rate`) and their **types** (e.g., `int`, `float`, `str`, `bool`, `list`, or any Python type).
  - **Generate methods** dynamically for each group (e.g., `config.dataset(...)` updates the `dataset` group).
- **Field Subfields**: Every learned field has two required subfields:
  - `type`: the field's data type.
  - `default`: the field's default value.
  A plain YAML/JSON value (e.g., `batch_size: 32`) is sufficient — `type` is inferred and `default` is set to the value itself. The file *may* also spell a field out explicitly as `{type: ..., default: ...}`, but this is optional, not required.
- **Hyperparameter Opt Settings (bounds/values)**: A field's YAML/JSON entry need not contain any hyperparameter search-space info. The intended flow is:
  1. Load the YAML/JSON file as-is (fields only carry `type`/`default`).
  2. Attach hyperparameter opt settings afterward via dedicated methods on the `Config` (or its groups), e.g. `config.set_bounds("training", "learning_rate", min=1e-5, max=1e-2)` or `config.training.set_values("optimizer", ["adam", "sgd"])`.
  - A field's YAML/JSON entry *may* still declare `bounds`/`values` directly (e.g. `learning_rate: {type: "float", default: 1e-4, bounds: {min: 1e-5, max: 1e-2}}`) — the file format supports it, but it is not required.
- **W&B Sweep Compatibility**: The config system must support exporting groups to a W&B-compatible sweep configuration format (e.g., `sweep.yaml`). This includes:
  - Mapping groups/fields to W&B parameters (e.g., `training.learning_rate: {min: 1e-5, max: 1e-2}`).
  - Supporting continuous ranges (`min`/`max`), discrete values (`values`), and categorical values (`values`). Only fields with `bounds`/`values` set (via the file or via the update methods) are sweepable; others export as a fixed `value`.
- **No Silent Creation of Unknown Groups/Fields**: A group or field must already exist — via a loaded file or an explicit `config.define(group, field, default=...)` call — before it can be set through the fluent API. Calling `config.dataset(...)` for an unknown group, or setting an unknown field on a known group, raises a clear error naming the missing group/field instead of silently creating it.
- **argparse / CLI Overrides**: Every known field is exposed as a dotted, typed `--<group>.<field>` command-line option (standard AI-training pattern; same keys as the W&B sweep export, so `wandb agent` command lines like `--training.learning_rate=0.001` parse directly). Precedence is defaults < config file < CLI — unpassed options keep their config values. Entry points:
  - `Config.from_cli(argv=None, default_config=None)` — train-script one-liner handling `--config <file>` plus overrides.
  - `add_arguments(parser, groups=None)` / `apply_args(args)` — compose with an existing `argparse` parser; `apply_args` also accepts a plain dict of dotted keys (e.g. `wandb.config`) with string values coerced through each field's learned type.
  - `parse_args(argv=None, parser=None, strict=True)` — one-shot parse-and-apply on an existing `Config`.
  - Values convert per learned type (`bool` accepts bare flag or `true/false/1/0/...`; `list` accepts JSON arrays or comma-separated strings; `dict` accepts JSON objects); fields with `values` become argparse `choices`. Unknown `--group.field` overrides error rather than silently creating fields, matching the fluent API's strictness. Implementation lives in `configuration_system/cli.py`.
