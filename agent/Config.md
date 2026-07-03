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
  - **Learn field names** (e.g., `batch_size`, `learning_rate`) and their **types** (e.g., `int`, `float`, `str`, `list`).
  - **Support subfields** for each field to store hyperparameter bounds (e.g., `learning_rate: {type: "float", bounds: {min: 1e-5, max: 1e-2}}`).
  - **Generate methods** dynamically for each group (e.g., `config.dataset(...)` updates the `dataset` group).
- **W&B Sweep Compatibility**: The config system must support exporting groups to a W&B-compatible sweep configuration format (e.g., `sweep.yaml`). This includes:
  - Mapping groups/fields to W&B parameters (e.g., `training.learning_rate: {min: 1e-5, max: 1e-2}`).
  - Supporting continuous ranges (`min`/`max`), discrete values (`values`), and categorical values (`values`). 
