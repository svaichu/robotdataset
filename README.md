# robotdataset

A Python package for loading robot learning datasets, with support for the [Open X-Embodiment (OXE)](./doc/OXE.md) collection.

## Installation

```bash
pip install "robotdataset[oxe]"
```

Or from source:

```bash
git clone https://github.com/robotics-action-group/robotdataset.git
cd robotdataset
pip install -e ".[oxe]"
```

**Requirements:** Python >= 3.7, PyTorch >= 1.13.1, TensorFlow >= 2.11.1, TensorFlow Datasets >= 4.8.2

## Quick Start

```python
from robotdataset import OXEDataset

dataset = OXEDataset(dataset_name='droid', split='train')

print(len(dataset))
print(dataset.get_dataset_info())

for sample in dataset:
    print(sample.keys())
    break
```

## License

MIT — see [LICENSE](LICENSE).
