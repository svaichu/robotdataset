# OXE Datasets — `OXEDataset`

> **Status: alpha — usable but not stable.** The API works end-to-end but may
> change without notice between releases.

Loads datasets from the [Open X-Embodiment](https://robotics-transformer-x.github.io/)
collection hosted on the public GCS bucket `gs://gresearch/robotics`, and exposes
them as a TorchRL `BaseDatasetExperienceReplay` in TED format.

**Requires:** `tensorflow` and `tensorflow-datasets` (install with
`pip install "robotdataset[oxe]"`).

## Discovering datasets

```python
from robotdataset import list_datasets, validate_dataset_name, dataset2path

list_datasets()
# {'viola': ['0.1.0'], 'bridge_data_v2': ['0.0.1'], 'droid': ['1.0.1', '1.0.0'], ...}

validate_dataset_name("droid")            # True
validate_dataset_name("droid", "9.9.9")   # False

dataset2path("droid")                     # 'gs://gresearch/robotics/droid/1.0.1'
dataset2path("droid", version="1.0.0")    # 'gs://gresearch/robotics/droid/1.0.0'
```

- `list_datasets(refresh=False)` returns `{name: [versions]}`, versions sorted
  newest-first. The bucket index is cached in memory; pass `refresh=True` to rescan.
- `dataset2path` resolves to the latest version when `version` is omitted.

## Loading

```python
from robotdataset import OXEDataset

# Full train split — downloads all shards to the local cache
dataset = OXEDataset(dataset_name="viola", split="train", batch_size=16)

# Specific episodes only — streams those episodes from GCS, no full download
dataset = OXEDataset(
    dataset_name="cmu_playing_with_food",
    episodes=[0, 2],
    batch_size=6,
    control_frequency=5,    # Hz of the source data
)
```

### Constructor arguments

| Argument | Default | Description |
|---|---|---|
| `dataset_name` | `"droid"` | OXE dataset name (see `list_datasets()`) |
| `split` | `"train"` | TFDS split |
| `version` | latest | Specific dataset version |
| `episodes` | all | Episode indices to load. When given, shards are streamed from GCS instead of fully downloaded |
| `batch_size` | `32` | Transitions per `sample()` call |
| `root` | cache default | Override cache root directory |
| `delta_timestamps` | `{key: [0.0]}` | Temporal window per modality, in seconds (see [Samplers](../samplers.md)) |
| `control_frequency` | `10.0` | Steps per second, converts time deltas to step offsets |
| `load_str_fields` | `True` | Include string leaves (e.g. language instructions) in the cache and batches |

On first use the loader:

1. downloads TFDS metadata (`dataset_info.json`, features) — always cheap;
2. downloads/streams and converts only the **missing** episodes to per-episode TED
   memmaps;
3. assembles (or reuses) a combined memmap for the selected episode set;
4. loads it lazily via `TensorDict.load_memmap`.

All steps are idempotent and resumable; interrupted downloads are detected
(zero-length files) and retried.

## Sampling

```python
batch = dataset.sample()                  # or: batch = next(iter(dataset))
batch["observation/image"].shape          # (6, 480, 640, 3) — (B, H, W, C), T=1 default
batch["action"].shape                     # (6, 8)
batch["language_instruction"]             # NonTensorStack(['Grasp the carrot slice.', ...])
batch["collector/episode_id"]             # which episode each row came from
```

Batch keys are always **`"/"`-separated strings** — the flat view of the nested TED
structure.  For example, `"observation/image"`, `"next/observation/image"`,
`"collector/episode_id"`.

With a temporal window (set in the constructor or via `set_sampler`):

```python
from robotdataset import TemporalSampler

sampler = TemporalSampler(
    delta_timestamps={"observation/image": [-0.2, -0.1, 0.0]},
    control_frequency=10,
    image_keys=dataset.image_keys,        # enables HWC → CHW permutation
)
dataset.set_sampler(sampler)

batch = dataset.sample()
batch["observation/image"].shape           # (B, 3, C, H, W)
batch["next/observation/image"].shape      # (B, 3, C, H, W) — mirrored future window
```

## Inspecting the dataset

```python
dataset.num_episodes        # 2
len(dataset)                # 244 — total steps
dataset.modalities          # {'image': [...], 'state': [...], 'action': [...], 'text': [...]}
dataset.image_keys          # ['observation/finger_vision_1', 'observation/image', ...]
dataset.get_modalities()    # raw per-path specs: dtype, shape, kind, source
dataset.get_dataset_info()  # TFDS description / features / splits
dataset.data_path           # per-episode memmap dir
dataset.data_path_root      # dataset cache root
```

Example notebook: [`example/example_oxe.ipynb`](../../example/example_oxe.ipynb)

## Troubleshooting: GCS access on HPC clusters (e.g. RWTH)

`list_datasets()` and `OXEDataset` access the public GCS bucket
`gs://gresearch/robotics` via TensorFlow's GCS client, which requires valid
Google credentials and a working SSL CA bundle.  Two separate issues can arise
on HPC clusters.

### Issue 1 — no Google credentials

**Symptom:**
```
All attempts to get a Google authentication bearer token failed …
Retrieving token from files failed with "NOT_FOUND: Could not locate the credentials file."
```

**Fix:** authenticate with `gcloud` using the device flow (no browser needed on
the cluster):

```bash
gcloud auth application-default login --no-browser
```

This prints a URL.  Open it in a browser on your laptop, log in with any Google
account (a personal Gmail works — the bucket is public), and paste the
authorization code back into the cluster terminal.  Credentials are saved to
`~/.config/gcloud/application_default_credentials.json` and picked up
automatically.

---

### Issue 2 — SSL CA bundle not found

**Symptom** (appears after credentials are present):
```
libcurl code 77 … error setting certificate verify locations:
  CAfile: /etc/ssl/certs/ca-certificates.crt  CApath: none
```

The libcurl bundled with TensorFlow looks for CA certificates at a Debian path
that does not exist on RHEL/CentOS-based clusters like RWTH.

**Fix:** find the correct path and set the environment variable:

```bash
# On RWTH / RHEL-based systems the bundle is usually here:
ls /etc/pki/tls/certs/ca-bundle.crt

# If not found, use the certifi bundle (always present in a Python venv):
python3 -c "import certifi; print(certifi.where())"
```

Then export:

```bash
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
```

**Option A — terminal / interactive session (quickest):**
Add both exports to `~/.bashrc` so every new shell picks them up.

**Option B — SLURM job script (recommended for batch jobs and Jupyter):**
Add the exports to the script that launches your Jupyter server or training job,
before the `jupyter notebook` / `python` line.  The kernel inherits the
environment from the SLURM job, so this is the most reliable method.

```bash
#!/bin/bash
#SBATCH ...
export CURL_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
export REQUESTS_CA_BUNDLE=/etc/pki/tls/certs/ca-bundle.crt
jupyter notebook --no-browser --port=8888
```

**Option C — inside the notebook (always works as a fallback):**
Set the variables programmatically as the very first cell, before any import
that triggers GCS access:

```python
import os
os.environ["CURL_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"
os.environ["REQUESTS_CA_BUNDLE"] = "/etc/pki/tls/certs/ca-bundle.crt"

import robotdataset
robotdataset.list_datasets()
```

**Option D — `kernel.json` env block:**
Locate the active kernel spec with `jupyter kernelspec list` and edit its
`kernel.json` to add an `"env"` section:

```json
{
  "argv": ["python3", "-m", "ipykernel_launcher", "-f", "{connection_file}"],
  "display_name": "Python 3",
  "env": {
    "CURL_CA_BUNDLE": "/etc/pki/tls/certs/ca-bundle.crt",
    "REQUESTS_CA_BUNDLE": "/etc/pki/tls/certs/ca-bundle.crt"
  }
}
```

Restart the kernel after saving.  Note: on SLURM-launched Jupyter sessions the
job environment takes precedence, so Option B is more reliable than this one.
