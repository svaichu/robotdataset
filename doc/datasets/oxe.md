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

## Offline use and `gs://` access

The bucket is only consulted for what the local cache lacks. When a dataset's metadata
(`dataset_info.json`) is already cached, `OXEDataset(...)` does not list the bucket at all, and
`dataset.dataset_path` (the `gs://` path) is looked up on first use only; so cached episodes load on
a node without internet access. Missing episodes, or a first use, do need the bucket.

If the bucket cannot be read, the constructor raises a `RuntimeError` that carries the reason
TensorFlow gave (it is not reported as "unknown dataset"). On RHEL-family systems (for example
the RWTH cluster) the usual reason is libcurl error 77: TensorFlow's built-in `gs://` client expects
its CA bundle at `/etc/ssl/certs/ca-certificates.crt`, which does not exist there. Before the first
`gs://` access robotdataset sets `CURL_CA_BUNDLE` to the first bundle that exists (the file named by
`SSL_CERT_FILE`, `/etc/pki/tls/certs/ca-bundle.crt`, `/etc/ssl/certs/ca-bundle.crt`,
`/etc/ssl/ca-bundle.pem`, then `certifi`'s bundle), unless `CURL_CA_BUNDLE` is already set or the
Debian path exists. Set `CURL_CA_BUNDLE` yourself to override.

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
