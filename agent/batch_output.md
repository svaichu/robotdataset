# Batch Output Format & Key Management

## Output Format

All samplers (`TemporalSampler`, `EpisodeTubeletSampler`) return a **flat TensorDict** — not a nested one.

The last line of every sampler's `__call__` is:
```python
return batch.flatten_keys("/")
```

`flatten_keys("/")` is TensorDict's built-in method. It collapses nested structure into top-level string keys joined by `"/"`.

## Key Format

All keys in the output batch are `"/"` separated strings:
```
"observation/agentview_rgb"
"observation/ee_states"
"action"
"next/observation/agentview_rgb"
"collector/episode_id"
...
```

Accessed directly: `batch["observation/agentview_rgb"]`.

### Why not return a nested TensorDict

Nested TensorDicts require tuple keys for access: `batch[("observation", "agentview_rgb")]`. That's clunky and inconsistent with how modality paths are represented everywhere else (slash strings in `delta_timestamps`, `modalities`, `image_keys`).

### Why not return a plain Python dict

`flatten_keys("/")` returns a flat TensorDict, which still supports all TensorDict operations. A plain `dict` would lose that. Converting via `dict(td.flatten_keys("/").items())` is unnecessary.

### The "/" access rule in TensorDict

On a **nested** TensorDict, `td["observation/image"]` does NOT work — the slash is not interpreted as a separator. Only tuple keys work: `td[("observation", "image")]`.

On a **flat** TensorDict (after `flatten_keys("/")`), `td["observation/image"]` works because `"observation/image"` is literally a top-level key.

This is why `flatten_keys("/")` must be called before the batch leaves the sampler.

## Key Source of Truth: Storage, Not Schema

`default_dt` (the per-key delta_timestamps fed to `TemporalSampler`) is built from the **actual storage keys**:

```python
self._storage_keys = set(combined_td.flatten_keys("/").keys())
default_dt = {key: [0.0] for key in self._storage_keys}
```

Earlier it was built from `self._modalities` (inferred from the TFDS feature schema). This caused a `KeyError` when the schema listed `"observation/image"` but the actual stored data used named camera keys (`"observation/agentview_rgb"`, `"observation/eye_in_hand_rgb"`). Schema and data don't always agree in OXE datasets.

`image_keys` (used for HWC→CHW permutation) is also filtered against `_storage_keys` for the same reason.

## image_keys

`dataset.image_keys` returns `List[str]` — slash-separated strings matching the flat batch format:
```python
["observation/agentview_rgb", "observation/eye_in_hand_rgb"]
```

Internally, samplers convert these to tuples when indexing the **nested** storage (before flattening):
```python
self.image_keys = {
    tuple(k.split("/")) if isinstance(k, str) else k for k in image_keys
}
```

## Viz Tools

`batchViz` and `itemViz` accept the flat TensorDict directly. Key access is just `batch[key]` where `key` is a `"/"` string. No tuple splitting, no nested lookup.
