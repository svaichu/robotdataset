# Architectural Decisions

Running log of non-obvious decisions made during implementation. Read this before changing shared infrastructure.

---

## Flat keys are the project standard

All sampler `__call__` methods must return a **flat TensorDict with `/`-separated string keys**.

- `batch["observation/image"]` — correct
- `batch["observation"]["image"]` or `batch["observation", "image"]` — wrong

Both `TemporalSampler` and `EpisodeTubeletSampler` end with `return batch.flatten_keys("/")`. Tests use flat key access throughout. This applies to any future sampler or `_sample()` override.

---

## default_dt must exclude three categories of keys

When any Dataset class builds `default_dt` (the fallback `delta_timestamps` mapping), three filters are required. Missing any one causes downstream failures.

**1. Non-tensor leaves** — exclude anything that is not a `torch.Tensor`.

`language_instruction` and similar fields are stored as `NonTensorStack` (a list-like wrapper for strings). Indexing a `NonTensorStack` with a 2D `(B, T)` tensor raises:
```
TypeError: only integer tensors of a single element can be converted to an index
```

**2. `next/` and `collector/` prefixes** — exclude these entirely.

`TemporalSampler` auto-creates `("next",) + key_tuple` for every key in `delta_timestamps`. If `next/observation/image` is included in `default_dt`, the output will contain `next/next/observation/image`. `collector/` keys are episode metadata, not data modalities.

**3. `done` and `terminated`** — exclude these scalar boolean flags.

They are single-step episode boundary signals. Including them gives them shape `(B, T, 1)` — a spurious temporal dimension on a non-temporal field.

**Canonical pattern (`oxe_dataset.py`):**
```python
_EXCLUDED_PREFIXES = ("next/", "collector/")
_EXCLUDED_KEYS = {"done", "terminated"}
default_dt = {
    key: [0.0]
    for key in _storage_keys
    if isinstance(_flat_combined.get(key), torch.Tensor)
    and not any(key.startswith(p) for p in _EXCLUDED_PREFIXES)
    and key not in _EXCLUDED_KEYS
}
```

`table30v2_dataset.py` builds from `self.modalities` (dict from `infer_modalities_from_storage`) so the tensor check is `spec.get("dtype") is not None and spec.get("kind") != "text"`, but the prefix/key exclusions are identical.

---

## CI: one job per Dataset class

`.github/workflows/ci.yml` has one job per Dataset class, each with isolated extras and individually toggle-able via `workflow_dispatch` boolean inputs.

| Job | Test file | Extras |
|---|---|---|
| `test-agibot` | `test_agibot_dataset.py` | none (fully mocked) |
| `test-hf` | `test_table30v2_dataset.py` | `.[hf]` |
| `test-oxe` | `test_oxe_dataset.py` | `.[oxe]` |
| `test-oxe-jax` | `test_oxe_jax_dataset.py` | `.[oxe]` + jax |

The `if:` condition differs by whether the job's default is `true` or `false`:

- **Default `true`** (e.g. `test-oxe`): `if: ${{ github.event_name != 'workflow_dispatch' || inputs.run_oxe }}` — runs on every push/PR, and on manual dispatch when enabled.
- **Default `false`** (e.g. `test-agibot`, `test-hf`, `test-oxe-jax`): `if: ${{ github.event_name == 'workflow_dispatch' && inputs.run_X }}` — explicitly requires a manual dispatch AND the input to be true. On push/PR the first clause is false so the job is skipped.

The pattern `github.event_name != 'workflow_dispatch' || inputs.X` does NOT respect `default: false` on push/PR — the first clause short-circuits to `true`. Use AND with `== 'workflow_dispatch'` to make the intent explicit and correct.

When adding a new Dataset class: add a matching job, install only its required extras, add a `workflow_dispatch` boolean input, and pick the right `if:` pattern based on whether it should run by default on push/PR.

---

## unflatten_keys guard after load_memmap

After `TensorDict.load_memmap(...)`, always call:
```python
combined_td = combined_td.unflatten_keys("/")
```

Some versions of `tensordict` return flat-keyed TensorDicts from `load_memmap` instead of nested ones. This call is a no-op when the result is already nested, and restores nested structure when it is flat. Without it, storage access like `storage_td["observation", "image"]` raises `KeyError` in CI.

---

## test_get_cache_dir_default needs monkeypatch

The `ROBOTDATASET_CACHE` env var may be set in the execution environment (e.g. `/home/zeus/.cache/` in Lightning AI Studios). The default-path test must clear it:

```python
def test_get_cache_dir_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ROBOTDATASET_CACHE", raising=False)
    path = oxe._get_cache_dir()
    assert str(path).endswith(".cache/robotdataset")
```
