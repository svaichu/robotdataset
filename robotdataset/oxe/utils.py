from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import torch


@dataclass(frozen=True)
class ModalitySpec:
    path: str
    kind: str
    dtype: Optional[str]
    shape: Optional[Tuple[int, ...]]
    source: str


def normalize_version_key(version: str) -> Tuple[int, ...]:
    parts = [int(chunk) for chunk in re.split(r"[^0-9]+", version) if chunk]
    return tuple(parts) if parts else (0,)


def latest_version(versions: Sequence[str]) -> str:
    return sorted(versions, key=normalize_version_key)[-1]


def tf_to_torch(value: Any, tf_tensor_types: Tuple[type, ...] = ()) -> Any:
    """Convert a nested structure of TF tensors / numpy arrays to PyTorch tensors.

    Every numeric leaf becomes a torch.Tensor (zero-copy for numpy arrays via
    torch.from_numpy).  Non-numeric leaves (strings, bytes) are decoded to str
    since they cannot be represented as tensors.
    """
    if tf_tensor_types and isinstance(value, tf_tensor_types):
        value = value.numpy()  # fall through to numpy / bytes handling

    if isinstance(value, torch.Tensor):
        return value

    if isinstance(value, np.ndarray):
        if value.dtype.kind in {"S", "U", "O"}:  # string / object → decode to str
            decoded = value.flat[0] if value.ndim == 0 else value
            if value.dtype.kind == "S":
                return value.astype(str).tolist()
            return value.tolist()
        return torch.from_numpy(np.ascontiguousarray(value))

    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.str_) or np.issubdtype(value.dtype, np.bytes_):
            return value.item().decode("utf-8") if isinstance(value.item(), bytes) else str(value.item())
        return torch.as_tensor(value.item())

    if isinstance(value, (bytes, bytearray)):
        try:
            return value.decode("utf-8")
        except Exception:
            return bytes(value)

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return torch.tensor(value)

    if isinstance(value, (int, float)):
        return torch.tensor(value)

    if isinstance(value, Mapping):
        return {k: tf_to_torch(v, tf_tensor_types) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(tf_to_torch(v, tf_tensor_types) for v in value)
    if isinstance(value, list):
        return [tf_to_torch(v, tf_tensor_types) for v in value]

    return value


def infer_kind(path: str, value: Any = None) -> str:
    lowered = path.lower()
    if any(token in lowered for token in ("image", "camera", "rgb", "video", "frame")):
        return "image"
    if any(token in lowered for token in ("language", "instruction", "text", "caption", "prompt")):
        return "text"
    if any(token in lowered for token in ("action", "policy", "torque", "velocity")):
        return "action"
    if any(token in lowered for token in ("observation", "state", "proprio", "joint", "pose", "ee")):
        return "state"
    if isinstance(value, (str, bytes, bytearray)):
        return "text"
    if isinstance(value, np.ndarray) and value.ndim >= 3:
        return "image"
    return "generic"


def shape_and_dtype(value: Any, tf_tensor_types: Tuple[type, ...]) -> Tuple[Optional[Tuple[int, ...]], Optional[str]]:
    if isinstance(value, tf_tensor_types):
        shape = tuple(int(dim) if dim is not None else -1 for dim in value.shape)
        dtype = value.dtype.name
        return shape, dtype
    if isinstance(value, torch.Tensor):
        return tuple(int(dim) for dim in value.shape), str(value.dtype)
    if isinstance(value, np.ndarray):
        return tuple(int(dim) for dim in value.shape), str(value.dtype)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            shape = tuple(int(dim) for dim in value.shape)
        except Exception:
            shape = None
        return shape, str(value.dtype)
    return None, None


def flatten_structure(
    tree: Any,
    tf_tensor_types: Tuple[type, ...],
    prefix: str = "",
) -> Dict[str, ModalitySpec]:
    flattened: Dict[str, ModalitySpec] = {}
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            child_prefix = f"{prefix}/{key}" if prefix else str(key)
            flattened.update(flatten_structure(value, tf_tensor_types, child_prefix))
        return flattened
    if isinstance(tree, (list, tuple)) and tree and not isinstance(tree[0], (bytes, bytearray, str)):
        for index, value in enumerate(tree):
            child_prefix = f"{prefix}/{index}" if prefix else str(index)
            flattened.update(flatten_structure(value, tf_tensor_types, child_prefix))
        return flattened

    shape, dtype = shape_and_dtype(tree, tf_tensor_types)
    path = prefix or "value"
    flattened[path] = ModalitySpec(
        path=path,
        kind=infer_kind(path, tree),
        dtype=dtype,
        shape=shape,
        source="metadata" if prefix else "sample",
    )
    return flattened


def dict_to_tensordict(data: Dict[str, Any]) -> "TensorDict":
    """Convert a nested dict of torch tensors to a TensorDict (batch_size=[]).

    Tensor leaves become regular TensorDict fields.  String/non-tensor leaves
    are stored via set_non_tensor so they survive round-trips without coercion.
    Nested dicts become nested TensorDicts.
    """
    from tensordict import TensorDict

    tensors: Dict[str, Any] = {}
    non_tensors: Dict[str, Any] = {}

    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            tensors[key] = val
        elif isinstance(val, dict):
            tensors[key] = dict_to_tensordict(val)
        else:
            non_tensors[key] = val

    td = TensorDict(tensors, batch_size=[])
    for key, val in non_tensors.items():
        td.set_non_tensor(key, val)
    return td


def episode_to_ted_steps(
    episode: Any,
    episode_idx: int,
    tf_tensor_types: Tuple[type, ...] = (),
) -> List["TensorDict"]:
    """Convert one TFDS episode to a list of TED-format TensorDicts (one per step).

    Handles both TFDS OXE format (episode has a "steps" key whose value is an
    iterable of step dicts) and flat format (episode dict is itself one step).

    Each output TensorDict follows TorchRL TED convention:
        observation, action, done, terminated
        next/{observation, reward, done, terminated}
        collector/episode_id  (= episode_idx)

    For the terminal step, next/observation is a copy of the current observation.
    Missing reward / is_last / is_terminal fields default to 0 / positional / positional.
    """
    # Collect raw steps
    if "steps" in episode:
        raw = list(episode["steps"])
    else:
        raw = [episode]

    # Convert all steps to Python/torch in one shot
    steps = [tf_to_torch(s, tf_tensor_types) for s in raw]
    T = len(steps)

    traj_id = torch.tensor(episode_idx, dtype=torch.int64)
    ted_steps: List[Any] = []

    for t, step in enumerate(steps):
        is_last_val = step.get("is_last", t == T - 1)
        is_terminal_val = step.get("is_terminal", is_last_val)
        # Ensure bool scalars regardless of whether the source was a tensor or bool
        is_last = torch.as_tensor(is_last_val, dtype=torch.bool).view(1)
        is_terminal = torch.as_tensor(is_terminal_val, dtype=torch.bool).view(1)

        reward_raw = step.get("reward", 0.0)
        reward = torch.as_tensor(
            reward_raw.item() if isinstance(reward_raw, torch.Tensor) else float(reward_raw),
            dtype=torch.float32,
        ).view(1)

        next_step = steps[t + 1] if t < T - 1 else step
        obs = step.get("observation", {})
        next_obs = next_step.get("observation", {})

        td = dict_to_tensordict(
            {
                "observation": obs if isinstance(obs, dict) else {"obs": obs},
                "action": step.get("action", torch.zeros(1)),
                "done": is_last,
                "terminated": is_terminal,
                "next": {
                    "observation": next_obs if isinstance(next_obs, dict) else {"obs": next_obs},
                    "reward": reward,
                    "done": is_last,
                    "terminated": is_terminal,
                },
                "collector": {"episode_id": traj_id},
            }
        )

        # Pass through any extra string/non-tensor fields (e.g. language_instruction)
        for key, val in step.items():
            if key not in {"observation", "action", "reward", "is_last", "is_terminal",
                           "is_first", "discount", "steps"}:
                if not isinstance(val, (torch.Tensor, dict)):
                    td.set_non_tensor(key, val if not isinstance(val, bytes) else val.decode("utf-8"))

        ted_steps.append(td)

    return ted_steps
