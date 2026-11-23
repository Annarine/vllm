# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata and worker-owned staging for online LoRA weight updates."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from vllm.lora.peft_helper import PEFTHelper
from vllm.lora.request import LoRARequest, _TensorLoRARequest
from vllm.utils.gpu_sync_debug import gpu_sync_allowed

if TYPE_CHECKING:
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.worker_manager import WorkerLoRAManager


@dataclass
class LoRAWeightUpdateRequest:
    """A complete, PEFT-format adapter revision, without its tensor data."""

    update_id: str
    lora_name: str
    lora_int_id: int
    version: int
    expected_version: int | None
    peft_config: dict[str, Any]
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]

    def __post_init__(self) -> None:
        if not isinstance(self.update_id, str) or not self.update_id.strip():
            raise ValueError("update_id must be a nonempty string.")
        if not isinstance(self.lora_name, str) or not self.lora_name.strip():
            raise ValueError("lora_name must be a nonempty string.")
        if type(self.lora_int_id) is not int or self.lora_int_id < 1:
            raise ValueError("lora_int_id must be a positive integer.")
        if type(self.version) is not int or self.version < 1:
            raise ValueError("version must be a positive integer.")
        if self.expected_version is not None and (
            type(self.expected_version) is not int
            or self.expected_version < 1
            or self.version <= self.expected_version
        ):
            raise ValueError("version must be newer than expected_version.")
        if (
            not isinstance(self.names, list)
            or not self.names
            or any(not isinstance(name, str) or not name for name in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("LoRA tensor names must be nonempty and unique.")
        if (
            not isinstance(self.dtype_names, list)
            or not isinstance(self.shapes, list)
            or len(self.names) != len(self.dtype_names)
            or len(self.names) != len(self.shapes)
        ):
            raise ValueError("names, dtype_names and shapes must have equal lengths.")
        if not isinstance(self.peft_config, dict):
            raise ValueError("peft_config must be a configuration dictionary.")
        try:
            self.peft_config = json.loads(
                json.dumps(self.peft_config, sort_keys=True, allow_nan=False)
            )
            helper = PEFTHelper.from_dict(self.peft_config)
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError(f"Invalid LoRA configuration: {exc}") from exc
        if type(helper.r) is not int:
            raise ValueError("LoRA rank must be a positive integer.")
        targets = helper.target_modules
        if not (
            isinstance(targets, str)
            and bool(targets)
            or isinstance(targets, list)
            and bool(targets)
            and all(isinstance(target, str) and target for target in targets)
        ):
            raise ValueError("LoRA target_modules must be a nonempty string or list.")
        if any(type(rank) is not int for rank in helper.rank_pattern.values()):
            raise ValueError("LoRA rank_pattern must contain positive integers.")
        if helper.modules_to_save:
            raise ValueError("LoRA weight updates do not support modules_to_save.")
        problems = helper._validate_features()
        if helper.bias != "none":
            problems.append("LoRA weight updates do not support bias.")
        if problems:
            raise ValueError(" ".join(problems))
        modules: dict[str, dict[bool, tuple[list[int], str]]] = {}
        for name, dtype_name, shape in zip(self.names, self.dtype_names, self.shapes):
            if not isinstance(name, str) or not name.endswith(
                (".lora_A.weight", ".lora_B.weight")
            ):
                raise ValueError(f"Unsupported LoRA tensor name: {name!r}.")
            is_a = name.endswith(".lora_A.weight")
            suffix = ".lora_A.weight" if is_a else ".lora_B.weight"
            module_name = name.removeprefix("base_model.model.").removesuffix(suffix)
            if not module_name or any(not part for part in module_name.split(".")):
                raise ValueError(f"Unsupported LoRA tensor name: {name!r}.")
            if any(
                part in {"experts", "routed_experts", "shared_expert", "shared_experts"}
                for part in module_name.split(".")
            ):
                raise ValueError("Expert LoRA weight updates are not supported.")
            if dtype_name not in ("float16", "bfloat16", "float32"):
                raise ValueError(f"Unsupported LoRA tensor dtype: {dtype_name!r}.")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(d) is not int or d < 1 for d in shape)
            ):
                raise ValueError(f"LoRA tensor {name!r} must have a positive 2D shape.")
            rank, _ = helper.get_rank_and_scaling(module_name)
            if shape[0 if is_a else 1] != rank:
                raise ValueError(f"LoRA tensor {name!r} does not match rank {rank}.")
            pair = modules.setdefault(module_name, {})
            if is_a in pair:
                raise ValueError(f"Duplicate LoRA A/B tensor for {module_name!r}.")
            pair[is_a] = (shape, dtype_name)
        for module_name, pair in modules.items():
            if set(pair) != {True, False}:
                raise ValueError(f"Missing LoRA A/B pair for {module_name!r}.")
            if pair[True][1] != pair[False][1]:
                raise ValueError(f"LoRA A/B dtypes must match for {module_name!r}.")
        self.names = list(self.names)
        self.dtype_names = list(self.dtype_names)
        self.shapes = [list(shape) for shape in self.shapes]

    @property
    def schema_key(self) -> str:
        """Return the immutable configuration and ordered tensor schema."""
        return json.dumps(
            (self.peft_config, self.names, self.dtype_names, self.shapes),
            sort_keys=True,
            allow_nan=False,
        )

    def to_lora_request(self) -> LoRARequest:
        """Return the lightweight identity used by ordinary generation requests."""
        return LoRARequest(self.lora_name, self.lora_int_id, f"ipc://{self.lora_name}")


class LoRAWeightUpdateSession:
    """Stage a full adapter in private storage before replacing a live revision."""

    def __init__(
        self,
        manager: "WorkerLoRAManager",
        request: LoRAWeightUpdateRequest,
        device: torch.device,
    ) -> None:
        self.manager = manager
        self.request = request
        self.device = device
        self.state = "receiving"
        self._manifest = {
            name: (getattr(torch, dtype), tuple(shape))
            for name, dtype, shape in zip(
                request.names, request.dtype_names, request.shapes
            )
        }
        self._tensors: dict[str, torch.Tensor] = {}
        self._prepared: LoRAModel | None = None

    def consume(self, weights: list[tuple[str, torch.Tensor]]) -> None:
        """Copy received tensors before any LoRA normalization can mutate them."""
        if self.state != "receiving":
            raise RuntimeError("LoRA update is not receiving tensors.")
        with torch.no_grad(), gpu_sync_allowed():
            for name, tensor in weights:
                if name in self._tensors:
                    raise ValueError(f"Duplicate LoRA tensor: {name!r}.")
                if self._manifest.get(name) != (tensor.dtype, tuple(tensor.shape)):
                    raise ValueError(f"LoRA tensor {name!r} does not match manifest.")
                # .to() without copy=True may alias imported IPC storage. The
                # LoRA loader later scales B in place, so ownership is required.
                owned = tensor.detach().to(device=self.device, copy=True).contiguous()
                if not bool(torch.isfinite(owned).all()):
                    raise ValueError(f"LoRA tensor {name!r} contains nonfinite values.")
                self._tensors[name] = owned

    def prepare(self) -> None:
        """Validate the complete candidate without changing the installed adapter."""
        if self.state == "prepared":
            return
        if self.state != "receiving":
            raise RuntimeError("LoRA update cannot be prepared in its current state.")
        missing = set(self._manifest) - self._tensors.keys()
        if missing:
            raise ValueError(
                f"Incomplete LoRA update; missing tensors: {sorted(missing)}"
            )
        identity = self.request.to_lora_request()
        request = _TensorLoRARequest(
            identity.lora_name,
            identity.lora_int_id,
            identity.lora_path,
            _lora_tensors=self._tensors,
            _peft_config=self.request.peft_config,
        )
        try:
            self._prepared = self.manager.prepare_tensor_adapter(request)
        except Exception:
            # Normalization may have scaled some owned tensors before failing.
            # Retrying prepare could apply that scaling a second time.
            self.state = "failed"
            raise
        self.state = "prepared"

    def commit(self) -> None:
        """Install and pin a prepared revision while generation is paused."""
        if self.state == "committed":
            return
        if self.state != "prepared" or self._prepared is None:
            raise RuntimeError("LoRA update must be prepared before commit.")
        self.manager.install_prepared_adapter(self._prepared)
        self.state = "committed"
        self._tensors.clear()
        self._prepared = None

    def abort(self) -> None:
        """Release candidate storage; installed revisions are owned by the manager."""
        self._tensors.clear()
        self._prepared = None
        self.state = "aborted"
