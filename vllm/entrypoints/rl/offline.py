# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.entrypoints.common.offline import OfflineInferenceMixin
from vllm.logger import init_logger
from vllm.lora.weight_update import LoRAWeightUpdateRequest

logger = init_logger(__name__)


class RLOfflineMixin(OfflineInferenceMixin):
    _pending_lora_weight_update: LoRAWeightUpdateRequest | None = None

    def init_weight_transfer_engine(
        self, request: WeightTransferInitRequest | dict
    ) -> None:
        """Initialize weight transfer for RL training.

        Args:
            request: Weight transfer initialization request with backend-specific info

        """
        init_info_dict = (
            request["init_info"] if isinstance(request, dict) else request.init_info
        )

        self.llm_engine.collective_rpc(
            "init_weight_transfer_engine", kwargs={"init_info": init_info_dict}
        )

    def shutdown_weight_transfer_engine(self) -> None:
        """Release the trainer connection without shutting down inference."""
        self.llm_engine.collective_rpc("shutdown_weight_transfer_engine")

    def start_weight_update(self) -> None:
        """Start a new weight update."""
        self.llm_engine.collective_rpc("start_weight_update")

    def start_draft_weight_update(self) -> None:
        """Start a new weight update targeting the speculative draft model."""
        self.llm_engine.collective_rpc("start_draft_weight_update")

    def update_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """Update the weights of the model.

        Args:
            request: Weight update request with backend-specific update info

        """
        update_info_dict = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )

        self.llm_engine.collective_rpc(
            "update_weights", kwargs={"update_info": update_info_dict}
        )

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        """Finish the weight update and set its version if provided."""
        self.llm_engine.collective_rpc("finish_weight_update")
        if weight_version is not None:
            self.llm_engine.set_weight_version(weight_version)

    def start_lora_weight_update(self, request: dict) -> None:
        """Start a LoRA update between completed generation rounds.

        If failed-start cleanup is not acknowledged, explicitly abort this
        update before retrying. An existing owner cannot be preempted.
        """
        update = LoRAWeightUpdateRequest(**request)
        if self._pending_lora_weight_update is not None:
            raise RuntimeError("A LoRA weight update is already active")
        if self.llm_engine.has_unfinished_requests():
            raise RuntimeError("Finish all generation requests before updating LoRA")
        known = self.llm_engine._lora_adapters.get(update.lora_int_id)
        if known is not None and known[0] != update.lora_name:
            raise ValueError("The LoRA ID belongs to another adapter")
        self._pending_lora_weight_update = update
        try:
            self.llm_engine.collective_rpc(
                "start_lora_weight_update", kwargs={"request": request}
            )
        except Exception:
            try:
                self.abort_lora_weight_update(update.update_id)
            except Exception:
                logger.exception(
                    "Failed to abort LoRA update after start failed; "
                    "explicitly abort update %r before retrying",
                    update.update_id,
                )
            raise

    def update_lora_weights(self, request: WeightTransferUpdateRequest | dict) -> None:
        """Receive tensors for the active LoRA update."""
        update_info = (
            request["update_info"] if isinstance(request, dict) else request.update_info
        )
        self.llm_engine.collective_rpc(
            "update_lora_weights", kwargs={"update_info": update_info}
        )

    def finish_lora_weight_update(self, update_id: str | None = None) -> None:
        """Prepare every worker before installing the new adapter version."""
        if self._pending_lora_weight_update is None:
            raise RuntimeError("Start a LoRA weight update first")
        if (
            update_id is not None
            and update_id != self._pending_lora_weight_update.update_id
        ):
            raise ValueError("update_id does not match the active LoRA update")
        update_id = self._pending_lora_weight_update.update_id
        self.llm_engine.collective_rpc(
            "prepare_lora_weight_update", kwargs={"update_id": update_id}
        )
        self.llm_engine.collective_rpc(
            "commit_lora_weight_update", kwargs={"update_id": update_id}
        )
        identity = self._pending_lora_weight_update.to_lora_request()
        known = self.llm_engine._lora_adapters.get(identity.lora_int_id)
        identity.load_inplace = known is not None and known[1] != identity.lora_path
        self.llm_engine._register_lora(identity)
        self._pending_lora_weight_update = None

    def abort_lora_weight_update(self, update_id: str | None = None) -> None:
        """Discard an uncommitted LoRA update."""
        update = self._pending_lora_weight_update
        if update is None:
            raise RuntimeError("No LoRA weight update is active")
        if update_id is not None and update_id != update.update_id:
            raise ValueError("update_id does not match the active LoRA update")
        self.llm_engine.collective_rpc(
            "abort_lora_weight_update", kwargs={"update_id": update.update_id}
        )
        self._pending_lora_weight_update = None

    def update_weight_version(self, new_version: str) -> None:
        """Set the weight version without updating weights."""
        self.llm_engine.set_weight_version(new_version)

    def get_weight_version(self) -> str:
        """Return the latest committed weight version."""
        return self.llm_engine.get_weight_version()
