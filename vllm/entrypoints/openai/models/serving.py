# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from asyncio import Lock
from collections import defaultdict
from http import HTTPStatus
from typing import Any

from vllm.config import ModelConfig
from vllm.engine.protocol import EngineClient
from vllm.entrypoints.openai.models.protocol import BaseModelPath, LoRAModulePath
from vllm.entrypoints.serve import create_error_response
from vllm.entrypoints.serve.engine.protocol import (
    ErrorResponse,
    ModelCard,
    ModelList,
    ModelPermission,
)
from vllm.entrypoints.serve.lora.protocol import (
    LoadLoRAAdapterRequest,
    UnloadLoRAAdapterRequest,
)
from vllm.exceptions import LoRAAdapterNotFoundError
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.lora.weight_update import LoRAWeightUpdateRequest
from vllm.utils.counter import AtomicCounter

logger = init_logger(__name__)


class OpenAIModelRegistry:
    """Read-only view of the loaded base models with no engine dependency.

    Suitable for CPU-only / render-only contexts that have no engine client
    and no LoRA support.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        base_model_paths: list[BaseModelPath],
    ) -> None:
        self.model_config = model_config
        self.base_model_paths = base_model_paths
        self.lora_requests: dict[str, LoRARequest] = {}

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        return self.base_model_paths[0].name

    def is_base_model(self, model_name: str) -> bool:
        return any(model.name == model_name for model in self.base_model_paths)

    async def check_model(self, model_name: str | None) -> ErrorResponse | None:
        """Return an ErrorResponse if model_name is not served, else None."""
        if not model_name or self.is_base_model(model_name):
            return None
        return create_error_response(
            message=f"The model `{model_name}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
            param="model",
        )

    async def show_available_models(self) -> ModelList:
        """Show available models (base models only)."""
        max_model_len = self.model_config.max_model_len
        return ModelList(
            data=[
                ModelCard(
                    id=base_model.name,
                    max_model_len=max_model_len,
                    root=base_model.model_path,
                    permission=[ModelPermission()],
                )
                for base_model in self.base_model_paths
            ]
        )

    async def resolve_lora(self, lora_name: str):
        raise RuntimeError("The OpenAIModelRegistry has no LoRA support.")


class OpenAIServingModels:
    """Shared instance to hold data about the loaded base model(s) and adapters.

    Handles the routes:
    - /v1/models
    - /v1/load_lora_adapter
    - /v1/unload_lora_adapter
    """

    def __init__(
        self,
        engine_client: EngineClient,
        base_model_paths: list[BaseModelPath],
        *,
        lora_modules: list[LoRAModulePath] | None = None,
    ):
        super().__init__()

        self.registry = OpenAIModelRegistry(
            model_config=engine_client.model_config,
            base_model_paths=base_model_paths,
        )

        self.engine_client = engine_client
        self.base_model_paths = base_model_paths

        self.static_lora_modules = lora_modules
        self.lora_requests: dict[str, LoRARequest] = {}
        self.lora_id_counter = AtomicCounter(0)

        self.lora_resolvers: list[LoRAResolver] = []
        for lora_resolver_name in LoRAResolverRegistry.get_supported_resolvers():
            self.lora_resolvers.append(
                LoRAResolverRegistry.get_resolver(lora_resolver_name)
            )
        self.lora_resolver_lock: dict[str, Lock] = defaultdict(Lock)
        self._pending_lora_weight_update: LoRAWeightUpdateRequest | None = None
        self._last_lora_weight_update: LoRAWeightUpdateRequest | None = None
        self._lora_weight_update_names: dict[int, str] = {}

        self.model_config = self.engine_client.model_config
        self.renderer = self.engine_client.renderer
        self.input_processor = self.engine_client.input_processor

    async def init_static_loras(self):
        """Loads all static LoRA modules.
        Raises if any fail to load"""
        if self.static_lora_modules is None:
            return
        for lora in self.static_lora_modules:
            load_request = LoadLoRAAdapterRequest(
                lora_path=lora.path,
                lora_name=lora.name,
                is_3d_lora_weight=lora.is_3d_lora_weight,
            )
            load_result = await self.load_lora_adapter(
                request=load_request, base_model_name=lora.base_model_name
            )
            if isinstance(load_result, ErrorResponse):
                raise ValueError(load_result.error.message)

    def is_base_model(self, model_name: str) -> bool:
        return self.registry.is_base_model(model_name)

    def model_name(self, lora_request: LoRARequest | None = None) -> str:
        if lora_request is not None:
            return lora_request.lora_name
        return self.base_model_paths[0].name

    async def show_available_models(self) -> ModelList:
        """Show available models. This includes the base model and all
        adapters."""
        model_list = await self.registry.show_available_models()
        lora_cards = [
            ModelCard(
                id=lora.lora_name,
                root=lora.path,
                parent=lora.base_model_name
                if lora.base_model_name
                else self.base_model_paths[0].name,
                permission=[ModelPermission()],
            )
            for lora in self.lora_requests.values()
        ]
        model_list.data.extend(lora_cards)
        return model_list

    async def load_lora_adapter(
        self, request: LoadLoRAAdapterRequest, base_model_name: str | None = None
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_load_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            lora_path = request.lora_path
            lora_int_id = (
                self.lora_requests[lora_name].lora_int_id
                if lora_name in self.lora_requests
                else self.lora_id_counter.inc(1)
            )
            lora_request = LoRARequest(
                lora_name=lora_name,
                lora_int_id=lora_int_id,
                lora_path=lora_path,
                load_inplace=request.load_inplace,
                is_3d_lora_weight=request.is_3d_lora_weight,
            )
            if base_model_name is not None and self.is_base_model(base_model_name):
                lora_request.base_model_name = base_model_name

            # Validate that the adapter can be loaded into the engine
            # This will also preload it for incoming requests
            try:
                await self.engine_client.add_lora(lora_request)
            except Exception as e:
                if str(
                    LoRAAdapterNotFoundError(
                        lora_request.lora_name, lora_request.lora_path
                    )
                ) in str(e):
                    return create_error_response(
                        LoRAAdapterNotFoundError(
                            lora_request.lora_name, lora_request.lora_path
                        )
                    )
                return create_error_response(
                    message=str(e),
                    err_type="InternalServerError",
                    status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

            previous = self.lora_requests.get(lora_name)
            if previous is not None and previous.lora_path == lora_path:
                logger.warning(
                    "Reloaded LoRA adapter '%s' in place from the same path '%s'. "
                    "Prefix-cache blocks computed with its previous weights are "
                    "not invalidated; load changed weights from a new path to "
                    "avoid reusing them.",
                    lora_name,
                    lora_path,
                )
            # Reset load_inplace before caching so the flag does not leak to
            # subsequent generate requests.
            lora_request.load_inplace = False
            self.lora_requests[lora_name] = lora_request
            logger.info(
                "Loaded new LoRA adapter: name '%s', path '%s'", lora_name, lora_path
            )
            return f"Success: LoRA adapter '{lora_name}' added successfully."

    async def unload_lora_adapter(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | str:
        lora_name = request.lora_name

        # Ensure atomicity based on the lora name
        async with self.lora_resolver_lock[lora_name]:
            error_check_ret = await self._check_unload_lora_adapter_request(request)
            if error_check_ret is not None:
                return error_check_ret

            # Safe to delete now since we hold the lock
            del self.lora_requests[lora_name]
            logger.info("Removed LoRA adapter: name '%s'", lora_name)
            return f"Success: LoRA adapter '{lora_name}' removed successfully."

    async def start_lora_weight_update(self, request: dict[str, Any]) -> None:
        """Reserve an adapter identity before starting its IPC update.

        If failed-start cleanup is not acknowledged, explicitly abort this
        update before retrying. An existing owner cannot be preempted.
        """
        update = LoRAWeightUpdateRequest(**request)
        async with self.lora_resolver_lock[update.lora_name]:
            if self._pending_lora_weight_update is not None:
                raise ValueError("A LoRA weight update is already active")
            if self.is_base_model(update.lora_name):
                raise ValueError("A LoRA adapter cannot use a base model name")
            existing = self.lora_requests.get(update.lora_name)
            if existing is not None:
                if existing.lora_int_id != update.lora_int_id:
                    raise ValueError("A loaded adapter must keep its existing ID")
            elif (
                update.lora_int_id <= self.lora_id_counter.value
                and self._lora_weight_update_names.get(update.lora_int_id)
                != update.lora_name
            ):
                raise ValueError("A new adapter must use an unused LoRA ID")
            if any(
                adapter.lora_int_id == update.lora_int_id
                and adapter.lora_name != update.lora_name
                for adapter in self.lora_requests.values()
            ):
                raise ValueError("The LoRA ID belongs to another adapter")
            self.lora_id_counter.inc(
                max(0, update.lora_int_id - self.lora_id_counter.value)
            )
            self._lora_weight_update_names[update.lora_int_id] = update.lora_name
            self._pending_lora_weight_update = update
            try:
                await self.engine_client.start_lora_weight_update(request)
            except Exception:
                try:
                    await self.engine_client.abort_lora_weight_update(update.update_id)
                except Exception:
                    logger.exception(
                        "Failed to abort LoRA update after start failed; "
                        "explicitly abort update %r before retrying",
                        update.update_id,
                    )
                else:
                    self._pending_lora_weight_update = None
                raise

    async def finish_lora_weight_update(
        self, update_id: str | None = None
    ) -> LoRAWeightUpdateRequest:
        """Publish the served model name only after all workers commit."""
        update = self._pending_lora_weight_update
        if update is None:
            if self._last_lora_weight_update is not None and (
                update_id is None
                or update_id == self._last_lora_weight_update.update_id
            ):
                return self._last_lora_weight_update
            raise ValueError("Start a LoRA weight update first")
        if update_id is not None and update_id != update.update_id:
            raise ValueError("update_id does not match the active LoRA update")
        async with self.lora_resolver_lock[update.lora_name]:
            if self._pending_lora_weight_update is not update:
                if self._last_lora_weight_update is update:
                    return update
                raise ValueError(
                    "The LoRA weight update changed while awaiting its lock"
                )
            await self.engine_client.finish_lora_weight_update(update.update_id)
            self.lora_requests[update.lora_name] = update.to_lora_request()
            self._last_lora_weight_update = update
            self._pending_lora_weight_update = None
            return update

    async def abort_lora_weight_update(self, update_id: str | None = None) -> None:
        """Discard the pending version without changing the served identity."""
        update = self._pending_lora_weight_update
        if update is None:
            raise ValueError("No LoRA weight update is active")
        if update_id is not None and update_id != update.update_id:
            raise ValueError("update_id does not match the active LoRA update")
        async with self.lora_resolver_lock[update.lora_name]:
            if self._pending_lora_weight_update is not update:
                raise ValueError(
                    "The LoRA weight update changed while awaiting its lock"
                )
            await self.engine_client.abort_lora_weight_update(update.update_id)
            self._pending_lora_weight_update = None

    async def _check_load_lora_adapter_request(
        self, request: LoadLoRAAdapterRequest
    ) -> ErrorResponse | None:
        if self._pending_lora_weight_update is not None:
            return create_error_response(
                message="Finish or abort the active LoRA weight update first.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.CONFLICT,
            )
        if request.lora_name in self._lora_weight_update_names.values():
            return create_error_response(
                message="IPC LoRA adapters must be updated through IPC weight updates.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.CONFLICT,
            )
        # Check if both 'lora_name' and 'lora_path' are provided
        if not request.lora_name or not request.lora_path:
            return create_error_response(
                message="Both 'lora_name' and 'lora_path' must be provided.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        if self.is_base_model(request.lora_name):
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' conflicts with a "
                "served base model.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # If not loading inplace
        # Check if the lora adapter with the given name already exists
        if not request.load_inplace and request.lora_name in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' has already been "
                "loaded. If you want to load the adapter in place, set 'load_inplace'"
                " to True.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        return None

    async def _check_unload_lora_adapter_request(
        self, request: UnloadLoRAAdapterRequest
    ) -> ErrorResponse | None:
        if self._pending_lora_weight_update is not None:
            return create_error_response(
                message="Finish or abort the active LoRA weight update first.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.CONFLICT,
            )
        if request.lora_name in self._lora_weight_update_names.values():
            return create_error_response(
                message="IPC LoRA adapters remain pinned for the engine lifetime.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.CONFLICT,
            )
        # Check if 'lora_name' is not provided return an error
        if not request.lora_name:
            return create_error_response(
                message="'lora_name' needs to be provided to unload a LoRA adapter.",
                err_type="InvalidUserInput",
                status_code=HTTPStatus.BAD_REQUEST,
            )

        # Check if the lora adapter with the given name exists
        if request.lora_name not in self.lora_requests:
            return create_error_response(
                message=f"The lora adapter '{request.lora_name}' cannot be found.",
                err_type="NotFoundError",
                status_code=HTTPStatus.NOT_FOUND,
            )

        return None

    async def resolve_lora(self, lora_name: str) -> LoRARequest | ErrorResponse:
        """Attempt to resolve a LoRA adapter using available resolvers.

        Args:
            lora_name: Name/identifier of the LoRA adapter

        Returns:
            LoRARequest if found and loaded successfully.
            ErrorResponse (404) if no resolver finds the adapter.
            ErrorResponse (400) if adapter(s) are found but none load.

        """
        async with self.lora_resolver_lock[lora_name]:
            # First check if this LoRA is already loaded
            if lora_name in self.lora_requests:
                return self.lora_requests[lora_name]

            base_model_name = self.model_config.model
            unique_id = self.lora_id_counter.inc(1)
            found_adapter = False

            # Try to resolve using available resolvers
            for resolver in self.lora_resolvers:
                lora_request = await resolver.resolve_lora(base_model_name, lora_name)

                if lora_request is not None:
                    found_adapter = True
                    lora_request.lora_int_id = unique_id

                    try:
                        await self.engine_client.add_lora(lora_request)
                        self.lora_requests[lora_name] = lora_request
                        logger.info(
                            "Resolved and loaded LoRA adapter '%s' using %s",
                            lora_name,
                            resolver.__class__.__name__,
                        )
                        return lora_request
                    except BaseException as e:
                        logger.warning(
                            "Failed to load LoRA '%s' resolved by %s: %s. "
                            "Trying next resolver.",
                            lora_name,
                            resolver.__class__.__name__,
                            e,
                        )
                        continue

            if found_adapter:
                # An adapter was found, but all attempts to load it failed.
                return create_error_response(
                    message=(
                        f"LoRA adapter '{lora_name}' was found but could not be loaded."
                    ),
                    err_type="BadRequestError",
                    status_code=HTTPStatus.BAD_REQUEST,
                )
            else:
                # No adapter was found
                return create_error_response(
                    message=f"LoRA adapter {lora_name} does not exist",
                    err_type="NotFoundError",
                    status_code=HTTPStatus.NOT_FOUND,
                )
