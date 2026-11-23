# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""IPC-based weight transfer engine using CUDA IPC for communication."""

import pickle
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Self, cast

import pybase64 as base64
import torch
from torch.multiprocessing.reductions import rebuild_cuda_tensor, reduce_tensor

from vllm import envs
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngine,
    WeightTransferInitInfo,
    WeightTransferUpdateInfo,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.distributed.weight_transfer.base import VLLMLoRAWeightSyncClient
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    PackedBufferImporter,
    packed_ipc_consumer,
    packed_ipc_producer,
)


@dataclass
class IPCWeightTransferInitInfo(WeightTransferInitInfo):
    """Worker-side init info for IPC weight transfer. No rendezvous needed.

    `packed` is a must-agree wire param: the trainer ships it here at the init
    handshake so the worker decodes with the same setting the trainer encoded
    with. The consumer rebuilds from the IPC handle + `tensor_sizes`, so it does
    not need the buffer size (producer-only)."""

    packed: bool = False


@dataclass
class IPCTrainerInitInfo(TrainerInitInfo):
    """Trainer-side init info for IPC weight transfer. No rendezvous needed;
    `rank` (from `TrainerInitInfo`) identifies this trainer process — rank 0
    ships the merged IPC handles. All ranks still join the handle all-gather.

    `packed` / `packed_buffer_size_bytes` are the transfer's wire params. The
    trainer propagates `packed` to the worker at `trainer_init` so the two sides
    cannot disagree. `backend` is the factory dispatch key."""

    backend: ClassVar[str] = "ipc"
    packed: bool = False
    packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES


@dataclass
class IPCWeightTransferUpdateInfo(WeightTransferUpdateInfo):
    """Per-round update info for the IPC weight transfer backend."""

    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple]] | dict[str, tuple] | None = None
    """IPC handles mapping physical GPU UUID to rebuild_cuda_tensor args.
    For non-packed mode: list of per-parameter handle dicts.
    For packed mode: single handle dict for the packed buffer."""
    ipc_handles_pickled: str | None = None
    """Base64-encoded pickled IPC handles, used for HTTP transport."""
    tensor_sizes: list[int] | None = None
    """Per-parameter sizes in bytes within the packed buffer.
    Required when packed=True, unused otherwise."""

    def __post_init__(self):
        if self.ipc_handles_pickled is not None:
            if self.ipc_handles is not None:
                raise ValueError(
                    "Cannot specify both `ipc_handles` and `ipc_handles_pickled`"
                )

            if not envs.VLLM_ALLOW_INSECURE_SERIALIZATION:
                raise ValueError(
                    "Refusing to deserialize `ipc_handles_pickled` without "
                    "VLLM_ALLOW_INSECURE_SERIALIZATION=1"
                )

            self.ipc_handles = pickle.loads(base64.b64decode(self.ipc_handles_pickled))
            self.ipc_handles_pickled = None

        if self.ipc_handles is None:
            raise ValueError(
                "Either `ipc_handles` or `ipc_handles_pickled` must be provided"
            )
        num_params = len(self.names)
        if len(self.dtype_names) != num_params:
            raise ValueError(
                f"`dtype_names` should be of the same size as `names`: "
                f"got {len(self.dtype_names)} and {len(self.names)}"
            )
        if len(self.shapes) != num_params:
            raise ValueError(
                f"`shapes` should be of the same size as `names`: "
                f"got {len(self.shapes)} and {len(self.names)}"
            )
        # Unpacked transfers carry a per-parameter handle dict (a list); packed
        # transfers carry a single dict for the whole buffer, so the list check
        # only applies to the list case.
        if isinstance(self.ipc_handles, list) and len(self.ipc_handles) != num_params:
            raise ValueError(
                f"`ipc_handles` should be of the same size as `names`: "
                f"got {len(self.ipc_handles)} and {len(self.names)}"
            )


class IPCWeightTransferEngine(
    WeightTransferEngine[IPCWeightTransferInitInfo, IPCWeightTransferUpdateInfo]
):
    """Weight transfer engine using CUDA IPC between trainer and workers.

    This implementation uses CUDA IPC to transfer weights from the trainer (rank 0)
    to all inference workers in a process group. IPC handles are used to share
    memory between processes on the same node.
    """

    # Define backend-specific dataclass types
    init_info_cls = IPCWeightTransferInitInfo
    update_info_cls = IPCWeightTransferUpdateInfo

    def __init__(
        self,
        config: WeightTransferConfig,
        vllm_config: "VllmConfig",
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Set from the trainer-supplied init info at the handshake; defaults are
        # only for the (unreachable) receive-before-init case.
        self.packed = False
        # Shared across all chunks of every packed transfer this engine
        # receives; see PackedBufferImporter for the refcount contract.
        self._packed_importer = PackedBufferImporter()

    @property
    def supports_lora_weight_updates(self) -> bool:
        """Whether this engine can receive a complete adapter without reloading."""
        return not self.packed

    def _get_device_uuid(self) -> str:
        """Return the opaque physical-device key used in unpacked IPC handles."""
        return str(torch.cuda.get_device_properties(self.device.index).uuid)

    def _rebuild_ipc_tensor(self, args: tuple) -> torch.Tensor:
        """Import a CUDA tensor on this worker's assigned logical device.

        Device plugins override this hook with their own IPC reconstruction.
        The exported logical index may differ between sender and receiver.
        """
        local_args = list(args)
        # CUDA IPC's device_index is element 6 of reduce_tensor's arguments.
        local_args[6] = self.device.index
        return rebuild_cuda_tensor(*local_args)

    def init_transfer_engine(self, init_info: IPCWeightTransferInitInfo) -> None:
        """Initialize the weight transfer mechanism. No data-plane rendezvous is
        needed for IPC; this just records the trainer-supplied wire params so the
        worker decodes exactly as the trainer encoded.

        Args:
            init_info: IPC initialization info (carries `packed`).

        """
        self.packed = init_info.packed

    def start_weight_update(self) -> None:
        """Begin reloading the incoming checkpoint weights."""
        from vllm.model_executor.model_loader.reload import start_reload

        start_reload(self.model)

    def finish_weight_update(self) -> None:
        """Finish reloading after all weights have been received."""
        from vllm.model_executor.model_loader.reload import finish_reload

        finish_reload(self.model, self.model_config, self.device)
        # Every reduce_tensor call is a fresh export with its own refcount
        # slot, so releasing once per update always balances this update's
        # export and lets the trainer reclaim its staging buffer. Callers
        # that skip finish are still covered by the replace-on-next-export
        # path inside the importer.
        self._packed_importer.close()

    def receive_weights(self, update_info: IPCWeightTransferUpdateInfo) -> None:
        """Receive weights from the trainer via CUDA IPC handles and load them.

        Whether the transfer is packed is read from `self.packed`, set at the
        init handshake from the trainer's init info, so it is guaranteed to
        match how the trainer encoded.

        Args:
            update_info: IPC update info containing parameter names, dtypes, shapes,
                        and IPC handles. Each IPC handle is a mapping between physical
                        GPU UUID and the rebuild_cuda_tensor args tuple.

        """
        self.model.load_weights(self.receive_tensors(update_info))

    def receive_tensors(
        self, update_info: IPCWeightTransferUpdateInfo
    ) -> list[tuple[str, torch.Tensor]]:
        """Rebuild IPC tensors without applying them to the base model.

        Unpacked tensors alias the trainer's storage. A consumer that retains
        or transforms them must first copy them into its own storage.
        """
        if self.packed:
            if update_info.tensor_sizes is None:
                raise ValueError("`tensor_sizes` is required when packed=True")
            assert isinstance(update_info.ipc_handles, dict)
            weights = packed_ipc_consumer(
                ipc_handle=update_info.ipc_handles,
                names=update_info.names,
                shapes=update_info.shapes,
                dtype_names=update_info.dtype_names,
                tensor_sizes=update_info.tensor_sizes,
                device_index=self.device.index,
                importer=self._packed_importer,
            )
        else:
            assert isinstance(update_info.ipc_handles, list)
            weights = []
            physical_device_id = self._get_device_uuid()
            for name, ipc_handle in zip(
                update_info.names,
                update_info.ipc_handles,
            ):
                if physical_device_id not in ipc_handle:
                    raise ValueError(
                        f"IPC handle not found for device UUID "
                        f"{physical_device_id}. "
                        f"Available UUIDs: {list(ipc_handle.keys())}"
                    )

                weight = self._rebuild_ipc_tensor(ipc_handle[physical_device_id])
                weights.append((name, weight))

        return weights

    def shutdown(self) -> None:
        self._packed_importer.close()


class IPCTrainerWeightTransferEngine(TrainerWeightTransferEngine[IPCTrainerInitInfo]):
    """Trainer-side CUDA IPC weight transfer engine.

    Called on every trainer rank. For multi-rank (e.g. FSDP) trainers all ranks
    drive the source's `iter_full()` (materializing each tensor) and contribute
    to the IPC-handle all-gather, so every rank must own every parameter; only
    the sender (rank 0) ships the merged handles to the inference side. IPC
    transfer is straight-line (no concurrent broadcast like NCCL):
    `update_weights` *is* the transfer, and it rides the client, so it no-ops
    on non-senders.

    `packed` / `packed_buffer_size_bytes` come from `IPCTrainerInitInfo`; the
    sender propagates `packed` to the worker at `trainer_init`.
    """

    init_info_cls = IPCTrainerInitInfo

    def __init__(
        self,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource,
        is_sender: bool = True,
        packed: bool = False,
        packed_buffer_size_bytes: int = DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    ) -> None:
        super().__init__(client=client, source=source, is_sender=is_sender)
        self.packed = packed
        self.packed_buffer_size_bytes = packed_buffer_size_bytes
        self._initialize_device()
        self._pending_lora_exports: list[torch.Tensor] | None = None

    def _initialize_device(self) -> None:
        """Set the local CUDA device and its opaque IPC identity.

        Device plugins override this hook and call the trainer constructor so
        the shared transfer state is initialized as well.
        """
        self.device_index = torch.accelerator.current_device_index()
        self.device = torch.device("cuda", self.device_index)
        self.gpu_uuid = str(torch.cuda.get_device_properties(self.device_index).uuid)

    def _export_ipc_tensor(self, tensor: torch.Tensor) -> tuple:
        """Return opaque CUDA IPC reconstruction arguments for a tensor.

        Device plugins override this hook to use their native tensor reducer.
        """
        _, args = reduce_tensor(tensor)
        return args

    def _synchronize_device(self) -> None:
        """Wait until exported tensors are ready for another process to read."""
        torch.accelerator.synchronize()

    @classmethod
    def trainer_init(
        cls,
        init_info: IPCTrainerInitInfo,
        *,
        client: VLLMWeightSyncClient,
        source: WeightSource | None = None,
    ) -> Self:
        if source is None:
            raise ValueError("IPC trainer weight transfer requires a WeightSource.")
        if not source.implements_iter_full():
            raise TypeError(
                f"IPC sends full tensors, but {type(source).__name__} does not "
                "implement iter_full()."
            )
        # Each rank exports its own copies for its colocated worker, merged by
        # position, so every rank must hold every parameter.
        if len(source.owned_names(init_info.rank)) != len(source.metadata):
            raise ValueError(
                f"rank {init_info.rank} does not own every parameter, but IPC "
                "needs every trainer rank to own every parameter."
            )
        engine = cls(
            client=client,
            source=source,
            is_sender=init_info.is_sender,
            packed=init_info.packed,
            packed_buffer_size_bytes=init_info.packed_buffer_size_bytes,
        )
        # IPC needs no data-plane rendezvous. The sender ships the must-agree
        # `packed` flag so the worker decodes exactly as this trainer encodes.
        if engine.is_sender:
            engine.client.init_weight_transfer_engine({"packed": init_info.packed})
        return engine

    def send_weights(self) -> None:
        assert self.source is not None
        source = self.source
        if self.is_sender:
            self.client.start_weight_update()
        # Unpacked returns strong refs to its IPC-shared copies; they must stay
        # alive across `finish_weight_update` too.
        weight_refs = self._send(source)
        if self.is_sender:
            self.client.finish_weight_update()
        self._post_send_sync()
        del weight_refs

    def send_lora_weights(self, request: dict) -> None:
        """Send a complete, frozen adapter through the LoRA update APIs.

        Call on every colocated trainer rank, with identical metadata. The
        caller must keep the source unchanged while its snapshot is exported.
        Packed transfers are not supported for LoRA updates yet.
        """
        from vllm.lora.weight_update import LoRAWeightUpdateRequest

        if self.packed:
            raise ValueError("LoRA IPC updates require packed=False")
        if self._pending_lora_exports is not None:
            raise RuntimeError("A previous IPC transfer was not acknowledged")
        assert self.source is not None
        spec = None
        error = None
        try:
            spec = LoRAWeightUpdateRequest(**request)
            metadata = self.source.metadata
            if (
                [m.name for m in metadata] != spec.names
                or [str(m.dtype).removeprefix("torch.") for m in metadata]
                != spec.dtype_names
                or [list(m.shape) for m in metadata] != spec.shapes
            ):
                raise ValueError("LoRA source metadata differs from the manifest")
        except Exception as exc:
            error = str(exc)
        identity = (
            repr(
                (
                    spec.update_id,
                    spec.lora_name,
                    spec.lora_int_id,
                    spec.version,
                    spec.expected_version,
                    spec.schema_key,
                )
            )
            if spec is not None
            else None
        )
        self._check_lora_export_status(error, identity)
        assert spec is not None
        weight_refs: list[torch.Tensor] = []
        handles: list[dict[str, tuple]] = []
        error = None
        try:
            for index, (name, tensor) in enumerate(self.source.iter_full()):
                if index >= len(spec.names) or (
                    name != spec.names[index]
                    or list(tensor.shape) != spec.shapes[index]
                    or str(tensor.dtype).removeprefix("torch.")
                    != spec.dtype_names[index]
                    or tensor.device != self.device
                ):
                    raise ValueError("LoRA source tensors differ from the manifest")
                weight = tensor.detach().clone(memory_format=torch.contiguous_format)
                if not torch.isfinite(weight).all().item():
                    raise ValueError(f"LoRA tensor {name!r} contains non-finite values")
                weight_refs.append(weight)
                args = self._export_ipc_tensor(weight)
                handles.append({self.gpu_uuid: args})
            if len(weight_refs) != len(spec.names):
                raise ValueError("LoRA source did not provide every declared tensor")
            self._synchronize_device()
        except Exception as exc:
            error = str(exc)
        self._check_lora_export_status(error, identity)
        handles = self._all_gather_and_merge_handles(handles)
        client = cast("VLLMLoRAWeightSyncClient", self.client)
        self._pending_lora_exports = weight_refs
        outcome: list[tuple[str | None, bool]] = [(None, True)]
        try:
            if self.is_sender:
                started = False
                try:
                    client.start_lora_weight_update(request)
                    started = True
                    client.update_lora_weights(
                        {
                            **asdict(
                                IPCWeightTransferUpdateInfo(
                                    names=spec.names,
                                    dtype_names=spec.dtype_names,
                                    shapes=spec.shapes,
                                    ipc_handles=handles,
                                )
                            ),
                            "update_id": spec.update_id,
                        }
                    )
                    client.finish_lora_weight_update(spec.update_id)
                except Exception as exc:
                    outcome[0] = (str(exc), not started)
                    if started:
                        try:
                            client.abort_lora_weight_update(spec.update_id)
                        except Exception:
                            pass
                        else:
                            outcome[0] = (str(exc), True)
            if torch.distributed.is_initialized():
                torch.distributed.broadcast_object_list(outcome, src=0)
            error, release_exports = outcome[0]
            self._post_send_sync()
            if release_exports:
                self._pending_lora_exports = None
            if error is not None:
                raise RuntimeError(f"IPC LoRA update failed: {error}")
        finally:
            # Unacknowledged exports stay owned by this engine: a timeout does
            # not prove the worker has finished opening or reading the handles.
            del weight_refs

    @staticmethod
    def _check_lora_export_status(error: str | None, identity: str | None) -> None:
        statuses = [(error, identity)]
        if torch.distributed.is_initialized():
            gathered: list[tuple[str | None, str | None] | None] = [
                None
            ] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, (error, identity))
            statuses = [status for status in gathered if status is not None]
        errors = [status[0] for status in statuses if status[0] is not None]
        if errors:
            raise ValueError(f"LoRA export validation failed: {errors}")
        if len({status[1] for status in statuses}) != 1:
            raise ValueError("All trainer ranks must use the same LoRA update manifest")

    # ---- data plane (runs on all ranks; only the sender ships) ----

    def _send(self, source: WeightSource) -> list[torch.Tensor] | None:
        if self.packed:
            self._send_packed(source)
            return None
        return self._send_unpacked(source)

    def _all_gather_and_merge_handles(
        self,
        handles: list[dict[str, tuple]],
    ) -> list[dict[str, tuple]]:
        """All-gather and merge IPC handle dicts across ranks in one call.

        Each rank contributes a list of {gpu_uuid: ipc_args} dicts (one
        per parameter or one per chunk). A single all_gather_object
        collects every rank's full list, then the sender merges per-index so
        each dict maps every GPU UUID to its args.

        The all-gather runs over the *default* process group; this assumes the
        default group is exactly the set of colocated trainer ranks and that the
        sender is a member. No-op (returns handles unchanged) when no distributed
        group exists.
        """
        if (
            not torch.distributed.is_initialized()
            or torch.distributed.get_world_size() == 1
        ):
            return handles

        world_size = torch.distributed.get_world_size()
        gathered: list[list[dict[str, tuple]] | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, handles)
        torch.distributed.barrier()
        self._synchronize_device()

        if self.is_sender:
            merged: list[dict[str, tuple]] = []
            for param_idx in range(len(handles)):
                m: dict[str, tuple] = {}
                for rank_handles in gathered:
                    if rank_handles is not None:
                        m.update(rank_handles[param_idx])
                merged.append(m)
            return merged
        return [{} for _ in handles]

    @staticmethod
    def _post_send_sync() -> None:
        """Barrier + ipc_collect after a send; no-op if single-GPU."""
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            torch.distributed.barrier()
        torch.cuda.ipc_collect()

    def _send_unpacked(self, source: WeightSource) -> list[torch.Tensor]:
        """Iterate the source, build one IPC handle per param, all-gather the
        handles across ranks, and (sender) ship them in one update call.

        Returns the strong refs to every contiguous copy. reduce_tensor's args
        do NOT keep storage alive, and non-contiguous inputs allocate fresh
        storage in .contiguous(); the caller must keep these alive until the
        post-send barrier (past `finish`) so the consumer's IPC views stay valid.
        """
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        ipc_handles: list[dict[str, tuple]] = []
        weight_refs: list[torch.Tensor] = []

        for name, tensor in source.iter_full():
            # FSDP shards were gathered by the source; ensure contiguity here.
            names.append(name)
            dtype_names.append(str(tensor.dtype).split(".")[-1])
            shapes.append(list(tensor.shape))

            weight = tensor.detach().contiguous()
            weight_refs.append(weight)
            ipc_args = self._export_ipc_tensor(weight)
            ipc_handles.append({self.gpu_uuid: ipc_args})

        ipc_handles = self._all_gather_and_merge_handles(ipc_handles)
        self._do_send(
            names=names,
            dtype_names=dtype_names,
            shapes=shapes,
            ipc_handles=ipc_handles,
        )
        return weight_refs

    def _send_packed(self, source: WeightSource) -> None:
        """Send weights in bounded-memory chunks (packed mode)."""
        for chunk in packed_ipc_producer(
            iterator=source.iter_full(),
            gpu_uuid=self.gpu_uuid,
            post_iter_func=lambda item: item[1],
            buffer_size_bytes=self.packed_buffer_size_bytes,
        ):
            ipc_handle = self._all_gather_and_merge_handles([chunk.ipc_handle])[0]
            self._do_send(
                names=chunk.names,
                dtype_names=chunk.dtype_names,
                shapes=chunk.shapes,
                ipc_handles=ipc_handle,
                tensor_sizes=chunk.tensor_sizes,
            )
            # Per-chunk barrier: the producer reuses a single IPC buffer across
            # chunks, but only the sender waits for the consumers (via _do_send).
            # Without syncing every rank here, non-sender ranks race ahead and
            # overwrite their buffer while their colocated worker is still
            # reading the current chunk, silently corrupting the transfer.
            self._post_send_sync()

    def _do_send(
        self,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict[str, tuple]] | dict[str, tuple],
        tensor_sizes: list[int] | None = None,
    ) -> None:
        """Build one update payload and ship it via the client. Only the sender
        ships (non-sender ranks already contributed to the handle all-gather).

        Emits raw `ipc_handles`; transports that cannot carry them natively
        (HTTP/JSON) pickle them in their client (see `HTTPVLLMWeightSyncClient`).
        """
        if not self.is_sender:
            return
        update_fields: dict[str, Any] = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        }
        if tensor_sizes is not None:
            update_fields["tensor_sizes"] = tensor_sizes

        update_info = IPCWeightTransferUpdateInfo(**update_fields)
        self.client.update_weights(asdict(update_info))
