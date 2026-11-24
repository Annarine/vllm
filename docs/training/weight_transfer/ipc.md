# IPC Engine

The IPC weight transfer engine uses **CUDA IPC** (Inter-Process Communication) handles to share GPU memory directly between the trainer and inference workers on the **same GPU**. This avoids staging weight data through CPU memory when colocating training and inference. Multi-GPU setups are supported — weights are all gathered by each GPU and are extracted by the correct colocated process.

## When to Use IPC

- Training and inference share the **same GPU(s)** (colocated)

## How It Works

1. The trainer creates CUDA tensors for each weight and generates IPC handles using `torch.multiprocessing.reductions.reduce_tensor`. In multi-GPU setups (e.g. FSDP), each trainer rank materializes the full tensor for each parameter onto its own GPU before generating the handle — which `ModuleSource` does for you. Each rank serves its colocated worker, so every rank must own every parameter; a partitioned source is rejected.
2. Every trainer rank contributes its handles to an all-gather; the sender merges them so each payload maps every GPU UUID to its args, and ships the merged handles to the inference engine through the client. Each worker reads only the handle for its own GPU.
3. The inference worker reconstructs the tensors from the handles using `rebuild_cuda_tensor`, reading directly from the trainer's GPU memory.

Unlike NCCL, IPC transfer is straight-line: `update_weights` *is* the transfer,
and it rides the client, so there is no concurrent broadcast to overlap with.

!!! note
    The handle all-gather in step 2 runs over the **default** process group, so it
    assumes that group is exactly the set of colocated trainer ranks and that the
    sender is a member. It is a no-op when no distributed group exists.

!!! warning
    IPC handles involve sending serialized Python objects. When using HTTP transport, you must set `VLLM_ALLOW_INSECURE_SERIALIZATION=1` on both the server and client. This is because IPC handles are pickled and base64-encoded for HTTP transmission.

## Inference Side

```python
from vllm import LLM
from vllm.config import WeightTransferConfig

llm = LLM(model="my-model", weight_transfer_config=WeightTransferConfig(backend="ipc"))
```

```bash
vllm serve my-model --weight-transfer-config '{"backend": "ipc"}'
```

IPC needs no data-plane rendezvous, so `init_transfer_engine` opens no channel —
it only records the `packed` flag the trainer ships at the handshake, which
`receive_weights` then reads. Whether a transfer is packed is therefore never
something you configure on the inference side.

## Trainer Side

```python
from vllm.distributed.weight_transfer import (
    ModuleSource,
    HTTPVLLMWeightSyncClient,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

engine = WeightTransferTrainerFactory.trainer_init(
    init_info=IPCTrainerInitInfo(rank=0, packed=False),   # rank 0 is the sender
    client=HTTPVLLMWeightSyncClient("http://localhost:8000"),
    source=ModuleSource(model),
)

engine.send_weights()   # once per sync
```

`send_weights()` drives `start_weight_update`, the transfer itself, and
`finish_weight_update`, and holds strong references to the IPC-shared copies
until after the post-send barrier — the consumer's views would otherwise dangle.

Any [`VLLMWeightSyncClient`](base.md#vllmweightsyncclient) works here — the
built-in HTTP and Ray clients, or an adapter for your own stack. The engine is
identical either way.

### `IPCTrainerInitInfo`

| Field | Default | Description |
| ----- | ------- | ----------- |
| `rank` | — | Keyword-only. This trainer process's rank; **0 is the sender** |
| `packed` | `False` | Chunked, bounded-memory transfer (see below) |
| `packed_buffer_size_bytes` | 1 GiB | Chunk size when `packed=True` |

`packed` is a must-agree wire param: `trainer_init` ships it to the worker, which
records it and decodes accordingly. It is not a `WeightTransferConfig` field and
not a per-round `update_weights` field. `packed_buffer_size_bytes` is
producer-only — the consumer rebuilds from the IPC handle plus the per-chunk
`tensor_sizes`, so it never needs the buffer size.

## Packed (Chunked) Transfer

By default, all weights are sent in a single `update_weights` call. For large
models, this requires the full model to reside in GPU memory on both sides
simultaneously. Setting `packed=True` enables **chunked transfer** with bounded
GPU memory:

- Weights are concatenated into fixed-size packed buffers (`packed_buffer_size_bytes`).
- Each chunk is sent as a separate `update_weights` call within a single `start_weight_update` / `finish_weight_update` bracket, so the layerwise reload pass is initialized once at the start and finalized once at the end regardless of chunk count.
- After each chunk is consumed, the GPU memory for that chunk can be reclaimed.

```python
engine = WeightTransferTrainerFactory.trainer_init(
    init_info=IPCTrainerInitInfo(
        rank=0,
        packed=True,
        packed_buffer_size_bytes=256 * 1024 * 1024,  # 256 MB chunks
    ),
    client=client,
    source=ModuleSource(model),
)
```

On a multi-rank trainer the producer reuses one buffer across chunks, so packed
mode carries a per-chunk barrier across ranks: without it, a rank could overwrite
its buffer while its colocated worker is still reading the current chunk. This is
handled inside `send_weights()`.

## Trainer Without vLLM over HTTP

A colocated trainer can speak the [HTTP endpoints](README.md#api-endpoints)
directly, with only PyTorch and an HTTP client. The server needs
`VLLM_ALLOW_INSECURE_SERIALIZATION=1`; without it, init and start return 200 and
the first `/update_weights` returns 500. Reference:
[`rlhf_http_ipc_no_vllm.py`](../../../examples/rl/rlhf_http_ipc_no_vllm.py).

`POST /init_weight_transfer_engine` body: `{"init_info": {"packed": false}}`.

`POST /update_weights` body `{"update_info": {...}}`:

| Field | Meaning |
| ----- | ------- |
| `names`, `dtype_names`, `shapes` | Checkpoint names, `torch` dtype attribute names (`"bfloat16"`), full shapes |
| `ipc_handles_pickled` | `base64(pickle.dumps(handles))`. Unpacked: a list with one dict per name. Packed: one dict for the shared buffer |
| `tensor_sizes` | Packed only: byte size of each tensor in the chunk, in `names` order |

- A handle dict is `{gpu_uuid: reduce_tensor(t)[1]}` with one entry per GPU the
  server uses (`torch.multiprocessing.reductions.reduce_tensor`).
  `gpu_uuid = str(torch.cuda.get_device_properties(i).uuid)`, **without** the
  `GPU-` prefix that `nvidia-smi` prints. The worker overwrites `args[6]`
  (device index) with its own.
- `ipc_handles` (unpickled) is not JSON-serializable; over HTTP use
  `ipc_handles_pickled`.
- Packed: export one `uint8` buffer once and reuse it for every chunk. Copy
  `t.contiguous().view(-1).view(torch.uint8)` in `names` order and close a chunk
  **before** the next tensor would overflow it (unlike NCCL, which closes after).
  A tensor larger than the buffer is an error. Synchronize before each
  `/update_weights`; do not overwrite the buffer until the call returns.
- Keep every exported tensor alive until `/finish_weight_update` returns, then
  drop the references and call `torch.cuda.ipc_collect()`.
- Multi-rank trainers: each rank exports handles for its own GPU, the ranks
  gather the `{uuid: args}` dicts to rank 0 over their own `torch.distributed`
  group and merge them per index, and rank 0 sends all HTTP. In packed mode,
  barrier before and after each `/update_weights`.

## Rank-Local Updates

For different parameter subsets per worker, `update_info` may be a list indexed
by worker rank. This form is IPC-only because collective backends require every
worker to participate.

## LoRA Weight Updates for RL

`IPCTrainerWeightTransferEngine.send_lora_weights(request)` transfers a selected
adapter's complete PEFT-format LoRA A/B tensors from a trainer to vLLM. It does
not write an adapter checkpoint or update the base model. This is an opt-in
path: enable LoRA and set `weight_transfer_config` to the IPC backend. Regular
`send_weights()` continues to update base-model weights.

Start with the runnable
[`rlhf_lora_ipc.py`](../../../examples/rl/rlhf_lora_ipc.py) example. It creates
a PEFT training model and a separate vLLM inference process on one GPU, sends
an initial adapter, takes an optimizer step and sends the next revision. Both
generations use the ordinary identity
`LoRARequest("policy", 1, "ipc://policy")`; training tensors are never attached
to generation requests.

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn \
    python examples/rl/rlhf_lora_ipc.py
```

For HTTP, start a server on the same physical GPU as the trainer:

```bash
VLLM_SERVER_DEV_MODE=1 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
VLLM_ALLOW_INSECURE_SERIALIZATION=1 \
CUDA_VISIBLE_DEVICES=0 \
vllm serve Qwen/Qwen3-0.6B \
    --enable-lora --max-loras 1 --max-cpu-loras 1 --max-lora-rank 8 \
    --enforce-eager --gpu-memory-utilization 0.35 \
    --weight-transfer-config '{"backend": "ipc"}'
```

Use `HTTPVLLMWeightSyncClient` with a `WeightSource` that exposes **only the
selected adapter**, following `AdapterSource` in the example. Do not pass
`ModuleSource(peft_model)`: it also exposes base-model parameters. Every source
tensor must be a full PEFT A/B tensor; materialize any training-side shards
before exporting it.

### Request and Lifecycle

The dictionary passed to `send_lora_weights()` contains:

| Field | Meaning |
| ----- | ------- |
| `update_id` | Unique identifier for this transfer; each tensor payload carries the same ID |
| `lora_name`, `lora_int_id` | Stable adapter name and positive integer ID |
| `version` | Positive integer revision, increasing after each successful update |
| `expected_version` | Previously committed revision, or `null` for the first IPC revision |
| `peft_config` | JSON-serializable PEFT configuration, including `r`, `lora_alpha` and `target_modules` |
| `names`, `dtype_names`, `shapes` | Ordered complete manifest; dtype names are `torch` attributes such as `"bfloat16"` |

For a registered adapter, the PEFT configuration and ordered tensor schema must
stay fixed across revisions. Each module needs both A and B; missing,
duplicate, nonfinite, unknown or incompatible tensors are rejected. Supported
wire dtypes are `float16`, `bfloat16` and `float32`.

The trainer helper drives these control operations:

| Python API | HTTP endpoint and body |
| ---------- | ---------------------- |
| `start_lora_weight_update(request)` | `POST /start_lora_weight_update` with `{"request": request}` |
| `update_lora_weights(WeightTransferUpdateRequest(...))` | `POST /update_lora_weights` with `{"update_info": payload}` |
| `finish_lora_weight_update(update_id)` | `POST /finish_lora_weight_update` with `{"update_id": update_id}` |
| `abort_lora_weight_update(update_id)` | `POST /abort_lora_weight_update` with `{"update_id": update_id}` |

The update payload uses the unpacked IPC fields documented above, plus
`update_id`. HTTP uses `ipc_handles_pickled`; the existing insecure
serialization opt-in applies. `HTTPVLLMWeightSyncClient` performs that encoding.
Pass the same owner `update_id` when finishing or aborting: a stale command
must not complete or cancel another trainer's active session. The optional
no-argument forms remain available for compatibility; the trainer helper
always passes its own ID.

Online start drains in-flight requests with `pause_generation(mode="wait")`
and clears the KV/prefix caches. It rejects new generation requests until an
explicit `POST /resume` after successful finish. Offline updates require the
previous `generate()` call to have finished and reset prefix caching before
receiving. Do not update an adapter underneath a running request.

Each worker copies received IPC tensors into its **own** GPU storage before
constructing a candidate LoRA. All workers prepare before any worker commits;
successful commit replaces and pins the adapter, then publishes its identity.
Over HTTP, subsequent OpenAI-compatible requests can select `lora_name` as
their `model`.

The trainer helper validates the complete manifest on every trainer rank and
requires those ranks to agree before contacting vLLM. It explicitly clones the
selected A/B tensors into an immutable CUDA snapshot, then holds that snapshot
until the transfer is acknowledged. Keep optimizer writes stopped while the
helper captures the snapshot; the helper does not coordinate the optimizer or
promise that concurrently modified tensors belong to one training step.

When calling the raw APIs instead, the caller owns the export storage. Keep
every exported tensor alive and unchanged until finish or abort has been
acknowledged. `detach().contiguous()` does not necessarily create an
independent snapshot. Receiver-owned copies are also necessary because LoRA
normalization can scale B in place; IPC mappings must not become the inference
adapter's long-term backing storage.

If receive or prepare fails, abort the update to discard the candidate; the
previous adapter remains installed. If installation fails after commit begins,
the engine stays unavailable and must be restarted. An abort does not repair a
partially committed update. Never resume generation unconditionally in an
exception handler.

After start has been acknowledged, `send_lora_weights()` attempts abort with
its own `update_id` if a later operation fails. It does not abort after a
failed start, which could otherwise cancel a different active session. If
abort is not acknowledged, it retains the exported snapshots and rejects
subsequent sends. A control-plane timeout does not prove that a worker has
stopped opening or reading IPC handles, so those exports must not be freed or
reused on that assumption.

If workers reject a newly reserved start, the engine core aborts that owner
before restoring its previous scheduling state. A failed abort acknowledgement
keeps inference paused. A start rejected because another owner is already
active leaves that owner's transaction intact.

### Extending IPC for Device Plugins

The LoRA control plane checks the instantiated transport's
`supports_lora_weight_updates` property, rather than requiring the backend name
`"ipc"`. `WeightTransferEngine` defaults to `False`; the IPC implementation
enables it only for unpacked transfers. A registered device backend can reuse
the LoRA update APIs under its own backend name when its workers expose this
capability.

The receiver contract is `receive_tensors(update_info)`, returning named
tensors without loading them into the base model. Returned tensors may alias
the producer's storage; the common LoRA session copies them into worker-owned
storage before normalization. Implementing only `receive_weights()` does not
implement this contract, because that method applies base-model weights.

For an IPC subclass using the same payload, override device operations instead
of copying the transfer lifecycle. Base weights support both packed and
unpacked transfers; LoRA updates remain unpacked-only.

| Class | Hook | Plugin responsibility |
| ----- | ---- | --------------------- |
| `IPCWeightTransferEngine` | `_get_device_uuid()` | Return the physical device identity used by the producer's handle map |
| `IPCWeightTransferEngine` | `_rebuild_ipc_tensor(args)` | Reconstruct a tensor with the native device IPC API, remapping logical device indices as needed |
| `IPCTrainerWeightTransferEngine` | `_initialize_device()` | Set `device`, `device_index` and `gpu_uuid`; the latter is an opaque wire identity despite its historical name |
| `IPCTrainerWeightTransferEngine` | `_export_ipc_tensor(tensor)` | Return native reconstruction arguments for the tensor |
| `IPCTrainerWeightTransferEngine` | `_synchronize_device()` | Complete device work before another process reads the export |
| `IPCTrainerWeightTransferEngine` | `_collect_ipc()` | Perform native IPC cleanup if needed; the default calls the device module's `ipc_collect()` when available |

The default trainer uses `current_platform.device_type` for its device and
moves ordinary checkpoint tensors with `.to(device=self.device)`. Its reducer
is resolved when exporting each tensor, after device plugins have registered
their native reducers. The default physical-device key still comes from the
device module's properties, preserving existing CUDA IPC UUIDs; a plugin may
override the identity without changing the payload schema.

Packed transfers allocate on the trainer's device and use the same export,
identity, rebuild and synchronization hooks as unpacked transfers. Rebuilding
also remaps the receiving device through `_rebuild_ipc_tensor()`, rather than
assuming every device uses CUDA's argument layout. The shared importer opens
each export once across all chunks and releases it on finish or shutdown.
Received packed tensors own their storage, so the producer can reuse its
buffer after acknowledgement. Keep the inherited `_post_send_sync()` so its
cross-rank barrier and device synchronization run after every packed chunk.

For direct helper users, `packed_ipc_producer()` accepts `device`,
`export_tensor` and `synchronize` keyword arguments. `packed_ipc_consumer()`
accepts `device_uuid` and `rebuild_tensor`; pass one `PackedBufferImporter`
across all chunks with the same rebuild callback. Omitting the callback keeps
the existing CUDA reconstruction behavior. Trainer subclasses can set
`update_info_cls` to extend the update dataclass without duplicating payload
construction.

Call the parent trainer constructor so the shared state that retains
unacknowledged LoRA exports is initialized. Import a device-specific reducer
inside the export hook when its availability depends on plugin initialization;
do not rely on a CUDA reducer imported before the device plugin registered its
own reducer. A backend with a different wire format can override the complete
`receive_tensors()` implementation instead of the receiver hooks.

For example, an Ascend subclass would initialize `device` as an NPU device,
use its physical chip identity for the handle map, export through the native
`torch_npu` reducer and reconstruct through `rebuild_npu_tensor`. The inherited
post-send lifecycle would synchronize the NPU and call its cleanup hook.
These are extension
requirements, not an included or hardware-validated `npu_ipc` implementation.

Worker subclasses that override base-model updates, sleep, or adapter
load/remove operations must preserve the common LoRA guards. The shared
helpers are `_check_lora_update_idle()`, `_check_lora_adapter_load()` and
`_check_lora_adapter_remove()`. Keep the installed-adapter restriction for sleep
and base reloads, and the identity/version protection for adapter replacement.
The public start/receive/prepare/commit/abort lifecycle, owned staging and cache
invalidation remain shared. Device-specific LoRA layouts and tensor-parallel
loading still require validation on the target hardware.

### Initial Support

- Unpacked IPC only (`packed=False`), CUDA, and an awake engine. Process
  checkpoints are unsupported while an IPC adapter is installed.
- One IPC LoRA adapter per engine, with fixed configuration, inference PP=1,
  DP=1 and expert parallelism disabled. The adapter may target ordinary linear
  layers in either a dense or an MoE base model, such as an attention output
  projection. Each target must use a supported linear LoRA wrapper and provide
  full A/B shapes; the inference LoRA loader performs TP slicing.
- Each trainer GPU must provide the full adapter for its colocated inference
  worker. Different physical GPUs, arbitrary pre-sharded LoRA tensors, routed
  or shared expert LoRA, speculative decoding and updates while sleeping are
  unsupported. Expert tensor paths are rejected in the manifest, and the
  worker also rejects target layers outside the ordinary linear LoRA wrappers.
- Use one API server process for HTTP so the published adapter identity has a
  single owner. Reserve a LoRA slot for the pinned, memory-only adapter.
- After its first commit, the IPC adapter stays pinned for the engine's
  lifetime. Update it through these IPC APIs; disk-based replacement and
  unloading of that adapter are rejected even between update rounds.
- Base-model reloads and sleep are rejected while an IPC adapter is installed,
  including after commit. Restart the engine to return to those workflows.
- During an update, loading or removing any adapter is also rejected.

## Limitations

- **Colocated only.** Each trainer rank shares memory with the inference worker
  on its own GPU; trainer and inference on separate GPUs need another backend.
- **Pipeline-parallel trainers are not supported yet (work in progress).** Every
  trainer rank serves its colocated worker the whole model, so every rank must
  own every parameter. A source whose ranks own different parameters (pipeline
  stages, or experts held by only some ranks) is rejected at `trainer_init`.
- **A full copy per GPU.** Each rank materializes every parameter in full on its
  GPU. Unpacked, the whole model stays resident until the round finishes; set
  `packed=True` to bound it to one chunk.
- The handle all-gather runs over the default process group, which must be
  exactly the colocated trainer ranks.

## Examples

- [RLHF with IPC weight syncing (`vllm serve`, HTTP)](../../../examples/rl/rlhf_http_ipc.py) - **Start here.** Server and training model share a single GPU; HTTP control plane, CUDA IPC data plane. Launches and tears down its own server
- [RL LoRA weight updates through IPC](../../../examples/rl/rlhf_lora_ipc.py) - A PEFT optimizer step followed by direct A/B tensor transfer into an existing vLLM adapter
- [IPC weight syncing from a trainer that does not import vLLM](../../../examples/rl/rlhf_http_ipc_no_vllm.py) - `torch` + `requests` only, `ipc_handles_pickled` over HTTP; see [Trainer Without vLLM over HTTP](#trainer-without-vllm-over-http)
- [RLHF with IPC + FSDP2 and expert parallelism](../../../examples/rl/rlhf_ipc_fsdp_ep.py) - Multi-rank trainer colocated with a `--data-parallel-size 4` server on the same 4 GPUs: every FSDP rank builds an engine and joins the handle all-gather, with packed chunking and sleep/wake around the transfer
