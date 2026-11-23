# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Script to test add_lora, remove_lora, pin_lora, list_loras functions."""

import json
from pathlib import Path

import pytest
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.engine.arg_utils import AsyncEngineArgs, EngineArgs
from vllm.entrypoints.launchers.api_server.entry import (
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.openai.models.protocol import BaseModelPath
from vllm.entrypoints.openai.models.serving import OpenAIServingModels
from vllm.lora.request import LoRARequest
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.llm_engine import LLMEngine
from vllm.v1.executor.uniproc_executor import UniProcExecutor

from ..models.utils import check_logprobs_close

MODEL_PATH = "Qwen/Qwen3-0.6B"
LORA_MODULE_PATH = "vllm-project/self_cognition_Alice"
LORA_RANK = 8


def make_lora_request(lora_id: int):
    return LoRARequest(
        lora_name=f"{lora_id}", lora_int_id=lora_id, lora_path=LORA_MODULE_PATH
    )


def test_lora_functions_sync():
    max_loras = 4
    # Create engine in eager-mode. Due to high max_loras, the CI can
    # OOM during cuda-graph capture.
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    llm = LLMEngine.from_engine_args(engine_args)

    def run_check(fn, args, expected: list):
        fn(args)
        assert set(llm.list_loras()) == set(expected)

    run_check(llm.add_lora, make_lora_request(1), [1])
    run_check(llm.add_lora, make_lora_request(2), [1, 2])

    # Pin LoRA 1 and test that it is never removed on subsequent adds.
    run_check(llm.pin_lora, 1, [1, 2])
    run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
    run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
    run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
    run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
    run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
    run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
    run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
    run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

    # Remove LoRA 1 and continue adding.
    run_check(llm.remove_lora, 1, [8, 9, 10])
    run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
    run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

    # Remove all LoRAs.
    run_check(llm.remove_lora, 13, [12, 10, 11])
    run_check(llm.remove_lora, 12, [10, 11])
    run_check(llm.remove_lora, 11, [10])
    run_check(llm.remove_lora, 10, [])


@pytest.mark.asyncio
async def test_lora_functions_async():
    max_loras = 4
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )

    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        # The builder always yields an AsyncLLM; EngineClient only declares add_lora.
        assert isinstance(llm, AsyncLLM)

        async def run_check(fn, args, expected: list):
            await fn(args)
            assert set(await llm.list_loras()) == set(expected)

        await run_check(llm.add_lora, make_lora_request(1), [1])
        await run_check(llm.add_lora, make_lora_request(2), [1, 2])

        # Pin LoRA 1 and test that it is never removed on subsequent adds.
        await run_check(llm.pin_lora, 1, [1, 2])
        await run_check(llm.add_lora, make_lora_request(3), [1, 2, 3])
        await run_check(llm.add_lora, make_lora_request(4), [1, 2, 3, 4])
        await run_check(llm.add_lora, make_lora_request(5), [1, 5, 3, 4])
        await run_check(llm.add_lora, make_lora_request(6), [1, 5, 6, 4])
        await run_check(llm.add_lora, make_lora_request(7), [1, 5, 6, 7])
        await run_check(llm.add_lora, make_lora_request(8), [1, 8, 6, 7])
        await run_check(llm.add_lora, make_lora_request(9), [1, 8, 9, 7])
        await run_check(llm.add_lora, make_lora_request(10), [1, 8, 9, 10])

        # Remove LoRA 1 and continue adding.
        await run_check(llm.remove_lora, 1, [8, 9, 10])
        await run_check(llm.add_lora, make_lora_request(11), [8, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(12), [12, 9, 10, 11])
        await run_check(llm.add_lora, make_lora_request(13), [12, 13, 10, 11])

        # Remove all LoRAs
        await run_check(llm.remove_lora, 13, [12, 10, 11])
        await run_check(llm.remove_lora, 12, [10, 11])
        await run_check(llm.remove_lora, 11, [10])
        await run_check(llm.remove_lora, 10, [])


def test_same_lora_name_with_different_ids(
    qwen3_meowing_lora_files, qwen3_woofing_lora_files
):
    """An adapter unloaded and loaded again under the same name gets a new id
    while requests for the old id may still be running; each request in the
    shared batch must use its own adapter."""
    llm = LLM(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        gpu_memory_utilization=0.8,
        enforce_eager=True,
    )
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": "Make your favorite animal noise."},
    ]
    outputs = llm.chat(
        [messages, messages],
        SamplingParams(temperature=0, max_tokens=10),
        chat_template_kwargs={"enable_thinking": False},
        lora_request=[
            LoRARequest("pet", 1, qwen3_meowing_lora_files),
            LoRARequest("pet", 2, qwen3_woofing_lora_files),
        ],
    )
    assert "Meow Meow Meow" in outputs[0].outputs[0].text
    assert "Woof Woof Woof" in outputs[1].outputs[0].text


@pytest.mark.parametrize("enforce_eager", [True, False])
def test_dummy_batch_keeps_loras(enforce_eager):
    """Idle data parallel ranks run dummy batches between steps. Those must
    not unload adapters, including ones whose ids match the dummy LoRA ids."""
    engine_args = EngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_loras=2,
        max_cpu_loras=3,
        max_lora_rank=LORA_RANK,
        max_model_len=128,
        max_num_seqs=8,
        gpu_memory_utilization=0.8,
        enforce_eager=enforce_eager,
    )
    llm = LLMEngine.from_engine_args(engine_args)

    llm.add_lora(make_lora_request(1))
    llm.add_lora(make_lora_request(3))
    assert set(llm.list_loras()) == {1, 3}

    for _ in range(3):
        llm.engine_core.execute_dummy_batch()
    assert set(llm.list_loras()) == {1, 3}

    llm.add_request(
        "req-1",
        "Who are you?",
        SamplingParams(max_tokens=4),
        lora_request=make_lora_request(1),
    )
    while llm.has_unfinished_requests():
        llm.step()
    assert set(llm.list_loras()) == {1, 3}


class _LostLoRAStartAcknowledgementExecutor(UniProcExecutor):
    """Lose RPC acknowledgements after the real worker starts a transaction."""

    def __init__(self, *args, **kwargs):
        self.failed_start_ids: set[str] = set()
        self.abort_failures = 2
        super().__init__(*args, **kwargs)

    def collective_rpc(
        self,
        method,
        timeout=None,
        args=(),
        kwargs=None,
        non_block=False,
        single_value=False,
        **options,
    ):
        kwargs = kwargs or {}
        if (
            method == "abort_lora_weight_update"
            and kwargs.get("update_id") == "start-unacknowledged"
            and self.abort_failures
        ):
            self.abort_failures -= 1
            raise TimeoutError("Abort acknowledgement was lost")
        result = super().collective_rpc(
            method, timeout, args, kwargs, non_block, single_value, **options
        )
        if method == "start_lora_weight_update":
            update_id = kwargs["request"]["update_id"]
            if update_id in ("start-cleaned", "start-unacknowledged") and (
                update_id not in self.failed_start_ids
            ):
                self.failed_start_ids.add(update_id)
                raise ConnectionError("Start acknowledgement was lost")
        return result


def _lora_start_request(model_config, update_id):
    config = model_config.hf_text_config
    head_dim = getattr(config, "head_dim", None) or (
        config.hidden_size // config.num_attention_heads
    )
    module = "base_model.model.model.layers.0.self_attn.q_proj"
    return {
        "update_id": update_id,
        "lora_name": "policy",
        "lora_int_id": 1,
        "version": 1,
        "expected_version": None,
        "peft_config": {
            "r": LORA_RANK,
            "lora_alpha": LORA_RANK,
            "target_modules": ["q_proj"],
            "bias": "none",
        },
        "names": [f"{module}.lora_A.weight", f"{module}.lora_B.weight"],
        "dtype_names": ["float32", "float32"],
        "shapes": [
            [LORA_RANK, config.hidden_size],
            [head_dim * config.num_attention_heads, LORA_RANK],
        ],
    }


def test_ipc_lora_update_failed_offline_start_keeps_owner_until_abort_ack(vllm_runner):
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_model_len=64,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        distributed_executor_backend=_LostLoRAStartAcknowledgementExecutor,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    ) as runner:
        llm = runner.get_llm()
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"packed": False})
        )
        params = SamplingParams(temperature=0, max_tokens=4)
        before = llm.generate("A training system", params)[0].outputs[0].token_ids
        for update_id in ("start-cleaned", "start-unacknowledged"):
            request = _lora_start_request(llm.llm_engine.model_config, update_id)
            with pytest.raises(Exception, match="Start acknowledgement was lost"):
                llm.start_lora_weight_update(request)
            if update_id == "start-unacknowledged":
                with pytest.raises(RuntimeError, match="already active"):
                    llm.start_lora_weight_update({**request, "update_id": "retry"})
                with pytest.raises(ValueError, match="update_id"):
                    llm.abort_lora_weight_update("foreign")
                llm.abort_lora_weight_update(update_id)
            llm.start_lora_weight_update({**request, "update_id": "retry"})
            # Omitting the ID must still send this frontend's owned ID.
            llm.abort_lora_weight_update()
            with pytest.raises(RuntimeError, match="No LoRA weight update"):
                llm.abort_lora_weight_update()
        foreign = {**request, "update_id": "foreign"}
        llm.collective_rpc("start_lora_weight_update", kwargs={"request": foreign})
        with pytest.raises(RuntimeError, match="No LoRA weight update"):
            llm.abort_lora_weight_update()
        llm.collective_rpc(
            "abort_lora_weight_update", kwargs={"update_id": foreign["update_id"]}
        )
        after = llm.generate("A training system", params)[0].outputs[0].token_ids
        assert after == before


@pytest.mark.asyncio
async def test_ipc_lora_update_failed_serving_start_keeps_owner_until_abort_ack():
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        enable_lora=True,
        max_lora_rank=LORA_RANK,
        max_model_len=64,
        enforce_eager=True,
        gpu_memory_utilization=0.3,
        distributed_executor_backend=_LostLoRAStartAcknowledgementExecutor,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    )
    async with build_async_engine_client_from_engine_args(engine_args) as llm:
        assert isinstance(llm, AsyncLLM)
        await llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"packed": False})
        )
        models = OpenAIServingModels(
            llm, [BaseModelPath(name="base", model_path=MODEL_PATH)]
        )
        for update_id in ("start-cleaned", "start-unacknowledged"):
            request = _lora_start_request(llm.model_config, update_id)
            with pytest.raises(Exception, match="Start acknowledgement was lost"):
                await models.start_lora_weight_update(request)
            if update_id == "start-unacknowledged":
                with pytest.raises(ValueError, match="already active"):
                    await models.start_lora_weight_update(
                        {**request, "update_id": "retry"}
                    )
                with pytest.raises(ValueError, match="update_id"):
                    await models.abort_lora_weight_update("foreign")
                await models.abort_lora_weight_update()
            await models.start_lora_weight_update({**request, "update_id": "retry"})
            await models.abort_lora_weight_update()
            with pytest.raises(ValueError, match="No LoRA weight update"):
                await models.abort_lora_weight_update()
        assert [model.id for model in (await models.show_available_models()).data] == [
            "base"
        ]
        await llm.resume_generation()
        outputs = [
            output
            async for output in llm.generate(
                "A training system",
                SamplingParams(temperature=0, max_tokens=4),
                request_id="after-start-recovery",
            )
        ]
        assert len(outputs[-1].outputs[0].token_ids) == 4


def test_ipc_lora_update_preserves_source_and_invalidates_prefix_cache(
    vllm_runner,
    monkeypatch,
    qwen3_meowing_lora_files,
    qwen3_woofing_lora_files,
):
    """Real IPC updates replace a policy without mutating the trainer's weights."""
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    messages = [
        {"role": "system", "content": "Follow the instructions to make animal noises"},
        {"role": "user", "content": "Make your favorite animal noise."},
    ]
    with vllm_runner(
        MODEL_PATH,
        enable_lora=True,
        max_loras=3,
        max_cpu_loras=3,
        max_lora_rank=8,
        max_model_len=128,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=0.3,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    ) as runner:
        llm = runner.get_llm()
        llm.init_weight_transfer_engine(
            WeightTransferInitRequest(init_info={"packed": False})
        )
        base_checksums = llm.collective_rpc("compute_weight_checksums")
        policy = LoRARequest("policy", 1, "ipc://policy")

        def generate(request):
            return llm.chat(
                [messages],
                SamplingParams(temperature=0, max_tokens=10, logprobs=5),
                chat_template_kwargs={"enable_thinking": False},
                lora_request=request,
            )[0].outputs[0]

        def compare(actual, reference):
            check_logprobs_close(
                outputs_0_lst=[(actual.token_ids, actual.text, actual.logprobs)],
                outputs_1_lst=[
                    (reference.token_ids, reference.text, reference.logprobs)
                ],
                name_0="ipc",
                name_1="checkpoint",
                always_check_logprobs=True,
            )

        def update(path, version, *, incomplete=False):
            tensors = torch.load(
                Path(path) / "adapter_model.bin",
                map_location="cuda:0",
                weights_only=True,
            )
            originals = {name: tensor.clone() for name, tensor in tensors.items()}
            names = sorted(tensors)
            llm.start_lora_weight_update(
                {
                    "update_id": f"policy-{version}",
                    "lora_name": "policy",
                    "lora_int_id": 1,
                    "version": version,
                    "expected_version": version - 1 if version > 1 else None,
                    "peft_config": json.loads(
                        (Path(path) / "adapter_config.json").read_text()
                    ),
                    "names": names,
                    "dtype_names": [
                        str(tensors[name].dtype).removeprefix("torch.")
                        for name in names
                    ],
                    "shapes": [list(tensors[name].shape) for name in names],
                }
            )
            if version == 1:
                with pytest.raises(Exception, match="update_id"):
                    llm.update_lora_weights(
                        WeightTransferUpdateRequest(
                            update_info={"update_id": "stale-policy-update"}
                        )
                    )
                with pytest.raises(Exception, match="update_id|session|active"):
                    llm.abort_lora_weight_update("stale-policy-update")
            sent_names = names[:-1] if incomplete else names
            uuid = str(torch.cuda.get_device_properties(0).uuid)
            handles = [{uuid: reduce_tensor(tensors[name])[1]} for name in sent_names]
            torch.accelerator.synchronize()
            llm.update_lora_weights(
                WeightTransferUpdateRequest(
                    update_info={
                        "update_id": f"policy-{version}",
                        "names": sent_names,
                        "dtype_names": [
                            str(tensors[name].dtype).removeprefix("torch.")
                            for name in sent_names
                        ],
                        "shapes": [list(tensors[name].shape) for name in sent_names],
                        "ipc_handles": handles,
                    }
                )
            )
            if incomplete:
                with pytest.raises(Exception, match="Incomplete LoRA update"):
                    llm.finish_lora_weight_update(f"policy-{version}")
                llm.abort_lora_weight_update(f"policy-{version}")
            else:
                llm.finish_lora_weight_update(f"policy-{version}")
            for name, original in originals.items():
                torch.testing.assert_close(tensors[name], original, rtol=0, atol=0)

        meow = generate(LoRARequest("meow", 2, qwen3_meowing_lora_files))
        update(qwen3_meowing_lora_files, 1)
        compare(generate(policy), meow)
        # Repeating the prompt populates prefix blocks under the policy identity.
        compare(generate(policy), meow)

        woof = generate(LoRARequest("woof", 3, qwen3_woofing_lora_files))
        update(qwen3_woofing_lora_files, 2)
        compare(generate(policy), woof)
        assert "Meow Meow Meow" in meow.text
        assert "Woof Woof Woof" in woof.text

        update(qwen3_meowing_lora_files, 3, incomplete=True)
        compare(generate(policy), woof)
        assert llm.collective_rpc("compute_weight_checksums") == base_checksums
