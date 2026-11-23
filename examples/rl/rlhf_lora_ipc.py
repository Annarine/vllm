# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train a LoRA step and update a colocated vLLM policy through CUDA IPC.

Run on one CUDA GPU with enough space for the trainer and inference model:

    VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python examples/rl/rlhf_lora_ipc.py

The inference worker runs in another process. Only LoRA A/B tensors are sent;
the base model stays fixed and no adapter checkpoint is written.
"""

import argparse
import os
from functools import cached_property

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from vllm import LLM, SamplingParams
from vllm.config import WeightTransferConfig
from vllm.distributed.weight_transfer.base import (
    ParamMeta,
    WeightSource,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
)
from vllm.lora.request import LoRARequest


class AdapterSource(WeightSource):
    """Expose the selected PEFT adapter as full, canonically named tensors."""

    def __init__(self, model):
        self.model = model

    @cached_property
    def metadata(self):
        return [
            ParamMeta(name, tensor.dtype, tuple(tensor.shape))
            for name, tensor in self.iter_full()
        ]

    def iter_full(self):
        yield from sorted(get_peft_model_state_dict(self.model).items())


class OfflineClient:
    """Adapt trainer-side dictionaries to the LLM's request objects."""

    def __init__(self, llm):
        self.llm = llm

    def init_weight_transfer_engine(self, init_info):
        self.llm.init_weight_transfer_engine(WeightTransferInitRequest(init_info))

    def start_weight_update(self):
        self.llm.start_weight_update()

    def update_weights(self, update_info):
        self.llm.update_weights(WeightTransferUpdateRequest(update_info))

    def finish_weight_update(self, weight_version=None):
        self.llm.finish_weight_update(weight_version)

    def start_lora_weight_update(self, request):
        self.llm.start_lora_weight_update(request)

    def update_lora_weights(self, update_info):
        self.llm.update_lora_weights(WeightTransferUpdateRequest(update_info))

    def finish_lora_weight_update(self, update_id=None):
        self.llm.finish_lora_weight_update(update_id)

    def abort_lora_weight_update(self, update_id=None):
        self.llm.abort_lora_weight_update(update_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.35)
    args = parser.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_loras=1,
        max_cpu_loras=1,
        max_lora_rank=8,
        max_model_len=256,
        enforce_eager=True,
        enable_prefix_caching=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        weight_transfer_config=WeightTransferConfig(backend="ipc"),
    )
    torch.accelerator.set_device_index(0)
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    trainer = get_peft_model(
        AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda(),
        config,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    optimizer = torch.optim.AdamW(
        (p for p in trainer.parameters() if p.requires_grad), lr=1e-3
    )
    source = AdapterSource(trainer)
    transfer = IPCTrainerWeightTransferEngine.trainer_init(
        IPCTrainerInitInfo(rank=0, packed=False),
        client=OfflineClient(llm),
        source=source,
    )
    peft_config = config.to_dict()
    peft_config["target_modules"] = sorted(peft_config["target_modules"])
    policy = LoRARequest("policy", 1, "ipc://policy")
    prompt = "The capital of France is"

    for version in (1, 2):
        if version == 2:
            trainer.train()
            batch = tokenizer(prompt + " Paris.", return_tensors="pt").to("cuda:0")
            optimizer.zero_grad()
            loss = trainer(**batch, labels=batch["input_ids"]).loss
            loss.backward()
            optimizer.step()
            print(f"Training loss: {loss.item():.4f}")
        # No optimizer step may modify this revision until the send returns.
        torch.accelerator.synchronize()
        transfer.send_lora_weights(
            {
                "update_id": f"policy-{version}",
                "lora_name": "policy",
                "lora_int_id": 1,
                "version": version,
                "expected_version": version - 1 if version > 1 else None,
                "peft_config": peft_config,
                "names": [meta.name for meta in source.metadata],
                "dtype_names": [
                    str(meta.dtype).removeprefix("torch.") for meta in source.metadata
                ],
                "shapes": [list(meta.shape) for meta in source.metadata],
            }
        )
        result = llm.generate(
            prompt,
            SamplingParams(temperature=0, max_tokens=16),
            lora_request=policy,
        )[0]
        print(f"Policy version {version}: {result.outputs[0].text!r}")


if __name__ == "__main__":
    main()
