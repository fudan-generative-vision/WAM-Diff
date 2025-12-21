# LINT_ME
"""Generate demo navigation predictions using NavSim-MOE model in MoE setting."""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

# Remove these if not needed in your environment.
import llava.s3_ops as st
import torch
import torch.distributed as dist

# llava / model imports
from llava.cache import dLLMCache, dLLMCacheConfig
from llava.conversation import conv_templates
from llava.hooks import register_cache_LLaDA_V
from llava.hooks.fast_dllm_hook import register_fast_dllm_hook
from llava.mm_utils import IMAGE_TOKEN_INDEX, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from transformers.hf_argparser import HfArgumentParser


# =========================
# Dataclass Args (HF-style)
# =========================
@dataclass
class DataArgs:
    """
    Data-related arguments.
    """
    navsim_root: str = field(default="None")
    steps_fut: int = 8
    steps_hist: int = 4
    camera_order: List[str] = field(
        default_factory=lambda: [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
    )


@dataclass
class ModelArgs:  # pylint: disable=R0902
    """Model Configuration Arguments."""
    pretrained_path: str
    conv_template: str = "llava_llada"
    use_fast_dllm: bool = True
    use_dllm_cache: bool = False
    prompt_interval_steps: int = 25
    gen_interval_steps: int = 7
    transfer_ratio: float = 0.25
    token_gen_steps: int = 32
    token_gen_length: int = 32
    token_block_length: int = 32


@dataclass
class RuntimeArgs:
    """Runtime / Distributed Training Arguments."""
    save_jsons_folder: str = "output/jsons"
    device_type: str = "cuda"  # cuda|npu|cpu
    backend: str = "nccl"  # nccl|gloo
    merge_after: bool = False
    debug_enable: bool = False


# =========================
# DDP helpers
# =========================
def setup_distributed(device_type: str = "npu", backend: str = "hccl"):
    """
    Initialize torch.distributed, pin local rank to the right device,
    and return (rank, world_size, local_rank, device_str).
    """
    # Figure out ranks
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Pin device
    device_type = device_type.lower()
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        torch.cuda.set_device(local_rank)
        device_str = f"cuda:{local_rank}"
    elif device_type == "npu":
        torch.npu.set_device(local_rank)
        device_str = f"npu:{local_rank}"
    else:
        device_str = "cpu"

    # Init process group
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend)

    return rank, world_size, local_rank, device_str


def get_rank_world() -> Tuple[int, int]:
    """Get the current process rank and world size for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def barrier():
    """DDP barrier."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def is_rank0() -> bool:
    """Check if current process is rank 0."""
    r, _ = get_rank_world()
    return r == 0


def get_local_rank() -> int:
    """Get local rank from env vars."""
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "RANK" in os.environ:
        num = torch.cuda.device_count() if torch.cuda.is_available() else 1
        return int(os.environ["RANK"]) % max(1, num)
    return 0


def pick_device(device_type: str, local_rank: int) -> str:
    """Pick device string based on type and local rank."""
    device_type = device_type.lower()
    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        return f"cuda:{local_rank}"
    if device_type == "npu":
        # Adjust if using a different NPU stack
        return f"npu:{local_rank}"
    if device_type == "cpu":
        return "cpu"
    raise ValueError(
        f"Unsupported --device_type '{device_type}'. Use one of: cuda|npu|cpu."
    )


# =========================
# Arg parsing (HF)
# =========================
def parse_args() -> tuple[DataArgs, ModelArgs, RuntimeArgs]:
    """
    Supports:
      - CLI flags
      - Config file (JSON/TOML/YAML): eval_nuscenes_ddp.py cfg.json
      - Mixed (file + overrides)
    """
    parser = HfArgumentParser((DataArgs, ModelArgs, RuntimeArgs))
    data_args, model_args, runtime_args = parser.parse_args_into_dataclasses(
        return_remaining_strings=False
    )
    runtime_args.device_type = runtime_args.device_type.lower()
    runtime_args.backend = runtime_args.backend.lower()
    return data_args, model_args, runtime_args


def _int_stem(p: Path) -> int:
    """Get integer from path stem."""
    try:
        return int(p.stem)
    except ValueError:  # narrow exception (fixes W0718)
        return 1_000_000_000


def merge_rank_jsons(save_jsons_folder: str, world_size: int) -> None:
    """Merge per-rank JSON files into a single JSONL file."""
    save_dir = Path(save_jsons_folder)
    merged_path = save_dir / "merged.jsonl"

    numbered: list[tuple[int, Path]] = []
    for r in range(world_size):
        rank_dir = save_dir / f"rank_{r}"
        if not rank_dir.exists():
            continue

        for p in rank_dir.glob("*.json"):
            idx = _int_stem(p)
            if idx is None:
                continue
            numbered.append((idx, p))

    numbered.sort(key=lambda t: t[0])

    count = 0
    with st.open_file(str(merged_path), "w") as out_f:
        for _, p in numbered:
            with st.open_file(str(p), "r") as in_f:
                out_f.write(in_f.read().rstrip("\n") + "\n")
            count += 1

    print(f"[Rank 0] merged {count} records -> {merged_path}")

# =========================
# Main
# =========================
def main(): # pylint: disable=R0912, R0914, R0915
    """Main function."""
    data_args, model_args, runtime_args = parse_args()
    # DDP init
    rank, world_size, _, device_str = setup_distributed(
        device_type=runtime_args.device_type,
        backend=runtime_args.backend,
    )

    # Device & IO
    device_map = device_str

    st.makedirs(runtime_args.save_jsons_folder)
    rank_out_dir = Path(runtime_args.save_jsons_folder) / f"rank_{rank}"
    st.makedirs(str(rank_out_dir))

    if is_rank0():
        print(
            f"[DDP] world_size={world_size} | saving to: {runtime_args.save_jsons_folder}"
        )
        # Save resolved config
        with st.open_file(
            str(Path(runtime_args.save_jsons_folder) / "resolved_args.json"), "w"
        ) as f:
            json.dump(
                {
                    "DataArgs": vars(data_args),
                    "ModelArgs": vars(model_args),
                    "RuntimeArgs": vars(runtime_args),
                },
                f,
                indent=2,
            )

    # Load model (per rank)
    model_name = "llava_llada"
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_args.pretrained_path,
        None,
        model_name,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    model = model.to(device_str)
    model.config.image_aspect_ratio = "anyres_max_2"
    model.eval()

    # Optional speed-ups
    if model_args.use_fast_dllm:
        register_fast_dllm_hook(model)
        if is_rank0():
            print("[Fast-dLLM] enabled")
    elif model_args.use_dllm_cache:
        dLLMCache.new_instance(
            **asdict(
                dLLMCacheConfig(
                    prompt_interval_steps=model_args.prompt_interval_steps,
                    gen_interval_steps=model_args.gen_interval_steps,
                    transfer_ratio=model_args.transfer_ratio,
                )
            )
        )
        register_cache_LLaDA_V(model, "model.layers")
        if is_rank0():
            print("[dLLM-Cache] enabled")
    else:
        if is_rank0():
            print("[Cache] disabled")

    torch.set_grad_enabled(False)

    surrounding_views = ["./assets/image.png"]

    images = [st.image_open(p) for p in surrounding_views]

    if len(images) > 1:
        model.config.image_aspect_ratio = "pad"

    image_tensor = process_images(images, image_processor, model.config)
    image_tensor = [
        _img.to(dtype=torch.float16, device=device_str) for _img in image_tensor
    ]
    image_sizes = [img.size for img in images]

    # Prompt
    conv = copy.deepcopy(conv_templates[model_args.conv_template])
    question = """Here is front views of a driving vehicle:
                <image>\nThe navigation information is: straight
                The current position is (0.00,0.00)
                Current velocity is: (8.92,-0.15) and current acceleration is: (-1.25,-0.01)
                Predict the optimal driving action for the next 4 seconds with 8 new waypoints.
                """
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device_str)
    )

    # Generate
    cont, track_x = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        steps=model_args.token_gen_steps,
        gen_length=model_args.token_gen_length,
        block_length=model_args.token_block_length,
        tokenizer=tokenizer,
        stopping_criteria=["<|eot_id|>"],
        prefix_refresh_interval=32,
        threshold=1,
    )
    text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=False)
    print(text_outputs)

    with open("output.txt", "w", encoding="utf-8") as f:
        for x_step in track_x:
            text_outputs = tokenizer.batch_decode(x_step, skip_special_tokens=False)

            # Write each decoded string on its own line
            for text in text_outputs:
                f.write(text + "\n")

            # Add separator after each step
            f.write("--------\n")

    barrier()

    # Merge (rank 0)
    if is_rank0() and runtime_args.merge_after:
        merge_rank_jsons(runtime_args.save_jsons_folder, world_size)


if __name__ == "__main__":
    main()
