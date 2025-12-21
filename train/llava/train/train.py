# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import ast
import copy
import logging
import os
import pathlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import tokenizers
import torch
import torch.distributed as dist
import transformers
from llava import conversation as conversation_lib
from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
)
from llava.mm_utils import (
    process_anyres_image,
    process_highres_image,
    process_highres_image_crop_split,
    tokenizer_image_token,
)
from llava.model import *
from llava.train.llava_trainer import LLaVATrainer
from llava.utils import (
    rank0_print,
    read_frames_gif,
    read_frames_pyav,
)
from packaging import version
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from transformers import AutoConfig

import train.llava.s3_ops as st

torch.multiprocessing.set_sharing_strategy("file_system")

ImageFile.LOAD_TRUNCATED_IMAGES = True
local_rank = None

IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse(
    "0.14"
)

import warnings

warnings.filterwarnings("ignore")


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_class_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Used to init model class, format is XXXXForCausalLM. e.g. currently XXXX is chosen from LlavaLlama, LlavaMixtral, LlavaMistral, Llama"
        },
    )

    mm_tunable_parts: Optional[str] = field(
        default=None,
        metadata={
            "help": 'Could be "mm_mlp_adapter", "mm_vision_resampler", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_vision_tower,mm_mlp_adapter,mm_language_model", "mm_mlp_adapter,mm_language_model"'
        },
    )
    # deciding which part of the multimodal model to tune, will overwrite other previous settings

    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_mm_vision_resampler: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    vision_tower_pretrained: Optional[str] = field(
        default=None
    )  # default to the last layer

    num_token: Optional[bool] = field(default=False)

    unfreeze_mm_vision_tower: bool = field(default=False)
    unfreeze_language_model: bool = field(default=False)
    mm_vision_select_layer: Optional[int] = field(
        default=-1
    )  # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default="linear")
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default="flat")
    mm_vision_select_feature: Optional[str] = field(default="patch")
    mm_resampler_type: Optional[str] = field(default=None)
    mm_mask_drop_mode: str = field(default="fixed")
    mm_mask_drop_skip_percentage: float = field(default=0.0)
    mm_mask_drop_ratio: float = field(default=0.25)
    mm_mask_drop_ratio_upper: Optional[float] = field(default=None)
    mm_mask_drop_ratio_lower: Optional[float] = field(default=None)
    mm_spatial_pool_stride: Optional[int] = field(default=None)
    mm_spatial_pool_mode: str = field(default="bilinear")
    mm_spatial_pool_out_channels: Optional[int] = field(default=None)
    mm_perceiver_depth: Optional[int] = field(default=3)
    mm_perceiver_latents: Optional[int] = field(default=32)
    mm_perceiver_ff_mult: Optional[float] = field(default=4)
    mm_perceiver_pretrained: Optional[str] = field(default=None)
    mm_qformer_depth: Optional[int] = field(default=3)
    mm_qformer_latents: Optional[int] = field(default=32)
    mm_qformer_pretrained: Optional[str] = field(default=None)

    rope_scaling_factor: Optional[float] = field(default=None)
    rope_scaling_type: Optional[str] = field(default=None)

    s2: Optional[bool] = field(default=False)
    s2_scales: Optional[str] = field(default="336,672,1008")

    use_pos_skipping: Optional[bool] = field(default=False)
    pos_skipping_range: Optional[int] = field(default=4096)

    mm_newline_position: Optional[str] = field(default="grid")
    delay_load: Optional[bool] = field(default=True)
    add_faster_video: Optional[bool] = field(default=False)
    faster_token_stride: Optional[int] = field(default=10)

    num_experts_per_tok: Optional[int] = field(default=-1)
    num_experts: Optional[int] = field(default=-1)
    moe_choice: Optional[str] = field(default="expert")
    moe_router_score_function: Optional[str] = field(default=None)
    moe_lora_rank: Optional[int] = field(default=32)
    moe_lora_alpha: Optional[int] = field(default=32)
    moe_lora_in_features: Optional[int] = field(default=4096)
    moe_lora_out_features: Optional[int] = field(default=4096)
    moe_lora_dropout: Optional[float] = field(default=0.0)
    capacity_factor: Optional[float] = field(default=0.1)


@dataclass
class DataArguments:
    data_path: str = field(
        default=None,
        metadata={
            "help": "Path to the training data, in llava's instruction.json format. Supporting multiple json files via /path/to/{a,b,c}.json"
        },
    )

    lazy_preprocess: bool = False
    is_multimodal: bool = False
    early_mix_text: bool = False
    use_webdataset: bool = False
    total_samples: Optional[int] = field(default=None)

    image_folder: Optional[str] = field(default=None)
    image_folder_2: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the second image folder, used as a fallback."},
    )
    image_aspect_ratio: str = "square"
    image_grid_pinpoints: Optional[str] = field(default=None)
    image_crop_resolution: Optional[int] = field(default=None)
    image_split_resolution: Optional[int] = field(default=None)

    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    frames_upbound: Optional[int] = field(default=0)
    add_time_instruction: Optional[bool] = field(default=False)
    force_sample: Optional[bool] = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_mm_vision_resampler: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=4096,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    mm_vision_tower_lr: Optional[float] = None
    group_by_varlen: bool = field(default=False)
    group_by_modality_length: bool = field(default=False)
    group_by_modality_length_auto: bool = field(default=False)
    auto_find_batch_size: bool = field(default=False)
    gradient_checkpointing: bool = field(default=True)
    verbose_logging: bool = field(default=False)
    attn_implementation: str = field(
        default="flash_attention_2",
        metadata={"help": "Use transformers attention implementation."},
    )
    use_conversation_mask: bool = field(default=True)
    split_batches: bool = field(default=False)
    torch_empty_cache_steps: int = field(default=1)
    fp16: bool = field(default=False)
    debug_mode: bool = field(default=False)
    obs_upload_path: str = field(
        default="", metadata={"help": "Path to the obs upload path."}
    )
    output_dir: str = field(
        default="exp", metadata={"help": "Path to the output directory."}
    )
    save_only_model: bool = True


# @dataclass
# class EvaluationArguments:
#     eval_num_processes: int = field(default=1)
#     task_names: str = field(default=None)
#     model: str = field(default="llava")
#     model_args: Optional[str] = field(default=None)
#     num_fewshot: Optional[int] = field(default=None)
#     batch_size: int = field(default=1)
#     device: Optional[str] = field(default=None)
#     limit: Optional[int] = field(default=None)
#     check_integrity: Optional[bool] = field(default=False)
#     show_task_to_terminal: Optional[bool] = field(default=False)
#     log_samples: Optional[bool] = field(default=True)
#     gen_kwargs: Optional[str] = field(default="")
#     log_samples_suffix: Optional[str] = field(default="")
#     output_path: Optional[str] = field(default="./logs/")


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(
                    f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}"
                )
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {
        k: t
        for k, t in named_params
        if any(key_match in k for key_match in keys_to_match)
    }
    to_return = {
        k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()
    }
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ["mm_projector", "vision_tower", "vision_resampler"]
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    if (
        hasattr(trainer.args, "tune_mm_mlp_adapter")
        and trainer.args.tune_mm_mlp_adapter
    ):
        check_only_save_mm_adapter_tunnable = True
    # only has mm_mlp_adapter and mm_vision_resampler in the tuneable parts
    elif hasattr(trainer.args, "mm_tunable_parts") and (
        len(trainer.args.mm_tunable_parts.split(",")) == 1
        and (
            "mm_mlp_adapter" in trainer.args.mm_tunable_parts
            or "mm_vision_resampler" in trainer.args.mm_tunable_parts
        )
    ):
        check_only_save_mm_adapter_tunnable = True
    else:
        check_only_save_mm_adapter_tunnable = False

    trainer.accelerator.wait_for_everyone()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.synchronize()
    # torch.cuda.synchronize()

    rank0_print(f"Only save projectors: {check_only_save_mm_adapter_tunnable}")
    if check_only_save_mm_adapter_tunnable:
        # Only save Adapter
        keys_to_match = ["mm_projector", "vision_resampler"]
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(["embed_tokens", "embed_in"])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(
            trainer.model.named_parameters(), keys_to_match
        )
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split("/")[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith("checkpoint-"):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(
                    weight_to_save,
                    os.path.join(mm_projector_folder, f"{current_folder}.bin"),
                )
            else:
                torch.save(
                    weight_to_save, os.path.join(output_dir, f"mm_projector.bin")
                )
        return

    if trainer.deepspeed:
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(
    strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx + 2 : cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = "unknown"
        sentence["value"] = (
            BEGIN_SIGNAL + from_str + ": " + sentence["value"] + END_SIGNAL
        )
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(sources: Sequence[str], data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            # TODO maybe this should be changed for interleaved data?
            # if DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
            # only check for num_im=1
            num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
            if (
                num_im == 1
                and DEFAULT_IMAGE_TOKEN in sentence["value"]
                and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN)
            ):
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

            # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
            sentence["value"] = sentence["value"].replace(
                "QA_GT_caption_based_noisy", ""
            )

    return sources


def preprocess_llada(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    max_len=2048,
    system_message: str = "You are a helpful language and vision assistant. You are able to understand the visual content that the user provides, and assist the user with a variety of tasks using natural language.",
) -> Dict:
    # roles = {"human": "<|start_header_id|>user<|end_header_id|>", "gpt": "<|start_header_id|>assistant<|end_header_id|>"}
    roles = {"human": "user", "gpt": "assistant", "system": "system"}

    # Add image tokens to tokenizer as a special tokens
    # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
    # tokenizer = copy.deepcopy(tokenizer)
    # # When there is actually an image, we add the image tokens as a special token
    # if has_image:
    #     tokenizer.add_tokens(["<image>"], special_tokens=True)
    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    bos_token_id = tokenizer.convert_tokens_to_ids("<|startoftext|>")
    start_header_id = tokenizer.convert_tokens_to_ids("<|start_header_id|>")
    end_header_id = tokenizer.convert_tokens_to_ids("<|end_header_id|>")
    eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")

    unmask_tokens = [
        "<|startoftext|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
        "\n\n",
    ]
    unmask_tokens_idx = [tokenizer.convert_tokens_to_ids(tok) for tok in unmask_tokens]
    # Reset LLaDA chat templates so that it won't include assistant message every time we apply
    chat_template = "{% for message in messages %}{{'<|startoftext|>' + '<|start_header_id|>' + message['role'] + '<|end_header_id|>' + '\n\n' + message['content'] + '<|eot_id|>'}}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    # After update, calling tokenizer of llama3 will
    # auto add bos id for the tokens. ヽ(｀⌒´)ﾉ
    def safe_tokenizer_llama3(text):
        input_ids = tokenizer(text).input_ids
        if input_ids[0] == bos_token_id:
            input_ids = input_ids[1:]
        return input_ids

    nl_tokens = tokenizer.convert_tokens_to_ids("\n\n")
    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] == roles["system"]:
            try:
                system_message = source[0]["content"]
            except:
                system_message = source[0]["value"]
            source = source[1:]

        input_id, target = [], []

        # New version, use apply chat template
        # Build system message for each sentence
        input_id += tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}]
        )
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            # Make sure llava data can load
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)

            if "video" in content:
                content = content.replace("<video>", "<image>")

            conv = [{"role": role, "content": content}]
            # First is bos token we don't need here
            encode_id = tokenizer.apply_chat_template(conv)[1:]
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target += encode_id

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        for idx, encode_id in enumerate(input_id):
            if encode_id in unmask_tokens_idx:
                target[idx] = encode_id
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
        input_ids.append(input_id)
        targets.append(target)
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,  # tensor(bs x seq_len)
        labels=targets,  # tensor(bs x seq_len)
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]["value"]
        source[0]["value"] = DEFAULT_IMAGE_TOKEN
        conversation = (
            source[0]["value"]
            + source[1]["value"]
            + conversation_lib.default_conversation.sep
        )
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [
        tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
        for prompt in conversations
    ]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]["value"], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "llava_llada":
        return preprocess_llada(sources, tokenizer, has_image=has_image)

    else:
        raise TypeError(
            f"Unknown conversation version: {conversation_lib.default_conversation.version}"
        )


def join_root_to_media_paths(data, root_path):
    keys = ("image", "video")
    for item in data:
        for key in keys:
            if key in item:
                value = item[key]
                if isinstance(value, str):
                    # Single path
                    if value.startswith("./"):
                        value = value[2:]
                    item[key] = st.join(root_path, value)
                elif isinstance(value, list):
                    # List of paths
                    item[key] = [
                        st.join(root_path, v[2:])
                        if v.startswith("./")
                        else st.join(root_path, v)
                        for v in value
                    ]
    return data


def load_jsons_from_dir(dir_path):
    all_data = []
    for single_file in st.listdir(dir_path):
        if single_file.endswith(("json", "jsonl")):
            all_data.extend(st.json_load(st.join(dir_path, single_file)))
    return all_data


def load_all_datasets(config_path):
    if st.isdir(config_path):
        return load_jsons_from_dir(config_path)

    config = st.json_load(config_path)
    if isinstance(config, List):
        return config

    all_data = []

    for name, info in config.items():
        data_path = info["annotation"]
        repeat_time = info.get("repeat_time", 1)
        root_path = info.get("root", None)

        rank0_print(f"Loading {name} from {data_path}")
        if not st.exists(data_path):
            rank0_print(f"⚠️  Skipping {name}, path not found: {data_path}")
            continue

        if st.isdir(data_path):
            cur_data = load_jsons_from_dir(data_path)
            print(f"Loaded {len(cur_data)} samples from {name}")
        else:
            cur_data = st.json_load(data_path)
            print(f"Loaded {len(cur_data)} samples from {name}")
        if root_path:
            rank0_print(f"Append root to image or video path")
            join_root_to_media_paths(cur_data, root_path)

        # Repeat or sample according to repeat_time
        if repeat_time > 1:
            # repeat N times
            repeated_data = cur_data * int(repeat_time)
            # if fractional part remains, sample proportionally
            fractional = repeat_time - int(repeat_time)
            if fractional > 0:
                extra = random.sample(cur_data, int(len(cur_data) * fractional))
                repeated_data.extend(extra)
            cur_data = repeated_data

        elif 0 < repeat_time < 1:
            # sample a fraction of data
            sample_size = int(len(cur_data) * repeat_time)
            cur_data = random.sample(cur_data, sample_size)

        elif repeat_time == 0:
            rank0_print(f"Skipping {name} because repeat_time=0")
            continue

        all_data.extend(cur_data)
        rank0_print(f"→ Added {len(cur_data)} samples (after repeat/sample)")

    rank0_print(f"\n✅ Total combined samples: {len(all_data)}")
    return all_data


def replace_last_images(sample: dict, old_tag="<image>", new_tag="image"):
    imgs = sample.get(new_tag, [])
    if not isinstance(imgs, list):
        imgs = [imgs] if imgs else []
    remaining = len(imgs)

    for msg in sample.get("conversations", []):
        s = msg.get("value", "")
        k = s.count(old_tag)
        if k == 0:
            continue

        if remaining <= 0:
            # All further tags are excess: replace all
            msg["value"] = s.replace(old_tag, new_tag)
            rank0_print("warning mismatch image num with <image> tag")
            continue

        if k <= remaining:
            # Still within quota: keep all, just reduce remaining
            remaining -= k
            continue

        rank0_print("warning mismatch image num with <image> tag")
        # This message crosses the quota: keep the first `remaining`, replace the rest
        parts = s.split(
            old_tag, remaining
        )  # split only at the first `remaining` occurrences
        head = (
            old_tag.join(
                parts[:remaining]
            )  # reconstruct the kept part (with separators)
            + (old_tag if remaining > 0 else "")
        )  # include the `remaining`-th tag itself
        tail = parts[remaining].replace(old_tag, new_tag)
        msg["value"] = head + tail
        remaining = 0


class LazySupervisedDataset(Dataset):
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer = None,
        data_args: DataArguments = None,
        data_list=List,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.tokenizer_with_image = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        self.tokenizer_with_image.add_tokens(["<image>"], special_tokens=True)
        self.list_data_dict = data_list

        self.tokenizer = tokenizer
        self.data_args = data_args

        self.min_num_frame = 4
        self.max_num_frame = 8

    @classmethod
    def load_data_from_json(self, data_args, data_path=None):
        list_data_dict = []
        if data_path:
            if "," in data_path:
                # process multi json files
                data_args.dataset_paths = data_path.split(",")
            else:
                data_args.dataset_paths = [data_path]
            rank0_print(f"Loading {data_args.dataset_paths}")
            for data_path in data_args.dataset_paths:
                cur_data_dict = load_all_datasets(data_path)
                rank0_print(f"Loaded {len(cur_data_dict)} samples from {data_path}")
                list_data_dict.extend(cur_data_dict)

        else:
            raise ValueError("No data path provided")

        rank0_print(
            f"Loaded {len(list_data_dict)} samples from {data_args.dataset_paths}"
        )
        rank0_print("Formatting inputs...Skip in lazy mode")

        return list_data_dict

    @classmethod
    def sanity_ckecking(self, list_data_dict: dict):
        rank0_print("sanity checking")
        from tqdm import tqdm

        for sample in tqdm(list_data_dict):
            # Concatenate all message contents
            if "image" in sample:
                replace_last_images(sample, "<image>", "image")
                image_dict = sample["image"]
                if not isinstance(image_dict, List):
                    image_dict = [image_dict]
                all_text = " ".join(
                    msg.get("value", "") for msg in sample["conversations"]
                )
                num_tags = all_text.count("<image>")
                assert len(image_dict) == num_tags, (
                    f"{all_text},{image_dict},{num_tags}"
                )
            elif "video" in sample:
                replace_last_images(sample, "<video>", "video")
                image_dict = sample["video"]
                if not isinstance(image_dict, List):
                    image_dict = [image_dict]
                all_text = " ".join(
                    msg.get("value", "") for msg in sample["conversations"]
                )
                num_tags = all_text.count("<video>")
                assert len(image_dict) == num_tags, (
                    f"{all_text},{image_dict},{num_tags}"
                )
            else:
                replace_last_images(sample, "<image>", "image")
                replace_last_images(sample, "<video>", "video")
                all_text = " ".join(
                    msg.get("value", "") for msg in sample["conversations"]
                )
                assert "<image>" not in all_text and "<video>" not in all_text, (
                    f"{all_text}"
                )
        return list_data_dict

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            image_tokens = 0
            if "image" in sample:
                if isinstance(sample["image"], str):
                    image_tokens = 2185
                elif isinstance(sample["image"], List):
                    if len(sample["image"]) == 1:
                        image_tokens = 2185
                    else:
                        image_tokens = 730 * 6
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            assert cur_len > 0, f"Conversation length is 0 for {sample}"
            if "image" in sample or "video" in sample or self.data_args.early_mix_text:
                length_list.append(cur_len + image_tokens)
            else:
                length_list.append(-cur_len)
        return length_list

    def process_image(self, image_file, overwrite_image_aspect_ratio=None):
        image_folder = getattr(self.data_args, "image_folder", None)
        image_folder_2 = getattr(self.data_args, "image_folder_2", None)
        processor = self.data_args.image_processor

        image_path = None
        if image_folder:
            primary_path = st.join(image_folder, image_file)
            if os.path.exists(primary_path):
                image_path = primary_path
            elif image_folder_2:
                secondary_path = os.path.join(image_folder_2, image_file)
                if os.path.exists(secondary_path):
                    image_path = secondary_path
            else:
                image_path = primary_path

        elif image_folder_2:  # Only secondary folder is provided
            image_path = os.path.join(image_folder_2, image_file)

        else:
            image_path = image_file

        try:
            # image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
            image = st.image_open(image_path).convert("RGB")
        except Exception as exn:
            print(
                f"Failed to open image {image_path} (derived from {image_file}). Exception:",
                exn,
            )
            raise exn

        image_size = image.size
        image_aspect_ratio = self.data_args.image_aspect_ratio
        if overwrite_image_aspect_ratio is not None:
            image_aspect_ratio = overwrite_image_aspect_ratio
        if image_aspect_ratio == "highres":
            image = process_highres_image(
                image,
                self.data_args.image_processor,
                self.data_args.image_grid_pinpoints,
            )
        elif image_aspect_ratio == "anyres" or "anyres_max" in image_aspect_ratio:
            image = process_anyres_image(
                image,
                self.data_args.image_processor,
                self.data_args.image_grid_pinpoints,
            )
        elif image_aspect_ratio == "crop_split":
            image = process_highres_image_crop_split(image, self.data_args)
        elif image_aspect_ratio == "pad":

            def expand2square(pil_img, background_color):
                width, height = pil_img.size
                if width == height:
                    return pil_img
                elif width > height:
                    result = Image.new(pil_img.mode, (width, width), background_color)
                    result.paste(pil_img, (0, (width - height) // 2))
                    return result
                else:
                    result = Image.new(pil_img.mode, (height, height), background_color)
                    result.paste(pil_img, ((height - width) // 2, 0))
                    return result

            image = expand2square(
                image, tuple(int(x * 255) for x in processor.image_mean)
            )
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        else:
            image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
        return image, image_size, "image"

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # TODO: define number of retries somewhere else
        num_base_retries = 3
        num_final_retries = 300

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sample = self._get_item(i)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                print(self.list_data_dict[i])
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                # sample_idx = random.choice(range(len(self)))
                sample = self._get_item(next_index)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sample = self._get_item(i)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if "image" in sources[0]:
            image_file = self.list_data_dict[i]["image"]
            if type(image_file) is list:
                image = [self.process_image(f) for f in image_file]
                # Handling multi images
                # overwrite to process with simple pad
                if len(image_file) > 1:
                    image = [self.process_image(f, "pad") for f in image_file]
                    image = [[im[0], im[1], "image"] for im in image]
            else:
                image = [self.process_image(image_file)]

            all_text = " ".join(
                msg.get("value", "") for msg in sources[0]["conversations"]
            )
            num_tags = all_text.count("<image>")
            assert len(image) == num_tags, f"{all_text},{image}"
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]), self.data_args
            )

        elif "video" in sources[0]:
            video_file = self.list_data_dict[i]["video"]

            if not st.exists(video_file):
                print("File {} not exist!".format(video_file))

            try:
                if video_file.endswith("gif"):
                    video = read_frames_gif(
                        video_file,
                        num_frames=self.max_num_frame,
                        min_num_frames=self.min_num_frame,
                    )
                else:
                    video = read_frames_pyav(
                        video_file,
                        num_frames=self.max_num_frame,
                        min_num_frames=self.min_num_frame,
                        clip=self.list_data_dict[i].get("clip", None),
                    )

                processor = self.data_args.image_processor
                image = processor.preprocess(video, return_tensors="pt")["pixel_values"]

                image = [(image, video[0].size, "video")]
                sources = preprocess_multimodal(
                    copy.deepcopy([e["conversations"] for e in sources]), self.data_args
                )
                # print(sources)
            except Exception as e:
                print(f"Error: {e}")
                print(f"Failed to read video file: {video_file}")
                return self._get_item(i + 1)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])

        has_image = ("image" in self.list_data_dict[i]) or (
            "video" in self.list_data_dict[i]
        )
        data_dict = preprocess(sources, self.tokenizer_with_image, has_image=has_image)

        if "prompt" in data_dict:
            prompt = data_dict["prompt"]
        else:
            prompt = None

        if isinstance(i, int):
            data_dict = dict(
                input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0]
            )

        # image exist in the data
        if "image" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif "video" in self.list_data_dict[i]:
            data_dict["image"] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict["image"] = [
                (
                    torch.zeros(1, 3, crop_size["height"], crop_size["width"]),
                    (crop_size["width"], crop_size["height"]),
                    "text",
                ),
            ]
        # prompt exist in the data
        if prompt is not None:
            data_dict["prompt"] = prompt

        data_dict["id"] = self.list_data_dict[i].get("id", i)
        if conversation_lib.default_conversation.version == "llada_plain":
            data_dict["is_plain"] = True
            data_dict["is_llada"] = True
        elif conversation_lib.default_conversation.version == "llava_llada":
            data_dict["is_plain"] = False
            data_dict["is_llada"] = True

        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    training_args: Optional[TrainingArguments] = (
        None  # Add training_args as a class attribute
    )

    def pad_sequence(self, input_ids, batch_first, padding_value):
        if self.tokenizer.padding_side == "left":
            input_ids = [torch.flip(_input_ids, [0]) for _input_ids in input_ids]
        # input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=batch_first, padding_value=padding_value)
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            input_ids = torch.flip(input_ids, [1])
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple(
            [instance[key] for instance in instances] for key in ("input_ids", "labels")
        )
        # input_ids, labels, ids = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "id"))
        input_ids = [
            _input_ids[: self.tokenizer.model_max_length] for _input_ids in input_ids
        ]
        labels = [_labels[: self.tokenizer.model_max_length] for _labels in labels]
        if self.tokenizer.pad_token_id is None:
            # self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  # FIXME: this could only be triggered for llama3 model.
            self.tokenizer.pad_token_id = (
                0  # This gets the best result. Don't know why.
            )

        if "is_llada" in instances[0] and instances[0]["is_llada"]:
            # Pad the sequence with the pad token id
            input_ids = self.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = self.pad_sequence(
                labels, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )

            batch = dict(
                input_ids=input_ids,
                labels=labels.long() if labels.dtype == torch.int32 else labels,
            )

            # Only add attention_mask for multi-round dialogs with conversation_mask enabled
            if (
                "is_plain" in instances[0]
                and not instances[0]["is_plain"]
                and self.training_args.use_conversation_mask
            ):
                # This is for multi-round dialogs
                assert len(input_ids) == 1 and len(labels) == 1, (
                    "Now, the batch size must be 1 for multi-round dialogs in LLaDA"
                )
                batch["attention_mask"] = torch.ones_like(
                    input_ids, device=input_ids.device
                )
        else:
            input_ids = self.pad_sequence(
                input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            labels = self.pad_sequence(
                labels, batch_first=True, padding_value=IGNORE_INDEX
            )
            batch = dict(
                input_ids=input_ids,
                labels=labels.long() if labels.dtype == torch.int32 else labels,
                attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            )
            # batch = dict(input_ids=input_ids, labels=labels, attention_mask=input_ids.ne(self.tokenizer.pad_token_id), ids=ids)

        if "image" in instances[0]:
            images = [instance["image"] for instance in instances]

            batch["image_sizes"] = [im[1] for im_list in images for im in im_list]
            batch["modalities"] = [im[2] for im_list in images for im in im_list]
            images = [im[0] for im_list in images for im in im_list]

            # if all(x is not None and x.shape == images[0].shape for x in images):
            # Image: (N, P, C, H, W)
            # Video: (N, F, C, H, W)
            #     batch["images"] = torch.stack(images)
            # else:
            batch["images"] = images

        if "prompt" in instances[0]:
            batch["prompts"] = [instance["prompt"] for instance in instances]

        return batch


def ddp_broadcast_object(obj, src: int = 0):
    if not dist.is_available() or not dist.is_initialized():
        return obj
    rank = dist.get_rank()
    obj_list = [obj if rank == src else None]
    dist.broadcast_object_list(obj_list, src=src)
    return obj_list[0]


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, training_args
) -> Dict:
    """Create dataset and collator for supervised fine-tuning."""
    # Check if using WebDataset format
    rank0_print("Loading data...")

    if dist.get_rank() == 0:
        data_list = LazySupervisedDataset.load_data_from_json(
            data_args=data_args, data_path=data_args.data_path
        )
        data_list = LazySupervisedDataset.sanity_ckecking(data_list)
    else:
        data_list = None

    dist.barrier()

    rank0_print("broadcasting data_list...")
    data_list = ddp_broadcast_object(data_list, src=0)
    dist.barrier()
    print(data_list[0])

    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer, data_args=data_args, data_list=data_list
    )

    data_collator = DataCollatorForSupervisedDataset(
        tokenizer=tokenizer, training_args=training_args
    )
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def get_model(model_args, training_args, bnb_model_from_pretrained_args):
    assert training_args.attn_implementation
    if training_args.attn_implementation == "sdpa" and torch.__version__ < "2.1.2":
        raise ValueError(
            "The 'sdpa' attention implementation requires torch version 2.1.2 or higher."
        )

    customized_kwargs = dict()
    customized_kwargs.update(bnb_model_from_pretrained_args)
    cfg_pretrained = None

    overwrite_config = {}
    if any(
        [
            model_args.rope_scaling_factor is not None,
            model_args.rope_scaling_type is not None,
            model_args.mm_spatial_pool_stride is not None,
            model_args.mm_spatial_pool_out_channels is not None,
            model_args.mm_spatial_pool_mode is not None,
            model_args.mm_resampler_type is not None,
            model_args.num_experts_per_tok is not None,
            model_args.num_experts is not None,
            model_args.moe_choice is not None,
            model_args.moe_lora_rank is not None,
            model_args.moe_lora_alpha is not None,
            model_args.moe_lora_in_features is not None,
            model_args.moe_lora_out_features is not None,
            model_args.moe_lora_dropout is not None,
        ]
    ):
        cfg_pretrained = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )

    if model_args.num_experts_per_tok is not None:
        overwrite_config["num_experts_per_tok"] = model_args.num_experts_per_tok
    if model_args.num_experts is not None:
        overwrite_config["num_experts"] = model_args.num_experts
    if model_args.moe_choice is not None:
        overwrite_config["moe_choice"] = model_args.moe_choice
    if model_args.capacity_factor is not None:
        overwrite_config["capacity_factor"] = model_args.capacity_factor
    if model_args.moe_lora_rank is not None:
        overwrite_config["moe_lora_rank"] = model_args.moe_lora_rank
    if model_args.moe_lora_alpha is not None:
        overwrite_config["moe_lora_alpha"] = model_args.moe_lora_alpha
    if model_args.moe_lora_in_features is not None:
        overwrite_config["moe_lora_in_features"] = model_args.moe_lora_in_features
    if model_args.moe_lora_out_features is not None:
        overwrite_config["moe_lora_out_features"] = model_args.moe_lora_out_features
    if model_args.moe_lora_dropout is not None:
        overwrite_config["moe_lora_dropout"] = model_args.moe_lora_dropout
    if (
        model_args.use_pos_skipping is not None
        and model_args.pos_skipping_range is not None
    ):
        overwrite_config["use_pos_skipping"] = model_args.use_pos_skipping
        overwrite_config["pos_skipping_range"] = model_args.pos_skipping_range

    if (
        model_args.rope_scaling_factor is not None
        and model_args.rope_scaling_type is not None
    ):
        overwrite_config["rope_scaling"] = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }
        if training_args.model_max_length is None:
            training_args.model_max_length = (
                cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
            )
            overwrite_config["max_sequence_length"] = training_args.model_max_length
        assert training_args.model_max_length == int(
            cfg_pretrained.max_position_embeddings * model_args.rope_scaling_factor
        ), print(
            f"model_max_length: {training_args.model_max_length}, max_position_embeddings: {cfg_pretrained.max_position_embeddings}, rope_scaling_factor: {model_args.rope_scaling_factor}"
        )
        # overwrite_config["max_sequence_length"] = model_args.max_sequence_length
        # overwrite_config["tokenizer_model_max_length"] = model_args.tokenizer_model_max_length

    if (
        model_args.mm_spatial_pool_stride is not None
        and model_args.mm_spatial_pool_out_channels is not None
        and model_args.mm_spatial_pool_mode is not None
        and model_args.mm_resampler_type is not None
    ):
        overwrite_config["mm_resampler_type"] = model_args.mm_resampler_type
        overwrite_config["mm_spatial_pool_stride"] = model_args.mm_spatial_pool_stride
        overwrite_config["mm_spatial_pool_out_channels"] = (
            model_args.mm_spatial_pool_out_channels
        )
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if model_args.mm_spatial_pool_mode is not None:
        overwrite_config["mm_spatial_pool_mode"] = model_args.mm_spatial_pool_mode

    if overwrite_config:
        assert cfg_pretrained is not None, "cfg_pretrained is None"

        rank0_print(f"Overwriting config with {overwrite_config}")
        for k, v in overwrite_config.items():
            setattr(cfg_pretrained, k, v)

        customized_kwargs["config"] = cfg_pretrained

    model = LlavaLLaDAModelLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=training_args.attn_implementation,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
        low_cpu_mem_usage=False,
        **customized_kwargs,
    )

    return model


@torch.no_grad()
def init_numeric_and_special_tokens(
    model,
    tokenizer,
    numeric_tokens,
    noise_scale: float = 0.01,
):
    """
    Initialize embeddings for:
      1) numeric tokens like "123.45"

    Args:
        model: HF model (e.g., AutoModelForCausalLM)
        tokenizer: HF tokenizer (same one used to add tokens)
        numeric_tokens: list[str] of numeric tokens (e.g., ["-100.00", ..., "100.00"])
        noise_scale: stddev of small Gaussian noise added to each initialized vector
    """
    emb = model.get_input_embeddings().weight
    device = emb.device
    dim = emb.shape[1]

    def tok_ids_for_text(text: str):
        # Get subword ids (no specials), filter out -100/None if any
        ids = tokenizer.encode(text, add_special_tokens=False)
        return [i for i in ids if isinstance(i, int) and i >= 0]

    def avg_embedding_for_text(text: str):
        ids = tok_ids_for_text(text)
        if not ids:
            return None
        vec = emb[torch.tensor(ids, device=device)].mean(dim=0)
        return vec

    # ---- Build a numeric "base" vector from digits and dot ----
    digit_ids = []
    for d in "0123456789":
        _ids = tok_ids_for_text(d)
        digit_ids.extend(_ids)
    dot_ids = tok_ids_for_text(".")

    base_chunks = []
    if digit_ids:
        base_chunks.append(emb[torch.tensor(digit_ids, device=device)].mean(dim=0))
    if dot_ids:
        base_chunks.append(emb[torch.tensor(dot_ids, device=device)].mean(dim=0))
    if base_chunks:
        numeric_base = torch.stack(base_chunks, dim=0).mean(dim=0)
    else:
        # fallback if tokenizer lacks digits/dot as standalone pieces
        numeric_base = torch.zeros(dim, device=device)

    # ---- Initialize numeric tokens ----
    for t in numeric_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        if tid is None or tid < 0:
            continue
        noise = noise_scale * torch.randn(dim, device=device)
        emb[tid] = numeric_base + noise


def train(attn_implementation=None):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if training_args.verbose_logging:
        rank0_print(f"Inspecting experiment hyperparameters:\n")
        rank0_print(f"model_args = {vars(model_args)}\n\n")
        rank0_print(f"data_args = {vars(data_args)}\n\n")
        rank0_print(f"training_args = {vars(training_args)}\n\n")
        # rank0_print(f"evaluation_args = {vars(evaluation_args)}\n\n")

    local_rank = training_args.local_rank

    if dist.get_rank() == 0 and training_args.debug_mode:
        import debugpy

        debugpy.listen(56780)
        print("Waiting for debugger attach")
        debugpy.wait_for_client()
        debugpy.breakpoint()
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    bnb_model_from_pretrained_args = {}

    model = get_model(model_args, training_args, bnb_model_from_pretrained_args)
    model.config.use_cache = False
    if (
        model_args.rope_scaling_factor is not None
        and model_args.rope_scaling_type is not None
    ):
        model.config.rope_scaling = {
            "factor": model_args.rope_scaling_factor,
            "type": model_args.rope_scaling_type,
        }

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if len(tokenizer) < 130000 and model_args.num_token:
        num_tokens = [f"{x:.2f}" for x in np.linspace(-100, 100, 20001)]
        tokenizer.add_tokens(num_tokens)

        model.resize_token_embeddings(len(tokenizer))

        init_numeric_and_special_tokens(
            model=model,
            tokenizer=tokenizer,
            numeric_tokens=num_tokens,
            noise_scale=0.01,
        )
    rank0_print(f"num_tokens_length: {len(tokenizer)}")

    rank0_print(f"Prompt version: {model_args.version}")
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                model_args.version
            ]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates[
                "vicuna_v1"
            ]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args, fsdp=training_args.fsdp
        )

        vision_tower = model.get_vision_tower()
        vision_tower.to(
            dtype=torch.bfloat16 if training_args.bf16 else torch.float16,
            device=training_args.device,
        )

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        if data_args.image_grid_pinpoints is not None:
            if (
                isinstance(data_args.image_grid_pinpoints, str)
                and "x" in data_args.image_grid_pinpoints
            ):
                try:
                    patch_size = data_args.image_processor.size[0]
                except Exception as e:
                    patch_size = data_args.image_processor.size["shortest_edge"]

                assert patch_size in [224, 336, 384, 448, 512], (
                    "patch_size should be in [224, 336, 384, 448, 512]"
                )
                # Use regex to extract the range from the input string
                matches = re.findall(r"\((\d+)x(\d+)\)", data_args.image_grid_pinpoints)
                range_start = tuple(map(int, matches[0]))
                range_end = tuple(map(int, matches[-1]))
                # Generate a matrix of tuples from (range_start[0], range_start[1]) to (range_end[0], range_end[1])
                grid_pinpoints = [
                    (i, j)
                    for i in range(range_start[0], range_end[0] + 1)
                    for j in range(range_start[1], range_end[1] + 1)
                ]
                # Multiply all elements by patch_size
                data_args.image_grid_pinpoints = [
                    [dim * patch_size for dim in pair] for pair in grid_pinpoints
                ]
            elif isinstance(data_args.image_grid_pinpoints, str):
                data_args.image_grid_pinpoints = ast.literal_eval(
                    data_args.image_grid_pinpoints
                )

        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints
        model.config.image_crop_resolution = data_args.image_crop_resolution
        model.config.image_split_resolution = data_args.image_split_resolution
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length
        model.config.mm_newline_position = model_args.mm_newline_position
        model.config.add_faster_video = model_args.add_faster_video
        model.config.faster_token_stride = model_args.faster_token_stride
        model.config.add_time_instruction = data_args.add_time_instruction
        model.config.force_sample = data_args.force_sample
        model.config.mm_spatial_pool_stride = model_args.mm_spatial_pool_stride

        ### Deciding train which part of the model
        if (
            model_args.mm_tunable_parts is None
        ):  # traditional way of deciding which part to train
            model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = (
                model_args.tune_mm_mlp_adapter
            )
            model.config.tune_mm_vision_resampler = (
                training_args.tune_mm_vision_resampler
            ) = model_args.tune_mm_vision_resampler
            if model_args.tune_mm_mlp_adapter or model_args.tune_mm_vision_resampler:
                model.requires_grad_(False)
            if model_args.tune_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if model_args.tune_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True

            model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
            if training_args.freeze_mm_mlp_adapter:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False

            model.config.freeze_mm_vision_resampler = (
                training_args.freeze_mm_vision_resampler
            )
            if training_args.freeze_mm_vision_resampler:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = False

            model.config.unfreeze_mm_vision_tower = model_args.unfreeze_mm_vision_tower
            if model_args.unfreeze_mm_vision_tower:
                vision_tower.requires_grad_(True)
            else:
                vision_tower.requires_grad_(False)

        else:
            rank0_print(f"Using mm_tunable_parts: {model_args.mm_tunable_parts}")
            model.config.mm_tunable_parts = training_args.mm_tunable_parts = (
                model_args.mm_tunable_parts
            )
            # Set the entire model to not require gradients by default
            model.requires_grad_(False)
            vision_tower.requires_grad_(False)
            model.get_model().mm_projector.requires_grad_(False)
            model.get_model().vision_resampler.requires_grad_(False)
            # Parse the mm_tunable_parts to decide which parts to unfreeze
            tunable_parts = model_args.mm_tunable_parts.split(",")
            if "mm_mlp_adapter" in tunable_parts:
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True
            if "mm_vision_resampler" in tunable_parts:
                for p in model.get_model().vision_resampler.parameters():
                    p.requires_grad = True
            if "mm_vision_tower" in tunable_parts:
                for name, param in model.named_parameters():
                    if "vision_tower" in name:
                        param.requires_grad_(True)
            if "mm_language_model" in tunable_parts:
                for name, param in model.named_parameters():
                    if (
                        "vision_tower" not in name
                        and "mm_projector" not in name
                        and "vision_resampler" not in name
                    ):
                        param.requires_grad_(True)
            if "embed_tokens" in tunable_parts:
                for name, param in model.named_parameters():
                    if "embed_tokens" in name:
                        param.requires_grad_(True)
            if "lm_head" in tunable_parts:
                for name, param in model.named_parameters():
                    if "lm_head" in name:
                        param.requires_grad_(True)
            if "moe" in tunable_parts:
                for name, param in model.named_parameters():
                    if "moe" in name:
                        param.requires_grad_(True)
        rank0_print("All Trainable Parameters")
        for name, param in model.named_parameters():
            if param.requires_grad:
                rank0_print(name)

        total_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters()
        )
        trainable_params = sum(
            p.ds_numel if hasattr(p, "ds_numel") else p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
        rank0_print(f"Total parameters: ~{total_params / 1e6:.2f} MB)")
        rank0_print(f"Trainable parameters: ~{trainable_params / 1e6:.2f} MB)")
        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(
                dtype=compute_dtype, device=training_args.device
            )

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = (
            model_args.mm_use_im_start_end
        )
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.mm_vision_tower_lr = training_args.mm_vision_tower_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )
    setattr(training_args, "use_webdataset", getattr(data_args, "use_webdataset"))
    trainer = LLaVATrainer(
        obs_upload_path=training_args.obs_upload_path,
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)

    rank0_print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()
