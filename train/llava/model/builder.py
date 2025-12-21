#    Copyright 2023 Haotian Liu
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


import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llava.model import *
from llava.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.utils import rank0_print
from trl.import_utils import is_npu_available


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", torch_dtype="float16",attn_implementation="flash_attention_2", customized_config=None, overwrite_config=None, **kwargs):
    kwargs["device_map"] = device_map

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    elif torch_dtype == "float16":
        kwargs["torch_dtype"] = torch.float16
    elif torch_dtype == "bfloat16":
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        import pdb;pdb.set_trace()

    if customized_config is not None:
        kwargs["config"] = customized_config

    if "multimodal" in kwargs:
        if kwargs["multimodal"] is True:
            is_multimodal = True
            kwargs.pop("multimodal")
    else:
        is_multimodal = False

    if "llava" in model_name.lower() or is_multimodal:
        # Load LLaVA model
        rank0_print(f"Loaded LLaVA model: {model_path}")
        if "llada" in model_name.lower() or "exp" in model_name.lower():
            from llava.model.language_model.llava_llada import LlavaLLaDAConfig

            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            if customized_config is None:
                llada_cfg = LlavaLLaDAConfig.from_pretrained(model_path)
            else:
                llada_cfg = customized_config

            if overwrite_config is not None:
                rank0_print(f"Overwriting config with {overwrite_config}")
                for k, v in overwrite_config.items():
                    setattr(llada_cfg, k, v)

            model = LlavaLLaDAModelLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llada_cfg, **kwargs)

        else:
            try:
                from llava.model.language_model.llava_llama import LlavaConfig

                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                if customized_config is None:
                    llava_cfg = LlavaConfig.from_pretrained(model_path)
                    if "v1.5" in model_path.lower():
                        llava_cfg.delay_load = True  # a workaround for correctly loading v1.5 models
                else:
                    llava_cfg = customized_config

                if overwrite_config is not None:
                    rank0_print(f"Overwriting config with {overwrite_config}")
                    for k, v in overwrite_config.items():
                        setattr(llava_cfg, k, v)
                model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, attn_implementation=attn_implementation, config=llava_cfg, **kwargs)
            except:
                raise ValueError(f"Model {model_name} not supported")

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel

            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print("Convert to FP16...")
            model.to(torch.float16)
        else:
            use_fast = False
            if "mpt" in model_name.lower().replace("prompt", ""):
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    rank0_print(f"Model Class: {model.__class__.__name__}")
    image_processor = None

    if "llava" in model_name.lower() or is_multimodal:
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != "auto":
            if is_npu_available():
                vision_tower.to(device="npu", dtype=torch.float16)
            else:
                vision_tower.to(device="cuda", dtype=torch.bfloat16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    elif hasattr(model.config, "max_position_embeddings"):
        context_len = model.config.max_position_embeddings
    elif hasattr(model.config, "tokenizer_model_max_length"):
        context_len = model.config.tokenizer_model_max_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
