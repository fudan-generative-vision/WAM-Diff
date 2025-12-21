import os

AVAILABLE_MODELS = {
    "llava_llada": "LlavaLLaDAModelLM, LlavaLLaDAConfig",
}

for model_name, model_classes in AVAILABLE_MODELS.items():
    exec(f"from .language_model.{model_name} import {model_classes}")