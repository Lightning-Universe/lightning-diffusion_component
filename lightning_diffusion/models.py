from typing import Dict

import torch
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from lightning.app.storage import Drive
from transformers import CLIPTextModel, CLIPTokenizer

def get_kwargs(pretrained_model_name_or_path: str, drive=None) -> Dict[str, str]:
    kwargs = {
        "revision": "fp16",
        "use_auth_token": "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",
        "pretrained_model_name_or_path": pretrained_model_name_or_path,
    }

    if drive.list() == ["model.pt"]:
        drive.get("model.pt", overwrite=True)
        kwargs = {"pretrained_model_name_or_path": "./model.pt"}

    return kwargs
