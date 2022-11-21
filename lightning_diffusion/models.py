from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from lightning.app.storage import Drive
from typing import Dict
import torch


HF_TOKEN = "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"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


def create_text_encoder():
    return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", revision="fp16", use_auth_token=HF_TOKEN)

def create_vae():
    return AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae", revision="fp16", use_auth_token=HF_TOKEN)

def create_unet(drive: Drive):
    return UNet2DConditionModel.from_pretrained(pretrained_model_name_or_path, subfolder="unet", revision="fp16", use_auth_token=HF_TOKEN, torch_dtype=torch.float32)

def create_tokenizer():
    return CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=HF_TOKEN)

def create_scheduler():
    return DDPMScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )

def create_feature_extractor():
    return None

def create_safety_checker():
    return None


extras = {
    "revision": "fp16",
    "use_auth_token": "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",
}


def get_kwargs(pretrained_model_name_or_path: str, drive = None) -> Dict[str, str]:
    kwargs = {
        "revision": "fp16",
        "use_auth_token": "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",
        "pretrained_model_name_or_path": pretrained_model_name_or_path
    }

    if drive.list() == ["model.pt"]:
        drive.get("model.pt", overwrite=True)
        kwargs = {"pretrained_model_name_or_path": "./model.pt"}

    print(kwargs)

    return kwargs
