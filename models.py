from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from lightning.app.storage import Drive
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import torch


HF_TOKEN = "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"
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"


def create_text_encoder():
    return CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder", use_auth_token=HF_TOKEN)

def create_vae():
    return AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=HF_TOKEN)

def create_unet(drive: Drive):
    if drive.list() == ["model.pt"]:
        drive.get("model.pt")
        source = "model.pt"
    else:
        source = pretrained_model_name_or_path
    return UNet2DConditionModel.from_pretrained(source, subfolder="unet", use_auth_token=HF_TOKEN, torch_dtype=torch.float32)

def create_tokenizer():
    return CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_auth_token=HF_TOKEN)

def create_scheduler():
    return DDPMScheduler.from_config(pretrained_model_name_or_path, subfolder="scheduler")

def create_feature_extractor():
    return CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

def create_safety_checker():
    return StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")