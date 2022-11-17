from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from lightning.app.storage import Drive

HF_TOKEN = "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"


def create_text_encoder():
    return CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")


def create_vae():
    return AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=HF_TOKEN),

def create_unet(drive: Drive):
    source = "CompVis/stable-diffusion-v1-4"
    if drive.list() == ["model.pt"]:
        drive.get("model.pt")
        source = "model.pt"
    return UNet2DConditionModel.from_pretrained(source, subfolder="unet", use_auth_token=HF_TOKEN),

def create_tokenizer():
    return CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

def create_scheduler():
    return DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

def create_feature_extractor():
    return CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")


def create_safety_checker():
    return None