import lightning as L
import torch.cuda

from base_diffusion import BaseDiffusion

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusion_serve import DreamBoothInput
from utils import image_decode

PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
HF_TOKEN = "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"


class DreamBooth(BaseDiffusion):

    def setup(self, *args, **kwargs):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="vae", use_auth_token=HF_TOKEN).to(device)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        safety_checker = None
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        model_path = kwargs.get("checkpoint_path")
        if model_path:
            if not isinstance(model_path, L.storage.Path):
                raise ValueError("checkpoint_path must be a lightning.storage.Path object")
            model_path.get()
            unet_path = model_path
        else:
            unet_path = PRETRAINED_MODEL_NAME
        unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", use_auth_token=HF_TOKEN).to(device)

        model = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        ).to(device)
        self._model = model

    def predict(self, data: DreamBoothInput):
        print("Predicting...")
        print(data.prompt)
        out = self._model(prompt=data.prompt, num_inference_steps=1)
        return {"image": image_decode(out[0][0])}


app = L.LightningApp(DreamBooth())
