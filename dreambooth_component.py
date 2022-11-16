from lightning.app import LightningApp

from base_diffusion import BaseDiffusion
import base64
from io import BytesIO
from typing import Any

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker


PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
HF_TOKEN = "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"


class DreamBooth(BaseDiffusion):

    def setup(self):
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="vae", use_auth_token=HF_TOKEN)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")
        default_safety_filter = StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker")
        safety_checker = self._safety_filter if self._safety_filter else default_safety_filter
        feature_extractor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")

        if self.model_path:
            self.model_path.get()
            unet_path = self.model_path
        else:
            unet_path = PRETRAINED_MODEL_NAME
        unet = UNet2DConditionModel.from_pretrained(unet_path, subfolder="unet", use_auth_token=HF_TOKEN)

        self._model = StableDiffusionPipeline(
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
            tokenizer=tokenizer,
            scheduler=scheduler,
            safety_checker=safety_checker,
            feature_extractor=feature_extractor
        )

    def predict(self, prompt: Any):
        out = self._model(prompt.payload)
        print(type(out))
        print(out)
        image = out["sample"][0]
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue())


app = LightningApp(DreamBooth())
