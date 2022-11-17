from lightning.app.storage import Path

from base_diffusion import BaseDiffusion
import base64
from io import BytesIO

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from diffusion_serve import DreamBoothInput

HF_TOKEN = "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"


class DreamBooth(BaseDiffusion):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path = None

    def setup(self):
        self._model = StableDiffusionPipeline(
            text_encoder=CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14"),
            vae=AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="vae", use_auth_token=HF_TOKEN),
            unet=UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=HF_TOKEN),
            tokenizer=CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14"),
            scheduler=DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear"),
            safety_checker=None,
            feature_extractor=CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
        )

    def predict(self, data: DreamBoothInput):
        out = self._model(prompt=data.prompt)
        images = out[0]
        image = images[0]
        image.save("image-ref.png")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return {"image": base64.b64encode(buffered.getvalue())}
