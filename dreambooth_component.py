import lightning as L

from base_diffusion import BaseDiffusion
import base64
from io import BytesIO

from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel

from diffusion_serve import DreamBoothInput

PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
HF_TOKEN = "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"


class DreamBooth(BaseDiffusion):

    def setup(self, *args, **kwargs):
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
        vae = AutoencoderKL.from_pretrained(PRETRAINED_MODEL_NAME, subfolder="vae", use_auth_token=HF_TOKEN)
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

    def predict(self, data: DreamBoothInput):
        prompt = data.prompt
        if data.quality == "low":
            num_steps = 1
        elif data.quality == "medium":
            num_steps = 10
        elif data.quality == "high":
            num_steps = 50
        else:
            raise ValueError("Invalid quality")
        out = self._model(prompt=prompt, num_inference_steps=num_steps)
        images = out[0]
        image = images[0]
        image.save("image-ref.png")
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return {"image": base64.b64encode(buffered.getvalue())}


app = L.LightningApp(DreamBooth())
