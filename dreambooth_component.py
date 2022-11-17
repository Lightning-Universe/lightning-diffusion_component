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
        StableDiffusionPipeline.from_pretrained()
        self._model = model

    def predict(self, data: DreamBoothInput):
        print("Predicting...")
        print(data.prompt)
        out = self._model(prompt=data.prompt, num_inference_steps=1)
        return {"image": image_decode(out[0][0])}


app = L.LightningApp(DreamBooth())
