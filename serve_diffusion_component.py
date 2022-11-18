# !pip install lightning_diffusion@git+https://github.com/Lightning-AI/lightning-diffusion.git
import lightning as L
import torch, diffusers
from lightning_diffusion import BaseDiffusion, models


class ServeDiffusion(BaseDiffusion):

    def setup(self, *args, **kwargs):
        self._model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.extras
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, data):
        out = self._model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
