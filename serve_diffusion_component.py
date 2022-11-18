import lightning as L
import torch, diffusers
from lightning_diffusion import BaseDiffusion, models


class ServeDiffusion(BaseDiffusion):

    def setup(self, *args, **kwargs):
        self._model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.extras
        ).to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
