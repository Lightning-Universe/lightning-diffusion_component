# !pip install lightning_diffusion@git+https://github.com/Lightning-AI/lightning-diffusion.git
import diffusers
import lightning as L

from lightning_diffusion import BaseDiffusion, models


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        self.model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", **models.extras
        ).to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
