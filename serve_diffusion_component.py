# !pip install lightning_diffusion@git+https://github.com/Lightning-AI/lightning-diffusion.git
import diffusers
import lightning as L
<<<<<<< HEAD
=======

from lightning_diffusion import BaseDiffusion, models
>>>>>>> main

from lightning_diffusion import BaseDiffusion, models, download_from_lightning_cloud

<<<<<<< HEAD

class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self.model = diffusers.StableDiffusionPipeline.from_pretrained("model").to(self.device)
=======
class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        self.model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", **models.extras
        ).to(self.device)
>>>>>>> main

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
