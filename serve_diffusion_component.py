# ! pip install torch diffusers git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import diffusers
import lightning as L
import torch

from lightning_diffusion import BaseDiffusion, download_from_lightning_cloud


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        download_from_lightning_cloud(
            "daniela/stable_diffusion", version="latest", output_dir="model"
        )
        self.model = diffusers.StableDiffusionPipeline.from_pretrained("model").to(
            self.device
        )

    def predict(self, data):
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            out = self.model(prompt=data.prompt, num_inference_steps=23)
            return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
