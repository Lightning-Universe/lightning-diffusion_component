# ! pip install torch diffusers lightning==1.8.3.post0 git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import lightning as L
from diffusers import StableDiffusionPipeline

from lightning_diffusion import BaseDiffusion, DreamBoothTuner, download_from_lightning_cloud, models


class ServeDreamBoothDiffusion(BaseDiffusion):
    def setup(self):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self.model = StableDiffusionPipeline.from_pretrained(**models.get_kwargs("model", self.weights_drive)).to(
            self.device
        )

    def finetune(self):
        DreamBoothTuner(
            image_urls=[
                "https://lightning-example-public.s3.amazonaws.com/2.jpeg",
                "https://lightning-example-public.s3.amazonaws.com/3.jpeg",
                "https://lightning-example-public.s3.amazonaws.com/5.jpeg",
                "https://lightning-example-public.s3.amazonaws.com/6.jpeg",
                ## You can change or add additional images here
            ],
            prompt="a photo of [sks] [cat clay toy] [riding a bicycle]",
        ).run(self.model)

    def predict(self, data):
        out = self.model(prompt=data.prompt)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDreamBoothDiffusion())
