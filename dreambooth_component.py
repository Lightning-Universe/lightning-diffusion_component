import lightning as L
from base_diffusion import BaseDiffusion
from lightning.app.storage import Drive
from diffusers import StableDiffusionPipeline
from diffusion_serve import DreamBoothInput
from dreambooth import DreamBoothTuner
from image import encode_to_base64
import models

class DreamBoothDiffusion(BaseDiffusion):

    def __init__(self):
        super().__init__()
        self.weights_drive = Drive("lit://weights")

    def setup(self):
        self.model = StableDiffusionPipeline(
            text_encoder=models.create_text_encoder(),
            vae=models.create_vae(),
            unet=models.create_unet(self.weights_drive),
            tokenizer=models.create_tokenizer(),
            scheduler=models.create_scheduler(),
            safety_checker=models.create_safety_checker(),
            feature_extractor=models.create_feature_extractor(),
        )

    def finetune(self):
        DreamBoothTuner(
            image_urls=[
                "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
                "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
                "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
                "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
                ## You can change or add additional images here
            ],
            prompt="a photo of [sks] [cat clay toy] [riding a bicycle]",
        ).run(self.model)

    def predict(self, data: DreamBoothInput):
        images = self.model(prompt=data.prompt)[0]
        return {"images": encode_to_base64(images)}


app = L.LightningApp(DreamBoothDiffusion())