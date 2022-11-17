from dreambooth import DreamBoothTuner
import lightning as L
from diffusers import StableDiffusionPipeline
import models
from lightning.app.storage import Drive

class DreamBooth(L.LightningWork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_drive = Drive("lit://weights")

    def setup(self):
        self._model = StableDiffusionPipeline(
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
        ).run(self._model)

    def run(self):
        self.setup()
        self.finetune()


app = L.LightningApp(DreamBooth())