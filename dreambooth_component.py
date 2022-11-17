import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothInput, DreamBoothTuner, encode_to_base64, models
from lightning.app.storage import Drive
from diffusers import StableDiffusionPipeline


class DreamBoothDiffusion(BaseDiffusion):

    def __init__(self):
        super().__init__()
        self.weights_drive = Drive("lit://weights")

    def setup(self):
        self._model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.extras,
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
            prompt="a photo of sks cat clay toy",
            preservation_prompt="a photo of cat clay toy",
            validation_prompt="a photo of sks cat clay toy riding a bicycle",
            max_steps=200,
            gradient_accumulation_steps=1,
        ).run(self._model)

    def predict(self, data: DreamBoothInput):
        images = self._model(prompt=data.prompt)[0]
        return {"images": encode_to_base64(images)}


app = L.LightningApp(DreamBoothDiffusion())