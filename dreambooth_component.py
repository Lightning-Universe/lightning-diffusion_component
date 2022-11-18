import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothTuner, models
from diffusers import StableDiffusionPipeline


class DreamBoothDiffusion(BaseDiffusion):

    def setup(self):
        self._model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.get_extras(self.weights_drive),
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

    def predict(self, data):
        images = self.model(prompt=data.prompt)[0]
        return {"images": self.serialize(images)}


app = L.LightningApp(
    DreamBoothDiffusion(
        serve_cloud_compute=L.CloudCompute("gpu"),
        finetune_cloud_compute=L.CloudCompute("gpu-fast"),
    )
)
