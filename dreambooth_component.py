import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothTuner, models
from diffusers import StableDiffusionPipeline
from lightning_diffusion.model_cloud import download_from_lightning_cloud


class ServeDreamBoothDiffusion(BaseDiffusion):

    def setup(self):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest",output_dir="model") 
        self._model = StableDiffusionPipeline.from_pretrained(
            **models.get_kwargs("model", self.weights_drive),
            revision= "fp16",
        ).to(self.device)

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
        out = self.model(prompt=data.prompt)
        return {"image": self.serialize(out[0][0])}



app = L.LightningApp(
    ServeDreamBoothDiffusion(
        serve_cloud_compute=L.CloudCompute("gpu", disk_size=80),
        finetune_cloud_compute=L.CloudCompute("gpu-fast", disk_size=80),
    )
)
