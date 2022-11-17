from dreambooth import DreamBoothTuner
import lightning as L
from diffusers import StableDiffusionPipeline
from lightning.app.components import LiteMultiNode
import models


class DreamBoothWork(L.LightningWork):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weights_drive = L.app.storage.Drive("lit://weights")

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
        ).run(self._model)

    def run(self):
        self.setup()
        self.finetune()


class DreamBooth(LiteMultiNode):

    def __init__(
        self,
        *args,
        cloud_compute = L.CloudCompute("gpu-fast"),
        num_nodes: int = 1,
        **kwargs
    ):
        super().__init__(
            *args,
            work_cls=DreamBoothWork,
            num_nodes=num_nodes,
            cloud_compute=cloud_compute,
            **kwargs
        )

app = L.LightningApp(DreamBooth())