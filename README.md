# Lightning Diffusion

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/).

```python
import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothInput, DreamBoothTuner, encode_to_base64, models
from diffusers import StableDiffusionPipeline


class DreamBoothDiffusion(BaseDiffusion):

    def setup(self):
        self._model = StableDiffusionPipeline.from_pretrained(
            **models.get_kwargs(
                pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4",
                drive=self.weights_drive,
            ),
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


app = L.LightningApp(
    DreamBoothDiffusion(
        serve_cloud_compute=L.CloudCompute("gpu"),
        finetune_cloud_compute=L.CloudCompute("gpu-fast"),
    )
)
```

__________

### Installation

```bash
pip install -e . --extra-index-url https://download.pytorch.org/whl/cu116
```
__________

# Usage:
```
# Run locally
lightning run app app.py

# Run in the cloud
lightning run app app.py --cloud
```
