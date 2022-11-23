# Lightning Diffusion Component

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/). For example, save this code snippet as `app.py` and run the below commands


### Serve ANY Diffusion Models

```python
# !pip install torch diffusers lightning_diffusion@git+https://github.com/Lightning-AI/lightning-diffusion.git
import lightning as L
import torch, diffusers
from lightning_diffusion import BaseDiffusion, models


class ServeDiffusion(BaseDiffusion):

    def setup(self, *args, **kwargs):
        self.model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.extras
        ).to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
```

### Serve ANY fine-tuned Diffusion Models

Use the DreamBooth fine-tuning methodology from the paper `Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) as follows:


```python
import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothTuner, models
import torch, diffusers

class ServeDreamBoothDiffusion(BaseDiffusion):

    def setup(self):
        self.model = diffusers.StableDiffusionPipeline.from_pretrained(
            **models.get_kwargs("CompVis/stable-diffusion-v1-4", self.weights_drive),
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
```

### Running locally

```bash
lightning run app {COMPONENT_NAME}.py --setup
```

### Running on cloud

```bash
lightning run app {COMPONENT_NAME}.py --setup --cloud
```
