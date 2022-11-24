# Lightning Diffusion Component

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/). For example, save this code snippet as `component.py` and run the below commands

### Serve ANY Diffusion Models

```python
# !pip install lightning_diffusion@git+https://github.com/Lightning-AI/lightning-diffusion.git
import lightning as L
import diffusers
from lightning_diffusion import BaseDiffusion, models


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        self.model = diffusers.StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4", **models.extras
        ).to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
```

### Serve ANY fine-tuned Diffusion Models

Use the DreamBooth fine-tuning methodology from the paper \`Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation\](https://arxiv.org/abs/2208.12242) as follows:

```python
import lightning as L
from lightning_diffusion import BaseDiffusion, DreamBoothTuner, models
from diffusers import StableDiffusionPipeline


class ServeDreamBoothDiffusion(BaseDiffusion):
    def setup(self):
        self.model = StableDiffusionPipeline.from_pretrained(
            **models.get_kwargs("CompVis/stable-diffusion-v1-4", self.weights_drive),
        ).to(self.device)

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


app = L.LightningApp(
    ServeDreamBoothDiffusion(
        serve_cloud_compute=L.CloudCompute("gpu", disk_size=80),
        finetune_cloud_compute=L.CloudCompute("gpu-fast", disk_size=80),
    )
)
```

To customize the Diffusion model to your own need, simply provide your own images URL(s) and an associated prompt.

The prompt needs to be in the following format with the `[...]` included.

Reference Format: `A photo of [NOUN] [DESCRIPTIVE CLASS] [DESCRIPTION FOR THE NEW GENERATED IMAGES]`.

Inspired from [here](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) and [here](https://colab.research.google.com/drive/1SyjkeuPrX7kd_xTBKhcvBGEC8G_ml9RU#scrollTo=1lKGmcIyJbCu).

### Running locally

```bash
lightning run app {COMPONENT_NAME}.py --setup
```

### Running on cloud

```bash
lightning run app {COMPONENT_NAME}.py --setup --cloud
```
