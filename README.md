# Lightning Diffusion Component

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/).

Create an account to get 3 free credits !

To get started, save this code snippet as `component.py` and run the below at the end of the README.

### Serve ANY Diffusion Models

```python
# ! pip install torch diffusers git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import diffusers
import lightning as L
from lightning_diffusion import BaseDiffusion, download_from_lightning_cloud


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self.model = diffusers.StableDiffusionPipeline.from_pretrained("model").to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
```

Run the component for free directly [there](https://lightning.ai/component/UJ7stJI225-Serve%20Dreambooth%20Diffusion).

### Serve ANY fine-tuned Diffusion Models

Use the DreamBooth fine-tuning methodology from the paper \`Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation\](https://arxiv.org/abs/2208.12242) as follows:

```python
# ! pip install torch diffusers lightning==1.8.3.post0 git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import lightning as L
from diffusers import StableDiffusionPipeline
from lightning_diffusion import BaseDiffusion, DreamBoothTuner, models, download_from_lightning_cloud


class ServeDreamBoothDiffusion(BaseDiffusion):
    def setup(self):
        download_from_lightning_cloud("daniela/stable_diffusion", version="latest", output_dir="model")
        self.model = StableDiffusionPipeline.from_pretrained(
            **models.get_kwargs("model", self.weights_drive)
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


app = L.LightningApp(ServeDreamBoothDiffusion())
```

Run the component for free directly [there](https://lightning.ai/component/67826ad38c-Serve%20Dreambooth%20Diffusion).

To customize the Diffusion model to your own need, simply provide your own images URL(s) and an associated prompt.

The prompt needs to be in the following format with the `[...]` included.

Reference Format: `A photo of [NOUN] [DESCRIPTIVE CLASS] [DESCRIPTION FOR THE NEW GENERATED IMAGES]`.

Inspired from [here](https://github.com/ShivamShrirao/diffusers/blob/main/examples/dreambooth/train_dreambooth.py) and [here](https://colab.research.google.com/drive/1SyjkeuPrX7kd_xTBKhcvBGEC8G_ml9RU#scrollTo=1lKGmcIyJbCu).

### Serve ANY Diffusion Models

```python
# ! pip install torch diffusers git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import diffusers
import lightning as L
from lightning.app.components import AutoScaler
from lightning_diffusion import BaseDiffusion, download_from_lightning_cloud

class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        download_from_lightning_cloud(
            "daniela/stable_diffusion", version="latest", output_dir="model")
        self.model = diffusers.StableDiffusionPipeline.from_pretrained("model").to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}

app = L.LightningApp(
    AutoScaler(
        ServeDiffusion,
        max_replicas=8,
        autoscale_interval=30,
    )
)
```

Run the component for free directly [there](https://lightning.ai/component/UJ7stJI225-Serve%20Dreambooth%20Diffusion).



### Running locally

```bash
lightning run app {COMPONENT_NAME}.py --setup
```

### Running on cloud

```bash
lightning run app {COMPONENT_NAME}.py --setup --cloud
```
