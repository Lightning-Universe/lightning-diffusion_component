# Lightning Diffusion

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/).

```python
import lightning as L
import base64, io, torch
from lightning_diffusion import base_diffusion, models
from diffusers import StableDiffusionPipeline


class ServeDiffusion(base_diffusion.BaseDiffusion):

    def setup(self, *args, **kwargs):
        self._model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **models.extras
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def serialize(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, data):
        out = self._model(prompt=data.prompt)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion())
```