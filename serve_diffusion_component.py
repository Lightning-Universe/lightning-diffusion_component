import lightning as L
import base64, io, base_diffusion, torch, models
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
