import lightning as L
import base64, io, base_diffusion, torch
from diffusers import StableDiffusionPipeline

extras = {
    "use_auth_token": "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",
}


class DreamBoothDiffusion(base_diffusion.BaseDiffusion):

    def setup(self, *args, **kwargs):
        self._model = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            **extras
        ).to("cuda" if torch.cuda.is_available() else "cpu")

    def serialize(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, data):
        out = self._model(prompt=data.prompt, num_inference_steps=1)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(DreamBoothDiffusion())
