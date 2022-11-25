# ! pip install torch diffusers git+https://github.com/Lightning-AI/lightning-diffusion-component.git
import lightning as L
from lightning_diffusion import BaseDiffusion
from diffusers import DiffusionPipeline, EulerDiscreteScheduler


class ServeDiffusion(BaseDiffusion):
    def setup(self, *args, **kwargs):
        import torch
        repo_id = "stabilityai/stable-diffusion-2"
        scheduler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler", prediction_type="v_prediction")
        self.model = DiffusionPipeline.from_pretrained(repo_id, torch_dtype=torch.float32, revision="fp16", scheduler=scheduler)
        # self.model = model.to(self.device)

    def predict(self, data):
        out = self.model(prompt=data.prompt, num_inference_steps=23)
        return {"image": self.serialize(out[0][0])}


app = L.LightningApp(ServeDiffusion(interactive=True))
