# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import base64
from typing import Optional
from pydantic import BaseModel
from io import BytesIO
import lightning as L
import lightning.app.components.serve as serve
from pathlib import Path
from ldm.lightning import PromptDataset, LightningStableDiffusion
from torch.utils.data import DataLoader


class Text(BaseModel):
    text: Optional[str]


class DiffusionServer(serve.PythonServer):

    def setup(self):
        weights_folder = Path("resources/stable_diffusion_weights")
        weights_folder.mkdir(parents=True, exist_ok=True)

        if not os.path.exists("checkpoint.ckpt"):
            os.system("curl https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt -o checkpoint.ckpt")

        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(accelerator="auto", devices=1, precision=precision, enable_progress_bar=False)

        self._model = LightningStableDiffusion(
            config_path="v2-inference-v.yaml", checkpoint_path="checkpoint.ckpt", device=self._trainer.strategy.root_device.type
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        image = self._trainer.predict(self._model, DataLoader(PromptDataset([request.text])))[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

component = DiffusionServer(
   input_type=Text, output_type=serve.Image, cloud_compute=L.CloudCompute("gpu-rtx")
)

app = L.LightningApp(component)
