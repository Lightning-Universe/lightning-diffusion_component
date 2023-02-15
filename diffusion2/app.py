# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import base64
import os
from io import BytesIO
from typing import Optional

import lightning as L
import lightning.app.components.serve as serve
import torch
from ldm.lightning import LightningStableDiffusion, PromptDataset
from pydantic import BaseModel

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Text(BaseModel):
    text: Optional[str]


class DiffusionServer(serve.PythonServer):
    def setup(self):
        os.system(
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        )
        os.system("echo *.ckpt > .lightningignore ")

        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
            inference_mode=torch.cuda.is_available(),
        )

        self._model = LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=768,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        image = self._trainer.predict(self._model, torch.utils.data.DataLoader(PromptDataset([request.text])),)[
            0
        ][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}


component = DiffusionServer(
    input_type=Text,
    output_type=serve.Image,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
)

app = L.LightningApp(component)
