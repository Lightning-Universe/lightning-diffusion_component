# !pip install nicegui
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference.yaml -o v2-inference.yaml
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import base64
from typing import Optional, Any, Callable
from pydantic import BaseModel
from io import BytesIO
import lightning as L
from ldm.lightning import PromptDataset, LightningStableDiffusion
from pathlib import Path
from torch.utils.data import DataLoader
from nicegui import ui
import asyncio
import functools


class Text(BaseModel):
    text: Optional[str]


async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


def webpage(predict_fn: Callable, host: Optional[str] = None, port: Optional[int] = None):

    async def generate_image():
        image.source = "https://dummyimage.com/600x400/ccc/000000.png&text=building+image..."
        prediction = await io_bound(predict_fn, request=Text(text=prompt.value))
        image.source = prediction['image']

    # User Interface
    with ui.row().style("gap:10em"):
        with ui.column():
            ui.label("Stable Diffusion 2.0 with Lightning.AI").classes("text-2xl")
            prompt = ui.input("prompt").style("width: 20em")
            ui.button("Generate", on_click=generate_image).style("width: 15em")
            image = ui.image().style("width: 60em")

    ui.run(host=host, port=port, reload=False)


class DiffusionServeInteractive(L.LightningWork):

    def setup(self):
        weights_folder = Path("resources/stable_diffusion_weights")
        weights_folder.mkdir(parents=True, exist_ok=True)

        if not os.path.exists("checkpoint.ckpt"):
            os.system("curl https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt -o checkpoint.ckpt")
            os.system("echo checkpoint.ckpt > .lightningignore ")

        running_local = os.getenv("LIGHTNING_CLOUD_APP_ID", None) is None

        if running_local:
            precision = 32
            accelerator = "cpu"
        else:
            precision = 16 if torch.cuda.is_available() else 32
            accelerator = "auto"

        self._trainer = L.Trainer(accelerator=accelerator, devices=1, precision=precision, enable_progress_bar=False)

        self._model = LightningStableDiffusion(
            config_path="v2-inference.yaml", checkpoint_path="checkpoint.ckpt", device=self._trainer.strategy.root_device.type
        )

        if not running_local and torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        image = self._trainer.predict(self._model, DataLoader(PromptDataset([request.text])))[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

    def run(self):
        self.setup()
        webpage(self.predict, host=self.host, port=self.port)


component = DiffusionServeInteractive(cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80))

app = L.LightningApp(component)
