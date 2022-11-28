# !pip install nicegui
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import os
import asyncio, torch, base64, functools
from io import BytesIO
from typing import Any, Callable, Optional
from ldm.lightning import LightningStableDiffusion, PromptDataset
from nicegui import ui
from pydantic import BaseModel


os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class Text(BaseModel):
    text: Optional[str]


async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    return await asyncio.get_event_loop().run_in_executor(
        None, functools.partial(callback, *args, **kwargs)
    )


def webpage(
    predict_fn: Callable, host: Optional[str] = None, port: Optional[int] = None
):
    async def generate_image():
        image.source = (
            "https://dummyimage.com/600x400/ccc/000000.png&text=building+image..."
        )
        prediction = await io_bound(predict_fn, request=Text(text=prompt.value))
        image.source = prediction["image"]

    # User Interface
    with ui.row().style("gap:10em"):
        with ui.column():
            ui.label("Stable Diffusion 2.0 with Lightning AI").classes("text-2xl")
            prompt = ui.input("prompt").style("width: 50em")
            prompt.value = "Woman painting a large red egg in a dali landscape"
            ui.button("Generate", on_click=generate_image).style("width: 15em")
            image = ui.image().style("width: 60em")

    ui.run(host=host, port=port, reload=False)


class DiffusionServeInteractive(L.LightningWork):
    def setup(self):
        os.system(
            "curl https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        )
        os.system("echo *.ckpt > .lightningignore ")

        running_local = os.getenv("LIGHTNING_CLOUD_APP_ID", None) is None

        if running_local:
            precision = 32
            accelerator = "cpu"
        else:
            precision = 16 if torch.cuda.is_available() else 32
            accelerator = "auto"

        self._trainer = L.Trainer(
            accelerator=accelerator,
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        self._model = LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=768,
        )

        if not running_local and torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        image = self._trainer.predict(
            self._model,
            torch.utils.data.DataLoader(PromptDataset([request.text])),
        )[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

    def run(self):
        self.setup()
        # Improve cold start by running a prediction before getting ready.
        self.predict(Text(text="Woman painting a large red egg in a dali landscape"))
        webpage(self.predict, host=self.host, port=self.port)


component = DiffusionServeInteractive(
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80)
)

app = L.LightningApp(component)
