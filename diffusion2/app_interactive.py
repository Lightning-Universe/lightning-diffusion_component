# !pip install nicegui
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import os
import inspect
import asyncio, torch, base64, functools, time
from io import BytesIO
from typing import Any, Callable, Optional
from ldm.lightning import LightningStableDiffusion, PromptDataset
from nicegui import ui
import openai

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    return await asyncio.get_event_loop().run_in_executor(
        None, functools.partial(callback, *args, **kwargs)
    )


def webpage(
    predict_fn: Callable, host: str, port: int, reference_inference_time: Optional[float], source: Optional[str]
):    

    async def progress_tracker():
        if progress.value >= 1.0 or progress.value == 0:
            return
        progress.value = round((progress.value * reference_inference_time + 0.1) / reference_inference_time, 3)

    async def generate_image():
        nonlocal reference_inference_time
        t0 = time.time()
        progress.value = 0.0001
        image.source = (
            "https://dummyimage.com/600x400/ccc/000000.png&text=building+image..."
        )
        prediction = await io_bound(predict_fn, text=prompt.value)
        image.source = prediction["image"]
        progress.value = 1.0
        reference_inference_time = (time.time() - t0)

    # User Interface
    with ui.row().style("gap:10em"):
        with ui.column():
            ui.label("Stable Diffusion 2.0 with Lightning AI").classes("text-2xl")
            prompt = ui.input("prompt").style("width: 50em")
            prompt.value = "a dragon with wings made of fire"
            ui.button("Generate", on_click=generate_image).style("width: 15em")
            progress = ui.linear_progress()
            ui.timer(interval=0.1, callback=progress_tracker)
            image = ui.image().style("width: 60em")
            if source:
                image.source = source

    # Note: Hack to enable running in spawn context.
    def stack_patch():
        class FakeFrame:
            filename = "random"
        return [FakeFrame(), None]
    inspect.stack = stack_patch
    ui.run(host=host, port=port, reload=False)


class DiffusionServeInteractive(L.LightningWork):

    _start_method = "spawn"

    def setup(self):
        if not os.path.exists("768-v-ema.ckpt"):
            os.system(
                "curl https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
            )
            os.system("echo *.ckpt > .lightningignore")

        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=16,
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

    def predict(self, text):
        image = self._trainer.predict(
            self._model,
            torch.utils.data.DataLoader(PromptDataset([text])),
        )[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}
         
    def run(self):
        self.setup()
        t0 = time.time()
        image = self.predict('a dragon with wings made of fire')["image"]
        webpage(self.predict, self.host, self.port, time.time() - t0, image)


component = DiffusionServeInteractive(
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80)
)

app = L.LightningApp(component)
