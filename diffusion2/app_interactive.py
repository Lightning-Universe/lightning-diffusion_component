# !pip install Pillow
# !pip install nicegui
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import List, Optional, Dict, Any, Callable
from pydantic import BaseModel
from io import BytesIO
import lightning as L
import base64, torch
from pathlib import Path
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from nicegui import ui
import numpy as np
import asyncio
import functools


class Text(BaseModel):
    text: Optional[str]

<<<<<<< HEAD
    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"text": "sample_data"}


=======
>>>>>>> main
class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]):
        super().__init__()
        self.prompts = prompts

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, i: int) -> str:
        return self.prompts[i]


class StableDiffusionModel(L.LightningModule):
    def __init__(
        self,
        config_path: str,
        checkpoint_path: str,
        device: torch.device,
    ):
        super().__init__()

        config = OmegaConf.load(f"{config_path}")
<<<<<<< HEAD
=======
        config.model.params.unet_config["params"]["use_fp16"] = False
>>>>>>> main
        config.model.params.cond_stage_config["params"] = {"device": device}

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        self.model = instantiate_from_config(config.model)
        self.model.load_state_dict(state_dict, strict=False)

        self.sampler = DDIMSampler(self.model)

    @torch.inference_mode()
    def predict_step(self, prompts: List[str], batch_idx: int):
        batch_size = len(prompts)

        with self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [""])
            c = self.model.get_learned_conditioning(prompts)
            shape = [4, 64, 64]
            samples_ddim, _ = self.sampler.sample(
                S=25,  # Number of inference steps, more steps -> higher quality
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc,
                eta=0.0,
            )

            x_samples_ddim = self.model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
            x_samples_ddim = x_samples_ddim.mul(255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
            pil_results = [Image.fromarray(x_sample) for x_sample in x_samples_ddim]
        return pil_results


async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


def webpage(predict_fn, host: Optional[str] = None, port: Optional[int] = None):

    async def generate_image():
        image.source = 'https://dummyimage.com/600x400/ccc/000000.png&text=building+image...'
        prediction = await io_bound(predict_fn.predict, data=Text(prompt=prompt.value))
        image.source = f"data:image/png;base64,{prediction['image']}"

    # User Interface
    with ui.row().style('gap:10em'):
        with ui.column():
            ui.label('Stable Diffusion 2.0 with Lightning.AI').classes('text-2xl')
            prompt = ui.input('prompt').style('width: 20em')
            ui.button('Generate', on_click=generate_image).style('width: 15em')
            image = ui.image().style('width: 60em')

    ui.run(host=host, port=port, reload=False)


class DiffusionServeInteractive(L.LightningWork):

    def setup(self):
        weights_folder = Path("resources/stable_diffusion_weights")
        weights_folder.mkdir(parents=True, exist_ok=True)

        if not os.path.exists("checkpoint.ckpt"):
            os.system("curl https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/512-base-ema.ckpt -o checkpoint.ckpt")

        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(accelerator="auto", devices=1, precision=precision, enable_progress_bar=False)

        self._model = StableDiffusionModel(
            config_path="v2-inference-v.yaml", checkpoint_path="checkpoint.ckpt", device=self._trainer.strategy.root_device.type
        )

    def predict(self, request):
        image = self._trainer.predict(self._model, DataLoader(PromptDataset([request.text])))[0][0]
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": f"data:image/png;base64,{img_str}"}

    def run(self):
        self.setup()
        webpage(self.predict, host=self.host, port=self.port)


component = DiffusionServeInteractive(cloud_compute=L.CloudCompute('gpu'))

app = L.LightningApp(component)
