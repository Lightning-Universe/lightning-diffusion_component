# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/runwayml/stable-diffusion/main/configs/stable-diffusion/v1-inference.yaml -o v1-inference.yaml


import lightning as L
import base64, io, os, torch
from ldm.lightning import LightningStableDiffusion, PromptDataset


class ServeDiffusion(L.app.components.serve.PythonServer):
    def __init__(self, input_type=L.app.components.Text, output_type=L.app.components.Image, **kwargs):
        super().__init__(input_type=input_type, output_type=output_type, **kwargs)
        self._model = None

    def setup(self):
        os.system(
            "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/v1-5-pruned-emaonly.ckpt -o v1-5-pruned-emaonly.ckpt"
        )
        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=16 if torch.cuda.is_available() else 32,
            enable_progress_bar=False,
            inference_mode=False,
        )
        self._model = LightningStableDiffusion(
            config_path="v1-inference.yaml",
            checkpoint_path="v1-5-pruned-emaonly.ckpt",
            device=self._trainer.strategy.root_device.type,
            size=512,
        )
        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()


    def predict(self, request):
        with torch.no_grad():
            image = self._trainer.predict(
                self._model,
                torch.utils.data.DataLoader(PromptDataset([request.text])),
            )[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": img_str}


app = L.LightningApp(ServeDiffusion())
