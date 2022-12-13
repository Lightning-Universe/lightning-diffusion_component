# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
# !pip install lightning_triton@git+https://github.com/Lightning-AI/LAI-Triton-Serve-Component.git
import lightning as L
import torch, torch.utils.data as data
import os, base64, pydantic, io, ldm, lightning_triton as lt
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DiffusionServer(lt.TritonServer):
    def setup(self):
        cmd = "curl -C - https://pl-public-data.s3.amazonaws.com/dream_stable_diffusion/768-v-ema.ckpt -o 768-v-ema.ckpt"
        os.system(cmd)

        precision = 16 if torch.cuda.is_available() else 32
        self._trainer = L.Trainer(
            accelerator="auto",
            devices=1,
            precision=precision,
            enable_progress_bar=False,
        )

        self._model = ldm.lightning.LightningStableDiffusion(
            config_path="v2-inference-v.yaml",
            checkpoint_path="768-v-ema.ckpt",
            device=self._trainer.strategy.root_device.type,
        )

        if torch.cuda.is_available():
            self._model = self._model.to(torch.float16)
            torch.cuda.empty_cache()

    def predict(self, request):
        image = self._trainer.predict(
            self._model,
            data.DataLoader(ldm.lightning.PromptDataset([request.text])),
        )[0][0]
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return {"image": img_str}

class Text(pydantic.BaseModel):
    text: str

component = DiffusionServer(
    input_type=Text,
    output_type=L.app.components.Image,
    cloud_compute=L.CloudCompute("gpu-fast", disk_size=80, shm_size=512)
)

app = L.LightningApp(component)
