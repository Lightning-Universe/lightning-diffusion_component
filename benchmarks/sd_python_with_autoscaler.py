# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import torch, torch.utils.data as data
import os, base64, pydantic, io, ldm, typing
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


class DiffusionServer(L.app.components.PythonServer):
    def __init__(self, *args, **kwargs):
        super().__init__(
            input_type=BatchText,
            output_type=BatchResponse,
            *args,
            **kwargs,
        )

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

    def predict(self, requests):
        batch_size = len(requests.inputs)
        print(f"predicting with batch size {batch_size}")
        texts = []
        for request in requests.inputs:
            texts.append(request.text)

        images = self._trainer.predict(
            self._model,
            data.DataLoader(ldm.lightning.PromptDataset(texts), batch_size=batch_size),
        )[0]
        print(images)
        results = []
        for image in images:
            print(image)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)

        return BatchResponse(outputs=[{"image": image_str} for image_str in results])

class Text(pydantic.BaseModel):
    text: str

class BatchText(pydantic.BaseModel):
    # Note: field name must be `inputs` as of now
    inputs: typing.List[Text]

class BatchResponse(pydantic.BaseModel):
    # Note: field name must be `outputs`` as of now
    outputs: typing.List[L.app.components.Image]

component = L.app.components.AutoScaler(
    # work cls and args
    DiffusionServer,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=3,
    endpoint="/predict",
    autoscale_interval=10,
    max_batch_size=8,
    timeout_batching=3,
)

app = L.LightningApp(component)
