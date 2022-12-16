# !pip install diffusers
# !pip install 'git+https://github.com/Lightning-AI/LAI-API-Access-UI-Component.git@diffusion'
# !pip install 'git+https://github.com/Lightning-AI/stablediffusion.git@lit-no-progressbar'
# !curl https://raw.githubusercontent.com/Lightning-AI/stablediffusion/main/configs/stable-diffusion/v2-inference-v.yaml -o v2-inference-v.yaml
import lightning as L
import torch, torch.utils.data as data
import os, base64, pydantic, io, ldm, typing, diffusers
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
        self._model = diffusers.StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")

    def predict(self, requests):
        batch_size = len(requests.inputs)
        print(f"start predicting with batch size {batch_size}")
        texts = [request.text for request in requests.inputs]
        print(texts)
        images = self._model(prompt=texts, num_inference_steps=23).images
        results = []
        for image in images:
            print(image)
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            image_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
            results.append(image_str)
        print(f"finish predicting with batch size {batch_size}")
        return BatchResponse(outputs=[{"image": image_str} for image_str in results])

class Text(pydantic.BaseModel):
    text: str

class BatchText(pydantic.BaseModel):
    # Note: field name must be `inputs`
    inputs: typing.List[Text]

class BatchResponse(pydantic.BaseModel):
    # Note: field name must be `outputs`
    outputs: typing.List[L.app.components.Image]

component = L.app.components.AutoScaler(
    # work cls and args
    DiffusionServer,
    cloud_compute=L.CloudCompute("gpu-rtx", disk_size=80),
    # autoscaler args
    min_replicas=1,
    max_replicas=12,
    endpoint="/predict",
    autoscale_interval=10,
    max_batch_size=8,
    timeout_batching=3,
    input_type=Text,
    output_type=L.app.components.Image,
)

app = L.LightningApp(component)
