import abc
import base64
import io
import os
from copy import deepcopy
from typing import Optional
import platform
import lightning as L
from diffusers import StableDiffusionPipeline
from lightning.app.storage import Drive
from lightning.app.utilities.app_helpers import is_overridden

import operator
import platform
from lightning_utilities.core.imports import compare_version

from lightning_diffusion.diffusion_serve import DiffusionServe
from lightning_diffusion.lite_finetuner import Finetuner
from lightning_diffusion.diffusion_gradio import DiffusionServeGradio


def trimmed_flow(flow: "L.LightningFlow") -> "L.LightningFlow":
    """Trims a flow to not have any of the internal attributes."""
    backend = flow._backend
    flow._backend = None
    for f in flow.flows:
        f._backend = None
    for w in flow.works():
        w._backend = None

    # also taking a deep copy
    flow_copy = deepcopy(flow)
    if backend:
        L.LightningFlow._attach_backend(flow, backend)
    return flow_copy


class LoadBalancer(L.LightningFlow):
    def __init__(self, server: L.LightningWork, num_replicas: int = 1):
        super().__init__()
        self.server = server
        self.num_replicas = num_replicas
        self.url = ""

    def run(self):
        self.server.run()
        self.url = self.server.url

    def configure_layout(self):
        return {"name": "API", "content": self.server}


class BaseDiffusion(L.LightningFlow, abc.ABC):
    def __init__(
        self,
        finetune_cloud_compute: Optional[L.CloudCompute] = L.CloudCompute("gpu-fast", disk_size=80),
        serve_cloud_compute: Optional[L.CloudCompute] = L.CloudCompute("gpu", disk_size=80),
        num_replicas=1,
        gradio: bool = False,
    ):
        super().__init__()
        if not is_overridden("predict", instance=self, parent=BaseDiffusion):
            raise Exception("The predict method needs to be overriden.")

        self.weights_drive = Drive("lit://weights")
        self._model = None
        self._device = None
        self.gradio = gradio

        _trimmed_flow = trimmed_flow(self)

        self.finetuner = None
        if is_overridden("finetune", instance=self, parent=BaseDiffusion):
            self.finetuner = Finetuner(
                flow=_trimmed_flow,
                cloud_compute=finetune_cloud_compute,
            )

        if not self.gradio:    
            self.load_balancer = LoadBalancer(
                DiffusionServe(
                    _trimmed_flow,
                    cloud_compute=serve_cloud_compute,
                    # Starts only if there isn't a finetuner
                    start_with_flow=self.finetuner is None,
                ),
                num_replicas=num_replicas,
            )
        else:
            self.load_balancer = DiffusionServeGradio(
                flow=_trimmed_flow,
                cloud_compute=serve_cloud_compute,
                start_with_flow=self.finetuner is None,
            )

    @staticmethod
    def serialize(image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @property
    def model(self) -> StableDiffusionPipeline:
        assert self._model
        return self._model

    @model.setter
    def model(self, model) -> None:
        self._model = model

    @property
    def device(self):
        import torch

        _TORCH_GREATER_EQUAL_1_12 = compare_version("torch", operator.ge, "1.12.0")

        local_rank = int(os.getenv("LOCAL_RANK", "0"))

        if _TORCH_GREATER_EQUAL_1_12 and torch.backends.mps.is_available() and platform.processor() in ("arm", "arm64"):
            return torch.device("mps", local_rank)
        else:
            return f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, request):
        pass

    def finetune(self):
        raise NotImplementedError("Fine tuning is not implemented.")

    def run(self):
        if self.finetuner:
            self.finetuner.run()
            if self.finetuner.has_succeeded:
                self.load_balancer.run()
        else:
            self.load_balancer.run()

    def configure_layout(self):
        name = "Demo" if isinstance(self.load_balancer, DiffusionServeGradio) else "API"
        return [{"name": name, "content": self.load_balancer.url}]
