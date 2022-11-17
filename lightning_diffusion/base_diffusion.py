import abc

import lightning as L
from typing import Optional
from copy import deepcopy
from lightning.app.utilities.app_helpers import is_overridden
from lightning_diffusion.diffusion_serve import DiffusionServe
from lightning_diffusion.lite_finetuner import Finetuner
from lightning.app.storage import Drive


def trimmed_flow(flow: 'L.LightningFlow') -> 'L.LightningFlow':
    """Trims a flow to not have any of the internal attributes.
    """
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
        return {'name': 'API', 'content': self.server}


class BaseDiffusion(L.LightningFlow, abc.ABC):

    def __init__(
        self,
        finetune_cloud_compute: Optional[L.CloudCompute] = None,
        serve_cloud_compute: Optional[L.CloudCompute] = None,
        num_replicas=1
    ):
        super().__init__()
        if not is_overridden("predict", instance=self, parent=BaseDiffusion):
            raise Exception("The predict method needs to be overriden.")

        self.weights_drive = Drive("lit://weights")
        self._model = None

        _trimmed_flow = trimmed_flow(self)

        self.finetuner = None
        if is_overridden("finetune", instance=self, parent=BaseDiffusion):
            self.finetuner = Finetuner(
                flow=_trimmed_flow,
                cloud_compute=finetune_cloud_compute,
            )

        self.load_balancer = LoadBalancer(
            DiffusionServe(_trimmed_flow),
            num_replicas=num_replicas,
        )

    @property
    def model(self):
        return self._model

    @property
    def tuner_config(self):
        return self._tuner_config

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
        return {'name': 'API', 'content': self.load_balancer.url}
