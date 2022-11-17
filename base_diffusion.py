import abc
from copy import deepcopy

import lightning as L

from lightning.app.utilities.app_helpers import is_overridden

from diffusion_serve import DiffusionServe
from lambda_work import LambdaWork


def trimmed_flow(flow: 'LightningFlow') -> 'LightningFlow':
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

    def __init__(self, num_replicas=1):
        super().__init__()
        if not is_overridden("predict", instance=self, parent=BaseDiffusion):
            raise Exception("The predict method needs to be overriden.")

        self._model = None
        self.finetuner = None
        if is_overridden("finetune", instance=self, parent=BaseDiffusion):
            self.finetuner = LambdaWork(self.finetune, parallel=False)
        self.load_balancer = LoadBalancer(DiffusionServe(parent_flow=trimmed_flow(self), cloud_compute=L.CloudCompute(name="cpu-medium", disk_size=100)), num_replicas=num_replicas)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @abc.abstractmethod
    def setup(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def predict(self, request):
        pass

    def finetune(self, drive: L.storage.Drive):
        raise NotImplementedError("Fine tuning is not implemented.")

    def run(self):
        if self.finetuner:
            self.finetuner.run()
        self.load_balancer.run()

    def configure_layout(self):
        return {'name': 'API', 'content': self.load_balancer.url}
