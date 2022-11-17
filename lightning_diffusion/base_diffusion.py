import abc

from typing import Optional

import lightning as L

from lightning.app.utilities.app_helpers import is_overridden
from diffusion_serve import DiffusionServe
from lite_finetuner import Finetuner


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

        self.finetuner = None
        if is_overridden("finetune", instance=self, parent=BaseDiffusion):
            self.finetuner = Finetuner(flow=self)

        self.load_balancer = LoadBalancer(DiffusionServe(self), num_replicas=num_replicas)

        self._model = None

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
                self.server.run()
        else:
            self.server.run()

    def configure_layout(self):
        return {'name': 'API', 'content': self.load_balancer.url}
