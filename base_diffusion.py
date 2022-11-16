import abc

from lightning.app import LightningFlow, LightningWork
from lightning.app.storage import Drive

from lightning.app.utilities.app_helpers import is_overridden

from diffusion_serve import DiffusionServe
from lambda_work import LambdaWork


class LoadBalancer(LightningFlow):
    def __init__(self, server: LightningWork, num_replicas: int = 1):
        super().__init__()
        self.server = server
        self.num_replicas = num_replicas

    def run(self):
        self.server.run()


class BaseDiffusion(LightningFlow, abc.ABC):

    def __init__(self, num_replicas=1):
        super().__init__()
        if not is_overridden("predict", instance=self, parent=BaseDiffusion):
            raise Exception("The predict method needs to be overriden.")

        self.finetuner = None
        if is_overridden("finetune", instance=self, parent=BaseDiffusion):
            self.finetuner = LambdaWork(self.finetune, parallel=False)

        backend = self._backend
        self._backend = None
        self.load_balancer = LoadBalancer(DiffusionServe(self), num_replicas=num_replicas)
        self._backend = backend

    @abc.abstractmethod
    def predict(self, request):
        pass

    def finetune(self, drive: Drive):
        raise NotImplementedError("Fine tuning is not implemented.")

    def run(self):
        print("running base")
        if self.finetuner:
            self.finetuner.run()
        self.load_balancer.run()
