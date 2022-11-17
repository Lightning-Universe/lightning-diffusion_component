import lightning as L
from lightning.app.components import LiteMultiNode


class FlowLambdaWork(L.LightningWork):

    def __init__(self, *args, flow, **kwargs):
        super().__init__(*args, **kwargs)
        self._flow = flow

    def run(self):
        self._flow.setup()
        self._flow.finetune()


class Finetuner(LiteMultiNode):

    def __init__(
        self,
        *args,
        cloud_compute = L.CloudCompute("gpu-fast"),
        num_nodes: int = 1,
        **kwargs
    ):
        super().__init__(
            *args,
            work_cls=FlowLambdaWork,
            num_nodes=num_nodes,
            cloud_compute=cloud_compute,
            **kwargs
        )

    @property
    def has_succeeded(self) -> bool:
        return len(self.works()) > 0 and all(w.has_stopped for w in self.works())