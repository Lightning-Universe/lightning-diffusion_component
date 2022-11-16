from lightning import LightningApp

from diffusion_serve import DiffusionServe

from typing import Optional

import typing
from lightning.app.core.flow import LightningFlow

if typing.TYPE_CHECKING:
    from lightning.app.core.work import LightningWork


class Serve(LightningFlow):
    def __init__(self,
                 server: 'LightningWork',
                 finetuner: Optional['LightningWork'] = None,
                 loadbalancer: Optional['LightningWork'] = None
                 ):
        super().__init__()
        self.server = server
        self.finetuner = finetuner

    def run(self):
        if self.finetuner:
            self.finetuner.run()
        model_path = self.finetuner.model_path if self.finetuner else None
        self.server.run(model_path)


app = LightningApp(
    Serve(server=DiffusionServe(safety_filter=None), loadbalancer=None, finetuner=None)
)
