from typing import Optional

from lightning.app.components.serve import PythonServer
from lightning.app import LightningFlow


PRETRAINED_MODEL_NAME = "CompVis/stable-diffusion-v1-4"
HF_TOKEN = "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"


class DiffusionServe(PythonServer):
    def __init__(self,
                 parent_flow: LightningFlow,
                 host: Optional[str] = "127.0.0.1",
                 port: Optional[int] = 7777,
                 **kwargs):
        super().__init__(host=host, port=port, **kwargs)
        self._parent_flow = parent_flow

    def setup(self):
        self._parent_flow.setup()

    def predict(self, prompt: str):
        return self._parent_flow.predict(prompt)

