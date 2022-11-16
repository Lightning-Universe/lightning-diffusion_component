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
                 safety_filter=None,
                 **kwargs):
        super().__init__(host=host, port=port, **kwargs)
        self.parent_flow = parent_flow
        self.safety_filter = safety_filter

    def setup(self):
        self.parent_flow.setup()

    def predict(self, prompt: str):
        return self.parent_flow.predict(prompt)




