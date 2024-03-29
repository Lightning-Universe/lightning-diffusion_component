from typing import Any, Dict, Optional

from lightning.app import LightningFlow
from lightning.app.components.serve import PythonServer
from pydantic import BaseModel


class DreamBoothInput(BaseModel):
    prompt: str  # text prompt

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"prompt": "A photo of a person", "quality": "medium"}


class DreamBoothOutput(BaseModel):
    image: str  # base64 encoded image

    @staticmethod
    def _get_sample_data() -> Dict[Any, Any]:
        return {"image": "base64 encoded image"}


class DiffusionServe(PythonServer):
    def __init__(
        self, parent_flow: LightningFlow, host: Optional[str] = "127.0.0.1", port: Optional[int] = 7777, **kwargs
    ):
        super().__init__(host=host, port=port, input_type=DreamBoothInput, output_type=DreamBoothOutput, **kwargs)
        self._parent_flow = parent_flow

    def setup(self, *args, **kwargs):
        self._parent_flow.setup(*args, **kwargs)

    def predict(self, prompt: str):
        return self._parent_flow.predict(prompt)
