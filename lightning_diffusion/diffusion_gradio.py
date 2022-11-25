from lightning.app.components import ServeGradio
import gradio as gr
from pydantic import BaseModel
from typing import Dict, Any

class DreamBoothInput(BaseModel):
    prompt: str  # text prompt

    def _get_sample_data() -> Dict[Any, Any]:
        return {"prompt": "A photo of a person", "quality": "medium"}


class DiffusionServeGradio(ServeGradio):

    inputs = gr.inputs.Textbox(label="Text prompt")
    outputs = gr.outputs.Image(type="pil")
    examples = [["A pikachu fine dining with a view to the Eiffel Tower"], ["A high tech solarpunk utopia in the Amazon rainforest"]]

    def __init__(self, *args, flow, **kwargs):
        super().__init__(*args, parallel=True, **kwargs)
        self._flow = flow
        self._process = None

    def build_model(self):
        pass

    def predict(self, prompt):
        if self._flow._model is None:
            self._flow.setup()
        self._flow.predict(DreamBoothInput(prompt))
