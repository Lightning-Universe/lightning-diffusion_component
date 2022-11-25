#!/usr/bin/env python3
import justpy as jp
from lightning.app.components.serve.python_server import _PyTorchSpawnRunExecutor, WorkRunExecutor
from lightning import LightningWork
from nicegui import ui

flow = None

def my_click():
    response = flow.predict()
    print(response)

def webpage(host, port):
    ui.input(label='Text', placeholder='press ENTER to apply',
            on_change=lambda e: input_result.set_text('you typed: ' + e.value))
    input_result = ui.label()

    ui.run(host=host, port=port, reload=False)

class DiffusionServeJuspty(LightningWork):

    def __init__(self, *args, flow, **kwargs):
        super().__init__(*args, flow, **kwargs)
        self._flow = flow

        # Note: Enable to run inference on GPUs.
        # self._run_executor_cls = (
        #     WorkRunExecutor if os.getenv("LIGHTNING_CLOUD_APP_ID", None) else _PyTorchSpawnRunExecutor
        # )

    def run(self):
        global flow
        flow = self._flow

        self._flow.setup()

        webpage(host=self.host, port=self.port,)
