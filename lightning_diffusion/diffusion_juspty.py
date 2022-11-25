#!/usr/bin/env python3
from nicegui import ui
import justpy as jp
from lightning.app.components.serve.python_server import _PyTorchSpawnRunExecutor, WorkRunExecutor
from lightning import LightningFlow
import os

class DiffusionFrontend(LightningFlow):

    def __init__(self, *args, flow, **kwargs):
        super().__init__(*args, flow, **kwargs)
        self._flow = flow

        # Note: Enable to run inference on GPUs.
        self._run_executor_cls = (
            WorkRunExecutor if os.getenv("LIGHTNING_CLOUD_APP_ID", None) else _PyTorchSpawnRunExecutor
        )

    def run(self):
        def my_click(self, *_):
            pass
            # self._flow.setup()
            # response = self._flow.predict()
            # print(response)

        def webpage():
            wp = jp.WebPage()
            d = jp.Div(text="Hello ! Click Me!")
            d.on("click", my_click)
            wp.add(d)
            return wp

        jp.justpy(webpage, host=self.host, port=self.port)
