import lightning as L
from nicegui import ui
from typing import Optional
import asyncio
import functools
from typing import Callable, Any
from pydantic import BaseModel

class DreamBoothInput(BaseModel):
    prompt: str  # text prompt


async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


def webpage(flow, host: Optional[str] = None, port: Optional[int] = None):

    async def generate_image():
        image.source = 'https://dummyimage.com/600x400/ccc/000000.png&text=building+image...'
        prediction = await io_bound(flow.predict, data=DreamBoothInput(prompt=prompt.value))
        image.source = f"data:image/png;base64,{prediction['image']}"

    # User Interface
    with ui.row().style('gap:10em'):
        with ui.column():
            ui.label('Stable Diffusion 2.0 with Lightning.AI').classes('text-2xl')
            prompt = ui.input('prompt').style('width: 20em')
            ui.button('Generate', on_click=generate_image).style('width: 15em')
            image = ui.image().style('width: 60em')

    ui.run(host=host, port=port, reload=False)


class DiffusionServeJuspty(L.LightningWork):

    def __init__(self, *args, flow, **kwargs):
        super().__init__(*args, flow, **kwargs)
        self._flow = flow

    def run(self):
        self._flow.setup()
        webpage(self._flow, host=self.host, port=self.port)
