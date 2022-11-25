#!/usr/bin/env python3
import asyncio
import functools
from typing import Callable, Any
from nicegui import ui
import base64
from PIL import Image
from io import BytesIO

async def io_bound(callback: Callable, *args: Any, **kwargs: Any):
    '''Makes a blocking function awaitable; pass function as first parameter and its arguments as the rest'''
    return await asyncio.get_event_loop().run_in_executor(None, functools.partial(callback, *args, **kwargs))


class Flow:

    def predict(self, prompt):
        with open("data/preservation_images/0.jpg", "rb") as image_file:
            imgstr = base64.b64encode(image_file.read()).decode("UTF-8")
        return {"image": f"data:image/png;base64,{imgstr}"}


flow = Flow()


def webpage(host = "127.0.0.0", port=8080):

    async def generate_image():
        image.source = 'https://dummyimage.com/600x400/ccc/000000.png&text=building+image...'
        prediction = await io_bound(flow.predict, prompt=prompt.value)
        image.source = prediction['image']

    # User Interface
    with ui.row().style('gap:10em'):
        with ui.column():
            ui.label('Stable Diffusion (image generator)').classes('text-2xl')
            prompt = ui.input('prompt').style('width: 20em')
            ui.button('Generate', on_click=generate_image).style('width: 15em')
            image = ui.image().style('width: 60em')

    ui.run()


webpage()