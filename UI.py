import lightning as L
import torch
import warnings
warnings.simplefilter("ignore")
import logging
from lightning.app.storage import Path
from lightning.app.components.python import TracerPythonScript
from lightning.app.components.serve import ServeGradio
import gradio as gr
from lightning.app.storage import Drive

import gdown

from subprocess import Popen
import os
import os.path

import numpy as np
from PIL import Image
import torchvision.transforms as T
logger = logging.getLogger(__name__)

from lightning.app.storage.path import Path
from lightning.app.utilities.tracer import Tracer
from diffusers import StableDiffusionPipeline


#  gradio UI

class ImageServeGradio(ServeGradio):

    inputs = [gr.inputs.Textbox(label="Prompt"),gr.inputs.Textbox(label="Number of images")]
    outputs = gr.Gallery(label="Generated images", show_label=False, elem_id="gallery"
        ).style(grid=[2], height="auto")

    def __init__(self, cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.examples = None
        self.best_model_path = None


    def run(self,best_model):
        # create some examples
        self.best_model_path= best_model
        print(self.best_model_path)

        super().run()

    def predict(self, prompt, n_samples):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        images = self.model(prompt, num_images_per_prompt=int(n_samples), num_inference_steps=50).images
      
        #  guidance_scale=7.5)
        return images

    def build_model(self):
        # 1. Load the best model
        # if self.best_model_path:
        #     # model = mdl.Unet_pl()
        #     # model.load_from_checkpoint(self.best_model_path,n_channels=1, n_classes=2)
        # else:
        
        print("DONE training ")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = StableDiffusionPipeline.from_pretrained(
        self.best_model_path
        ).to(device)
    
 
        return model

