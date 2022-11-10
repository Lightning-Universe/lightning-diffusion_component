import lightning as L
#from quick_start.components import  ImageServeGradio, PLTracerPythonScript

import warnings
warnings.simplefilter("ignore")
from create_data import Create_Data
import os
from Prior_Preservation import Prior
from argparse import Namespace
from models import Training
from UI import ImageServeGradio


class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()

        # work that gets the data
        self.create_data = Create_Data(cloud_compute=L.CloudCompute("cpu",idle_timeout=30))

        # work to generate prior
        self.prior =  Prior(cloud_compute=L.CloudCompute("gpu",disk_size=30,idle_timeout=30))

         # work that trains my model
        self.train_work = Training(cloud_compute=L.CloudCompute("cpu-medium",
        disk_size=30))

        # UI 
        self.UI =  ImageServeGradio(L.CloudCompute("gpu",disk_size=30))

    def run(self):
         # Download pictures of my concept
        url = "https://drive.google.com/drive/folders/1PzdEZ0u87yxoy5cAMk12tJe0gE2LtYUA?usp=sharing"
        self.create_data.run(url=url)

        # Create previous  examples
        self.prior.run(data_dir=self.create_data.data_dir,class_prompt="a photo of a Daniela person",)

    # arguments for training
        args = Namespace(
            # Model Download Folder
                        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" ,
            # image Resolution
                        resolution=450,
            # Image transformation
                        center_crop=True,
            # My concept text prompt
                        instance_prompt="a photo of Daniela person" ,
            # learning rate
                        learning_rate=5e-06,
            # batch size
                        train_batch_size=1,
            # use 8bit optimizer from bitsandbytes
                        use_8bit_adam=True, 
            # Prior preservation
                        with_prior_preservation=True, 
            # Prior weight
                        prior_loss_weight=1,
            # prior text prompt
                        class_prompt="a photo of a sks person", 
            # folder to store re-trained mode;
                        output_dir="dreambooth-concept",
            # num epochs
                        num_epochs = 1)


       # run training
        self.train_work.run(self.create_data.data_dir,self.prior.prior_dir,args)
    
        # run UI
        self.UI.run(self.train_work.checkpoint_dir)
        
    def configure_layout(self):
        tab_1 = {"name": "Image Generator", "content": self.UI}
        return [tab_1]

app = L.LightningApp(TrainDeploy())

