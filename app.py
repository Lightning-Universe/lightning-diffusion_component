import lightning as L
#from quick_start.components import  ImageServeGradio, PLTracerPythonScript

import warnings
warnings.simplefilter("ignore")
from create_data import Create_Data
import os
from Prior_Preservation import Prior
from argparse import Namespace
from models import Training


class TrainDeploy(L.LightningFlow):
    def __init__(self):
        super().__init__()

        # work that gets the data
        self.create_data = Create_Data(cloud_compute=L.CloudCompute("cpu",disk_size=30))

        # work to generate prior
        self.prior =  Prior(self.create_data.drive_1, cloud_compute=L.CloudCompute("gpu",disk_size=60))

         # work that trains my model
        self.train_work = Training(cloud_compute=L.CloudCompute("cpu",disk_size=60))

    def run(self):
         # Download pictures of my concept
        url = "https://drive.google.com/drive/folders/1PzdEZ0u87yxoy5cAMk12tJe0gE2LtYUA?usp=sharing"
        self.create_data.run(url=url)

        # Create previous  examples
        self.prior.run("photo of a Daniela person ultra detailed")

        
    # arguments for training
        args = Namespace(
            # Model Download Folder
                        pretrained_model_name_or_path="CompVis/stable-diffusion-v1-4" ,
            # image Resolution
                        resolution=512,
            # Image transformation
                        center_crop=True,
            # Folder of my concept
                        instance_data_dir= self.create_data.drive_1,
            # My concept text prompt
                        instance_prompt="photo of a Daniela person ultra detailed",
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
            # Prior folder
                        class_data_dir=self.prior.drive_2, 
            # prior text prompt
                        class_prompt="photo of a sks person ultra detailed", 
            # number of examples
                        num_class_images=self.create_data.number_samples, 
            # folder to store re-trained mode;
                        output_dir="dreambooth-concept",
            # num epochs
                        num_epochs = 1)


        # run training
        self.train_work.run(args)
        
    # def configure_layout(self):
    #     tab_1 = {"name": "Model training", "content": self.train_work}
    #     return [tab_1]#, tab_2]

app = L.LightningApp(TrainDeploy())

