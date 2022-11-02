
import torch
from diffusers import StableDiffusionPipeline
from tqdm.auto import tqdm
from datasets import PromptDataset
from pathlib import Path
import gc
import lightning as L
from lightning.app.storage import Drive
import os
from lightning.app.storage import Payload


def get_filenames_of_path(path_s):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [path_s+"/"+path for path in os.listdir(path_s)]
        return filenames



class Prior(L.LightningWork):  
    def __init__(self,drive,cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        self.drive_1 = drive
        self.drive_2 = Drive("lit://drive_2")

    def run(self, class_prompt, YOUR_TOKEN="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",sample_batch_size = 2): 
        
        # type of devie
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # number of examples
        num_class_images =  len((self.drive_1.list()))
        cur_class_images = len((self.drive_2.list()))

        # create examples prior
        if cur_class_images < num_class_images:
            ## This need to change from hugging face
            pipeline = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4",use_auth_token=YOUR_TOKEN
            ).to(device)

            # number of examples to generate
            num_new_images = num_class_images - cur_class_images
            print(f"Number of class images to sample: {num_new_images}.")



            sample_dataset = PromptDataset(class_prompt, num_new_images)
            # this is to get the description of the images
            sample_dataloader = torch.utils.data.DataLoader(sample_dataset, batch_size=sample_batch_size)

            # generate the examples
            for example in tqdm(sample_dataloader, desc="Generating class images"):
                # example genration
                print("creating classes")
                images = pipeline(example["prompt"])
                # get image
                images = images["sample"]
                
                # save images
                for i, image in enumerate(images):
                    image.save( f"{example['index'][i] + cur_class_images}.jpg")
                    print(f"{example['index'][i] + cur_class_images}.jpg")
                    self.drive_2.put(f"{example['index'][i] + cur_class_images}.jpg")


            pipeline = None
            gc.collect()
            del pipeline
            with torch.no_grad():
                torch.cuda.empty_cache()






      