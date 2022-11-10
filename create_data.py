import gdown
import os
from PIL import Image
import PIL
import lightning as L
from lightning.app.storage import Drive

# import shutil


def get_filenames_of_path(path_s):
        """Returns a list of files in a directory/path. Uses pathlib."""
        filenames = [path_s+"/"+path for path in os.listdir(path_s)]
        return filenames

class Create_Data(L.LightningWork):  
    def __init__(self,cloud_compute, *args, **kwargs):
        super().__init__(*args, cloud_compute=cloud_compute, **kwargs)
        # diver for the data
        self.data_dir = "lit://data_dir"
        os.makedirs(self.data_dir, exist_ok=True)

        # numebr of concept examples
        self.number_samples = 0

    def run(self, url:str,reshape: bool =  False): 

        

        # download from the cloud
        print("\n Downloading data \n")
        gdown.download_folder(url, quiet=True)  
        images = get_filenames_of_path("MY_FACE")
        
        # Read them for and save them to drive
        for i, image in enumerate(images):
            img = Image.open(str(image))
            img = img.resize([450,450])
            img.save(f"{self.data_dir}/{i}.jpeg")


