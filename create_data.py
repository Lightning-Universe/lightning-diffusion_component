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
    def __init__(self):
        super().__init__()
        # diver for the data
        self.drive_1 = Drive("lit://drive_1")

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
            # width, height = img.size
            # if width> height:
            #     img = img.rotate(-90, PIL.Image.NEAREST, expand = 1)

            # save tem olcally
            temp = "my_concept"# 
            if not os.path.exists(temp):
                os.mkdir(temp)
            img.save(f"{i}.jpeg")

            # put the  in drive
            self.drive_1.put(f"{i}.jpeg")

            # delete image
            os.remove(f"{i}.jpeg")

        # number of concept examples
        self.number_samples = len((self.drive_1.list()))
        print("done")
        # remove downladed folder
        # shutil.rmtree("MY_FACE")


