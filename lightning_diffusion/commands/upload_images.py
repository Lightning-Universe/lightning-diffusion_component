from lightning.app.utilities.commands import ClientCommand
import argparse
import os
import sys

class UploadImagesClientCommand(ClientCommand):

    def run(self):
        parser = argparse.ArgumentParser(description="Parse to upload the images.")
        parser.add_argument("--folder", type=str, help="The image folder to be uploaded")
        hparams = parser.parse_args()

        if not os.path.isdir(hparams.folder):
            print(f"The provided {hparams.folder} needs to be a folder with images to upload.")
            sys.exit(0)
        elif len(os.listdir(hparams.folder)) == 0:
            print(f"The provided {hparams.folder} is empty.")
            sys.exit(0)