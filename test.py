from model_cloud.cloud_api import download_from_lightning_cloud
import zipfile
download_from_lightning_cloud("daniela/stable_diffusion", version="latest",output_dir="model") 
with zipfile.ZipFile("model/checkpoint.zip", 'r') as zip_ref:
    zip_ref.extractall("")