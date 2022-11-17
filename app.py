from dreambooth import DreamBoothFineTuner
import lightning as L

app = L.LightningApp(
    DreamBoothFineTuner(
        image_urls = [
            "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
            "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
            "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
            "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
            ## You can change or add additional images here
        ],
        prompt="a photo of sks toy", # `sks` is the special name
        preservation_prompt="a photo of a cat clay toy",
        learning_rate=5e-06,
        max_steps=50,
        cloud_compute=L.CloudCompute("gpu-fast"),
    )
)