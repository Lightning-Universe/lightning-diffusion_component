# Lightning Diffusion

Lightning Diffusion provides components to finetune and serve diffusion model on [lightning.ai](https://lightning.ai/).


```python
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
        max_train_steps=450,
        cloud_compute=L.CloudCompute("gpu-fast"),
    )
)
```

__________
# Usage:
```
# Run locally
lightning run app app.py

# Run in the cloud
lightning run app app.py --cloud
```
