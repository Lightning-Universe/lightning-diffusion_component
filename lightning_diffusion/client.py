import base64
import io

import requests
from PIL import Image

response = requests.post(
    "http://127.0.0.1:7777/predict",
    json={"prompt": "A photo of a person", "quality": "low"},
)
image = response.json()["image"]

img = base64.b64decode(image.encode("utf-8"))
img = Image.open(io.BytesIO(img))
img.save("image.png")
