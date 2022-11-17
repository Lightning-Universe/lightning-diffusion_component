import base64

from PIL import Image
from io import BytesIO


def image_decode(image: Image) -> bytes:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())
