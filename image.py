import base64
from io import BytesIO
from typing import List

def encode_to_base64(images: List) -> bytes:
    assert len(images) == 1
    image = images[0]
    image.save("image-ref.png")
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue())