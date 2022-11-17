from PIL import Image
import numpy

from safety_checker import DefaultSafetyFilter


def test_safety_filter_random_images():
    """Test that random images are not treated as NSFW.

    Note: There are images that can be treated as NSFW technically,
    but the probability should be so low that this test case always passes.
    """
    num_images = 10
    imgs = []
    for _ in range(num_images):
        imarray = numpy.random.rand(512, 512, 3) * 255
        im = Image.fromarray(imarray.astype("uint8")).convert("RGB")
        imgs.append(im)
    safety_filter = DefaultSafetyFilter()
    results = safety_filter(imgs)
    assert not any(results)


# def test_safety_filter_natural_images():
#     """Test that natural images (i.e., non-NSFW images) are treated as NSFW."""
#     img_urls = [
#         "https://huggingface.co/datasets/valhalla/images/resolve/main/2.jpeg",
#         "https://huggingface.co/datasets/valhalla/images/resolve/main/3.jpeg",
#         "https://huggingface.co/datasets/valhalla/images/resolve/main/5.jpeg",
#         "https://huggingface.co/datasets/valhalla/images/resolve/main/6.jpeg",
#     ]
#     imgs = []
#     im = Image.open("2.jpeg").convert("RGB")
#     imgs.append(im)
#     safety_filter = DefaultSafetyFilter()
#     assert not any(safety_filter(imgs))
