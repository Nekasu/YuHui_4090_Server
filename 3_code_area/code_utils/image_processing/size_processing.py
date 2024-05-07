from PIL import Image
import numpy as np

def get_a_lower_scale(img):
    """
    Resize the input image to a lower scale by half.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        The image resized to lower scale
    """
    img = img.resize((img.size[0] // 2, img.size[1] // 2))
    return img


