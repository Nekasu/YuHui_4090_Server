from PIL import Image
import numpy as np

def get_a_lower_scale(img):
    img = img.resize((img.size[0]//2, img.size[1]//2))
    return img