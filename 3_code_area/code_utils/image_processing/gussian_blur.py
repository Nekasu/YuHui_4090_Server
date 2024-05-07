# this is a util funciton file for gussian blur(高斯模糊)

from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter 

def gaussian_blur(image, sigma):
    """
    Apply Gaussian blur to a PIL image

    Parameters
    ----------
    image: PIL Image
        Image to be blurred
    sigma: float
        Standard deviation for Gaussian kernel

    Returns
    -------
    PIL Image
        Blurred image
    """
    # Convert PIL image to numpy array
    img = np.array(image)

    # Perform Gaussian blur on image
    blurred_img = gaussian_filter(img, sigma=sigma, truncate=2.0)

    # Convert back to PIL image and return
    blurred_image = Image.fromarray(blurred_img.astype('uint8'))
    return blurred_image


def main():
    path  = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/firefly.jpg'
    image = Image.open(fp=path)
    blurred_image = gaussian_blur(image, sigma=11.0)
    blurred_image.save('blurred_example.jpg')

if __name__ == "__main__":
    main()
