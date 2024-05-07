from PIL import Image
import numpy as np
from utils import convert_into_frequency as cif

def show_low_frequency_part(img):
    f = low_pass_filter(img)

    # inverse the fourier transform
    img_back = np.fft.ifft2(f)

    # take the real part of the image
    img_back = np.abs(img_back)

    # show the image
    img_back = Image.fromarray(img_back)
    img_back.show()
    return img_back


def show_high_frequency_part(img):
    f = high_pass_filter(img)

    # shift the zero frequency component back to the corner
    f_ishift = np.fft.ifftshift(f)

    # inverse the fourier transform
    img_back = np.fft.ifft2(f_ishift)

    # take the real part of the image
    img_back = np.abs(img_back)

    # show the image
    img_back = Image.fromarray(img_back)
    img_back.show()
    return img_back


# take the low frequency component
def low_pass_filter(img):
    f = cif.get_frequency_highcenter(img)
    rows, cols = img.size
    crow, ccol = rows//2, cols//2
    f[crow-125:crow+125, ccol-125:ccol+125] = 0

    return f

def high_pass_filter(img):
    f = cif.get_frequency_highcenter(img)
    f = f - low_pass_filter(img)

    return f


def main():
    img = Image.open('../0_images/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg')
    show_low_frequency_part(img)
    show_high_frequency_part(img)

if __name__ == '__main__':
    main()