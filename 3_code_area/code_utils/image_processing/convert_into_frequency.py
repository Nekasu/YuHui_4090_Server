# show frequency domain of the image
import re
from PIL import Image
import numpy as np

def show_frequency_highcenter(img):
    '''
    show the frequency domain of the image
    '''
    # convert the image into gray scale
    img_gray = img.convert('L')

    # change the image into frequency domain
    f = np.fft.fft2(img_gray)

    # show the frequency domain image
    magnitude_spectrum = 20*np.log(np.abs(f))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    magnitude_spectrum = Image.fromarray(magnitude_spectrum)
    magnitude_spectrum.show()
    return magnitude_spectrum

def show_frequency_lowcenter(img):
    '''
    show the frequency domain of the image, and the low frequency component is in the center
    '''
    img_gray = img.convert('L')
    f = np.fft.fft2(img_gray)
    f = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(f))
    magnitude_spectrum = np.asarray(magnitude_spectrum, dtype=np.uint8)
    magnitude_spectrum = Image.fromarray(magnitude_spectrum)
    magnitude_spectrum.show()
    return magnitude_spectrum

def get_frequency_highcenter(img):
    '''
    a fourier transform that convert the image into frequency domain, and the high frequency component is around the center
    '''
    img_gray = img.convert('L')
    f = np.fft.fft2(img_gray)
    return f

def get_shifted_frequency_lowcenter(img):
    '''
    a fourier shift that shift the zero frequency component to the center
    '''
    img_gray = img.convert('L')
    f = np.fft.fftshift(img_gray)
    return f