from PIL import Image
import numpy as np
import convert_into_frequency as cif

def show_low_frequency_part(img):
    """
    Applies low pass filter to the image and shows the result.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Result of the low pass filter
    """
    f = low_pass_filter(img)

    # Inverse the fourier transform
    img_back = np.fft.ifft2(f)

    # Take the real part of the image
    img_back = np.abs(img_back)

    # Show the image
    img_back = Image.fromarray(img_back)
    img_back.show()
    return img_back




def show_high_frequency_part(img):
    """
    Applies high pass filter to the image and shows the result.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    PIL.Image.Image
        Result of the high pass filter
    """
    f = high_pass_filter(img)  # apply high pass filter

    # shift the zero frequency component back to the corner
    f_ishift = np.fft.ifftshift(f)  # shift the fft image for visualization

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
    """
    Applies low pass filter to the image and returns the frequency domain result.

    The low pass filter removes high frequency components around the center of the image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    numpy.ndarray
        Frequency domain result of low pass filter
    """
    f = cif.get_frequency_highcenter(img)  # get the frequency domain of the image
    rows, cols = img.size  # get the size of the image
    crow, ccol = rows//2, cols//2  # calculate the center of the image
    f[crow-125:crow+125, ccol-125:ccol+125] = 0  # set the high frequency components around the center to zero

    return f  # return the frequency domain result


def high_pass_filter(img):
    """
    Applies high pass filter to the image and returns the frequency domain result.

    The high pass filter removes low frequency components around the center of the image.

    Parameters
    ----------
    img : PIL.Image.Image
        Input image

    Returns
    -------
    numpy.ndarray
        Frequency domain result of high pass filter
    """
    f = cif.get_frequency_highcenter(img)  # get the frequency domain of the image
    f = f - low_pass_filter(img)  # remove low frequency components around the center

    return f  # return the frequency domain result



def main():
    img = Image.open('/mnt/sda/zxt/z_datas/imgs/1_origin_data/01_cameraman.png')
    show_low_frequency_part(img)
    show_high_frequency_part(img)

if __name__ == '__main__':
    main()