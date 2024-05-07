# this is a util funciton file for gussian blur(高斯模糊)

from PIL import Image
import numpy as np
from scipy.ndimage.filters import gaussian_filter 

def gaussian_blur(image, sigma):
    # 将图像转换为numpy数组
    img = np.array(image)
    
    # 对图像进行高斯模糊处理
    blurred_img = gaussian_filter(img, sigma=sigma, truncate=2.0)
    
    # 将处理后的图像转换回PIL图像对象
    blurred_image = Image.fromarray(blurred_img.astype('uint8'))
    
    return blurred_image


def main():
    path  = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/firefly.jpg'
    image = Image.open(fp=path)
    blurred_image = gaussian_blur(image, sigma=11.0)
    blurred_image.save('blurred_example.jpg')

if __name__ == "__main__":
    main()
