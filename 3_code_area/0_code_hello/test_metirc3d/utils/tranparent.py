from PIL import Image
import numpy as np
import torch
import os

def make_black_transparent(img_path_name: str, save_path_name=None):
    """
    将PNG图像中的黑色部分变成透明。

    img_path_name: str, 输入图像路径, 需要包含文件名
    
    save_path_name: str, 输出图像保存路径, 如果不填则不会保存图像, 一般用于测试该函数, 需要包含文件名
    """
    # 读取图像并转换为RGBA
    image = Image.open(img_path_name).convert("RGBA")
    img_array = np.array(image)

    # 将图像转换为Tensor
    img_tensor = torch.from_numpy(img_array)

    # 获取RGB通道
    rgb_tensor = img_tensor[:, :, :3] # 最后一项 :3, 表示从下标0开始切片, 到下标3结束, 其中不包括下标为3的部分(仅有0,1,2)
    print(rgb_tensor.size())
    
    # 创建alpha通道：黑色部分透明，其他部分不透明
    alpha_channel = torch.ones_like(img_tensor[:, :, 3]) * 255
    alpha_channel[(rgb_tensor == 0).all(dim=2)] = 0

    # 组合RGB和alpha通道
    img_with_alpha = torch.cat((rgb_tensor, alpha_channel.unsqueeze(2)), dim=2)

    # 将结果转换为NumPy数组
    img_with_alpha_np = img_with_alpha.numpy().astype(np.uint8)

    if save_path_name != None:
        # 使用PIL保存带有alpha通道的图像
        result_image = Image.fromarray(img_with_alpha_np, "RGBA")
        result_image.save(save_path, "PNG")
    
def make_transparent_black(img_path: str, save_path: str):
    """
    将PNG图像中的透明部分变成黑色。

    img_path: str, 输入图像路径
    
    save_path: str, 输出图像保存路径
    """
    # 读取图像并转换为RGBA
    image = Image.open(img_path).convert("RGBA")
    img_array = np.array(image)

    # 将图像转换为Tensor
    img_tensor = torch.from_numpy(img_array)

    # 获取RGB通道
    rgb_tensor = img_tensor[:, :, :3]   # 最后一项 :3, 表示从下标0开始切片, 到下标3结束, 其中不包括下标为3的部分(仅有0,1,2)
    
    # 创建alpha通道：黑色部分透明，其他部分不透明
    alpha_channel = torch.ones_like(img_tensor[:, :, 3]) * 255
    alpha_channel[(rgb_tensor == 0).all(dim=2)] = 0

    # 组合RGB和alpha通道
    img_with_alpha = torch.cat((rgb_tensor, alpha_channel.unsqueeze(2)), dim=2)

    # 将结果转换为NumPy数组
    img_with_alpha_np = img_with_alpha.numpy().astype(np.uint8)

    # 使用PIL保存带有alpha通道的图像
    result_image = Image.fromarray(img_with_alpha_np, "RGBA")
    result_image.save(save_path, "PNG")

# 示例使用
if __name__ == '__main__':
    # 需要处理的图像所在的文件夹
    input_dir = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/02'
    # 获取文件夹中的内容
    all_items = os.listdir(input_dir)[2:]
    print(all_items)
    
    # 处理完毕后将图像保存的路径
    output_dir = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/transparent_outputs/img_background'
    
    for img_name in all_items:
        img_path = input_dir + '/' + img_name
        save_path = output_dir + '/' + 'transparent_' + img_name
        make_black_transparent(img_path)
