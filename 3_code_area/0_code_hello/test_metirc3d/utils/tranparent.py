from PIL import Image
import numpy as np
import torch

def make_black_transparent(img_path: str, save_path: str):
    """
    将PNG图像中的黑色部分变成透明。

    img_path: str, 输入图像路径
    
    save_path: str, 输出图像保存路径
    """
    # 读取图像并转换为RGBA
    image = Image.open(img_path).convert("RGBA")
    img_array = np.array(image)

    # 将图像转换为Tensor
    img_tensor = torch.from_numpy(img_array)

    # 获取RGB通道
    rgb_tensor = img_tensor[:, :, :3]
    
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
    img_path = './outputs/4.png'
    save_path = './outputs/4_transparent.png'
    make_black_transparent(img_path, save_path)
