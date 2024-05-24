"""
此程序将根据深度预测信息，将原始图像进行mask操作，得到mask后的图像
"""

from cv2 import merge
from scipy.__config__ import show
import torch
import numpy as np
import matplotlib.pyplot as plt
import mask_tensor
import use_metric3d
import engine

def show_save_masked_img(depth_tensor:torch.Tensor, t, origin_path:str, save_path:str):
    """
    一个用于将深度预测信息生成掩膜, 并应用到原图像上的函数.

    depth_tensor: torch.Tensor, 深度预测信息, 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成0;

    origin_path: str, 原始图像路径;

    save_path: str, 保存路径

    返回值: None, 无返回值
    """
    # 获取掩膜
    mask = mask_tensor.get_mask_tensor(
        depth_tensor=depth_tensor,
        t=t,
        mode='tensor'
    )
    print(mask)
    # 读取图像
    origin_img = plt.imread(origin_path)
    
    # 转换为Tensor类型
    origin_img_tensor = torch.from_numpy(origin_img)
    
    # print(f"size of origin_img_tensor is {origin_img_tensor.size()}")
    # print(f"size of mask is {mask.size()}")
    
    # unsqueeze 在最后一个维度添加一个维度
    expanded_mask = mask.unsqueeze(2)  # 形状 [853, 1280, 1]
    print(expanded_mask.size())
    # 广播 mask 使其与 origin_img_tensor 相同形状
    expanded_mask = expanded_mask.expand(-1, -1, 3)  # 形状 [853, 1280, 3]
    
    
    # 若掩膜中数值为0, 则图像对应位置也设置为0. 
    # 由于掩膜对应区域为0与1, 0代表需要消失的部分, 1表示不变的部分, 所以将原始图像张量(Tensor)与掩膜张量(Tensor)逐元素相乘, 即可得到掩膜处理后的图像
    masked_img_tensor = origin_img_tensor * expanded_mask
    

    masked_img_array = masked_img_tensor.numpy()

     # 使用 Matplotlib 显示带有掩膜的 图像
    plt.imshow(masked_img_array)
    plt.colorbar()  # 显示颜色条
    plt.title('Masked Image')
    plt.axis('off')  # 关闭坐标轴

    # 保存图像到文件
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    
    img_path = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/large_perspective_1.png'
    save_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/'
    depth_tensor= use_metric3d.use_metric3d(rgb_file_path=img_path)
    
    merged = engine.merge_depth(pred_depth=depth_tensor, t=0)
    for i in range(12):
        save_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/'+ str(i) + '.png'
        show_save_masked_img(
            depth_tensor = merged,
            t = i,
            origin_path = img_path,
            save_path = save_path
        )
        a = input("input '1' to continue:")