"""
该文件中包含一个工具函数, 用于完成以下任务：

我有一个深度检测的结果, 其数值类型是一个张量(Tensor). 我将该张量中数值不为4的分量全部改成0, 以达到掩膜的效果, 此时该数据还是一个张量(Tensor), 该文件中的函数实现了将这个带有黑色掩膜张量(Tensor)可视化的操作 

"""

import array
import torch
import numpy as np
import matplotlib.pyplot as plt

def get_mask_tensor(depth_tensor: torch.Tensor, t, mode='numpy'):
    """
    一个用于获取带有掩膜的深度检测张量(Tensor)的函数. 与t相同的数值将被设置为1, 否则被设置为0

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成0, 其余部分修改为1


    返回：masked_array, 一个掩膜numpy数组.
    """
    depth_tensor = depth_tensor.cpu()
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(depth_tensor == t, torch.tensor(1.0), torch.tensor(0.0))

    # print("Original Tensor:")
    # print(depth_tensor)
    # print("Masked Tensor:")
    # print(masked_tensor)
    mode = mode.upper()

    if mode == 'TENSOR':
        return masked_tensor
    elif mode == 'NUMPY':
        masked_array = masked_tensor.numpy()
        return masked_array
    else:
        print("wrong mode error")
    # 将 Tensor 转换为 NumPy 数组


def show_save_mask(depth_tensor: torch.Tensor, t, save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/masked.png'):
    """
    一个用于可视化带有掩膜的深度检测张量(Tensor)的函数.

    depth_tensor参数：需要可视化的深度预测张量(Tensor), 可以在cpu上, 也可以在gpu上;

    t参数: 需要掩膜的数值, 例如, 如果t=4, 则将深度检测张量中数值不为4的分量全部改成0;

    path参数：保存掩膜图像的位置;

    返回：无返回值.
    """
    depth_tensor = depth_tensor.cpu()
    # 将数值不为4的分量全部改成0
    masked_tensor = torch.where(depth_tensor == t, torch.tensor(1.0), torch.tensor(0.0))

    print("Original Tensor:")
    print(depth_tensor)
    print("Masked Tensor:")
    print(masked_tensor)

    # 将 Tensor 转换为 NumPy 数组
    masked_array = masked_tensor.numpy()


    # 使用 Matplotlib 显示带有掩膜的 Tensor
    plt.imshow(masked_array, cmap='gray')
    plt.colorbar()  # 显示颜色条
    plt.title('Masked Tensor Visualization')
    plt.axis('off')  # 关闭坐标轴

    # 保存图像到文件
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    import use_metric3d
    import engine
    
    img_path = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/large_perspective_1.png'
    pred_depth = use_metric3d.use_metric3d(rgb_file_path=img_path)
    merged_tensor = engine.merge_depth(pred_depth=pred_depth, t=0)
    
    show_save_mask(depth_tensor=merged_tensor, t=2)