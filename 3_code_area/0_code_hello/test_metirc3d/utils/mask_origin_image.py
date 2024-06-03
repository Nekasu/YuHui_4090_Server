import torch
import numpy as np
from PIL import Image
import mask_tensor
import use_metric3d
import engine

def show_save_masked_img(masked_depth: torch.Tensor, origin_path: str, save_path: str, isshow=False):
    """
    一个用于将深度预测信息掩膜应用到原图像上的函数.

    masked_depth: torch.Tensor, 掩膜化的深度信息, 由mask_tensor文件中的mask_tensor函数或mask_negative_tensor函数生成;

    origin_path: str, 原始图像路径;

    save_path: str, 保存路径
    
    isshow:bool, 用于判断该函数时候需要显示掩膜化的图像

    返回值: None, 无返回值
    """
    # 读取图像
    origin_img = Image.open(fp=origin_path)
    origin_img_array = np.array(origin_img)

    # 转换为Tensor类型
    origin_img_tensor = torch.from_numpy(origin_img_array).byte()

    # unsqueeze 在最后一个维度添加一个维度
    expanded_mask = masked_depth.unsqueeze(2)  # 形状 [853, 1280, 1]

    # 广播 mask 使其与 origin_img_tensor 相同形状
    expanded_mask = expanded_mask.expand(-1, -1, 3)  # 形状 [853, 1280, 3]

    # 逐元素相乘得到掩膜处理后的图像
    masked_img_tensor = origin_img_tensor * expanded_mask

    # 将结果转换为NumPy数组并转换为uint8类型
    masked_img_array = masked_img_tensor.numpy().astype(np.uint8)
    print(masked_img_array.shape)
    
    masked_img_image = Image.fromarray(masked_img_array)

    if isshow:
        # 使用 PIL.Image 显示带有掩膜的 图像
        masked_img_image.show()

    # 保存图像到文件
    if save_path is not None:
        masked_img_image.save(fp=save_path)

if __name__ == '__main__':
    img_path = '/mnt/sda/zxt/z_datas/imgs/1_origin_data/large_perspective_1.png'
    save_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/img_background/'
    depth_tensor = use_metric3d.use_metric3d(rgb_file_path=img_path)

    merged = engine.merge_depth(pred_depth=depth_tensor, t=0)
    for i in range(12):
        print(f"processing the {i}th depth info...")
        save_name = save_path + 'background_'+ str(i) + '.png'
        masked_tensor = mask_tensor.get_negative_mask_tensor(
            depth_tensor=merged,
            t = i,
            mode = 'tensor'
        )
        show_save_masked_img(
            masked_depth=masked_tensor,
            origin_path=img_path,
            save_path=save_name
        )
        
