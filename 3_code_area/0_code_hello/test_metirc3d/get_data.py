'''
本文件将metric3d的模型与自己写的各种工具结合使用, 能快速获得深度分离的图像.

img_path路径为需要处理的图像的路径
main_save_path为存储图像主体信息的路径
background_save_path为存储图像背景信息的路径

修改上述三个路径即可快速运行
'''

import torch
from torch import tensor
from utils import use_metric3d
from utils import engine
from utils import mask_tensor
from utils import mask_origin_image
from utils import tranparent


def process_save_train_data_single_depth(img_path:str, depth_tensor:torch.Tensor, main_save_path:str, background_save_path:str, lower_bound:int, higher_bound:int):
    """
    一个用于处理内容图像的函数, 用于将输入图像按传入的深度信息进行处理并分离
    
    process_save表示这个函数用于处理并存储图像；train_data表示这个函数用于处理并存储训练数据；single_depth表示该函数无法将多个深度信息融合
    
    img_path参数：表示需要处理的图像的路径
    
    depth_tensor参数：表示根据什么深度信息进行处理
    
    main_save_path：表示处理后图像主体内容的保存路径
    
    background_save_path：表示处理后图像背景部分的保存路径
    
    lower_bound与higer_bound参数：程序将对lower_bound与higer_bound之间(左闭右开)的深度进行分层, 其结果一共有higer_bound-lower_bound张图像
    
    """    
    for i in range(lower_bound,higher_bound):
        print(f"processing the {i}th depth info...")
        
        # 处理主体信息
        masked_tensor = mask_tensor.get_mask_tensor(depth_tensor=depth_tensor, t=i, mode='tensor')
        mask_origin_image.show_save_masked_img(
            masked_depth = masked_tensor,
            origin_path = img_path,
            save_path= main_save_path,
            isshow=False
        )
            
        # 处理背景信息
        negative_masked_tensor = mask_tensor.get_negative_mask_tensor(depth_tensor=depth_tensor, t=i, mode='tensor')
        mask_origin_image.show_save_masked_img(
            masked_depth=negative_masked_tensor,
            origin_path=img_path,
            save_path=background_save_path,
            isshow=False
        )

def process_save_train_data_multi_depth(img_path:str, depth_tensor:torch.Tensor, main_save_path:str, background_save_path:str, lower_bound:int, higher_bound:int):
    """
    一个用于处理内容图像的函数, 用于将输入图像按传入的深度信息进行处理并分离
    
    process_save表示这个函数用于处理并存储图像；train_data表示这个函数用于处理并存储训练数据；multi_depth表示该函数可以将多个深度信息当作掩膜
    
    img_path参数：表示需要处理的图像的路径
    
    depth_tensor参数：表示根据什么深度信息进行处理
    
    main_save_path：表示处理后图像主体内容的保存路径
    
    background_save_path：表示处理后图像背景部分的保存路径
    
    lower_bound与higer_bound参数：程序将对lower_bound与higher_bound之间(左闭右开)的深度均进行掩膜处理, 其结果将呈现在一张图上, 最终仅有一张图像
    
    """    
    # 将深度信息进行整合, 深度为5~25的全部整合为一张图像
    list_t = [x for x in range(lower_bound,higher_bound)]
    
    masked_tensor = mask_tensor.get_range_mask_tensor(depth_tensor=depth_tensor, list_t=list_t, mode='tensor')
    negative_masked_tensor = mask_tensor.get_range_negative_mask_tensor(depth_tensor=depth_tensor, list_t=list_t, mode='tensor')
    
    # 处理主体信息
    mask_origin_image.show_save_masked_img(
        masked_depth = masked_tensor,
        origin_path = img_path,
        save_path=main_save_path,
        isshow=False
    )
    
    # 处理背景信息
    mask_origin_image.show_save_masked_img(
        masked_depth=negative_masked_tensor,
        origin_path=img_path,
        save_path=background_save_path,
        isshow=False
    )
            
    
if __name__ == '__main__':
    img_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/01_cameraman.png'
    
    
    depth_tensor = use_metric3d.use_metric3d(rgb_file_path=img_path) #获取深度预测结果
    # engine.display_save_depth(pred_depth=depth_tensor, path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/01_cameraman_pred_depth.png')
    # engine.write_depth(file_path="/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/depth.txt", pred_depth=depth_tensor)
    
    process_save_train_data_single_depth(
        img_path=img_path,
        main_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_main',
        background_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_background',
        t=0,
        lower_bound=1,
        higher_bound=13
        )
    