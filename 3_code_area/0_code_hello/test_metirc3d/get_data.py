'''
本文件将metric3d的模型与自己写的各种工具结合使用, 能快速获得深度分离的图像.

img_path路径为需要处理的图像的路径
main_save_path为存储图像主体信息的路径
background_save_path为存储图像背景信息的路径

修改上述三个路径即可快速运行
'''

import time
import torch
from torch import tensor
from utils import use_metric3d
from utils import engine
from utils import mask_tensor
from utils import mask_origin_image
from utils import tranparent
from utils import isblack


def process_save_train_data_single_depth(img_path_name:str, depth_tensor:torch.Tensor, main_save_path:str, background_save_path:str, lower_bound:int, higher_bound:int):
    """
    一个用于处理内容图像的函数, 用于将输入图像按传入的深度信息进行处理并分离
    
    process_save表示这个函数用于处理并存储图像；train_data表示这个函数用于处理并存储训练数据；single_depth表示该函数无法将多个深度信息融合
    
    img_path_name参数：表示需要处理的图像的路径, 需要给出文件名
    
    depth_tensor参数：表示根据什么深度信息进行处理
    
    main_save_path：表示处理后图像主体内容的保存路径, 无须给出文件名
    
    background_save_path：表示处理后图像背景部分的保存路径, 无须给出文件名
    
    lower_bound与higer_bound参数：程序将对lower_bound与higer_bound之间(左闭右开)的深度进行分层, 其结果一共有higer_bound-lower_bound张图像
    
    """
    times = 1
    while (lower_bound % 1 != 0) or (higher_bound % 1 != 0):
        times *= 10
        lower_bound *= 10
        higher_bound *= 10
    
    # print(lower_bound, higher_bound)
    for i in range(int(lower_bound),int(higher_bound)):
        print(f"processing the {i/times} depth info...")
        
        
        # 处理主体信息
        main_save_path_name = main_save_path + '/' + str(i) + '.png'
        masked_tensor = mask_tensor.get_mask_tensor(depth_tensor=depth_tensor, t=i/times, mode='tensor')
        
        if isblack.isblack(masked_tensor):
            print(f"there is no {i/times} depth")
            continue
        
        mask_origin_image.show_save_masked_img(
            masked_depth = masked_tensor,
            origin_path_name = img_path_name,
            save_path_name= main_save_path_name,
            isshow=False
        )
            
        # 处理背景信息
        background_save_path_name = background_save_path + '/' + str(i) + '.png'
        negative_masked_tensor = mask_tensor.get_negative_mask_tensor(depth_tensor=depth_tensor, t=i/times, mode='tensor')
        mask_origin_image.show_save_masked_img(
            masked_depth=negative_masked_tensor,
            origin_path_name=img_path_name,
            save_path_name=background_save_path_name,
            isshow=False
        )

def process_save_train_data_multi_depth(img_path_name:str, depth_tensor:torch.Tensor, main_save_path:str, background_save_path:str, lower_bound:int, higher_bound:int):
    """
    一个用于处理内容图像的函数, 用于将输入图像按传入的深度信息进行处理并分离
    
    process_save表示这个函数用于处理并存储图像；train_data表示这个函数用于处理并存储训练数据；multi_depth表示该函数可以将多个深度信息当作掩膜
    
    img_path_name参数：表示需要处理的图像的路径, 需要给出文件名
    
    depth_tensor参数：表示根据什么深度信息进行处理
    
    main_save_path：表示处理后图像主体内容的保存路径, 需要给出文件名
    
    background_save_path：表示处理后图像背景部分的保存路径, 需要给出文件名
    
    lower_bound与higer_bound参数：程序将对lower_bound与higher_bound之间(左闭右开)的深度均进行掩膜处理, 其结果将呈现在一张图上, 最终仅有一张图像
    
    """    
    
    # 如果输入的数值是小数, 则将其进行转为整数处理
    times = 1
    while (lower_bound % 1 != 0) or (higher_bound % 1 != 0):
        times *= 10
        lower_bound *= 10
        higher_bound *= 10
    
    # 将深度信息进行整合, 深度为5~25的全部整合为一张图像
    list_t_times = [x for x in range(int(lower_bound),int(higher_bound))]
    
    list_t = [x/times for x in list_t_times]
    
    masked_tensor = mask_tensor.get_range_mask_tensor(depth_tensor=depth_tensor, list_t=list_t, mode='tensor')
    negative_masked_tensor = mask_tensor.get_range_negative_mask_tensor(depth_tensor=depth_tensor, list_t=list_t, mode='tensor')
    
    # 处理主体信息
    main_save_path_name = main_save_path + '/' + str(lower_bound) + '_' + str(higher_bound) +'_main.png'
    mask_origin_image.show_save_masked_img(
        masked_depth = masked_tensor,
        origin_path_name = img_path_name,
        save_path_name=main_save_path_name,
        isshow=False
    )
    
    # 处理背景信息
    background_save_path_name = background_save_path + '/' + str(lower_bound) + '_' + str(higher_bound) +'_main.png'
    mask_origin_image.show_save_masked_img(
        masked_depth=negative_masked_tensor,
        origin_path_name=img_path_name,
        save_path_name=background_save_path_name,
        isshow=False
    )
            
    
if __name__ == '__main__':
    img_path_name = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/02/02.png'
    
    
    depth_tensor = use_metric3d.use_metric3d(rgb_file_path_name=img_path_name) #获取深度预测结果
    # engine.display_save_depth(pred_depth=depth_tensor, path_name='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/02/02_pred_depth.png')    
    
    merged_depth = engine.merge_depth(pred_depth=depth_tensor, t=0)
    
    # engine.write_depth(file_path_name="/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/depth.txt", pred_depth=merged_depth)
    
    lower_bound = torch.min(merged_depth)
    higher_bound = torch.max(merged_depth)+1
    
    process_save_train_data_single_depth(
        img_path_name=img_path_name,
        depth_tensor=merged_depth,
        main_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/03/img_main',
        background_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/03/img_background',
        lower_bound=int(lower_bound),
        higher_bound=int(higher_bound)
        )
    
    # process_save_train_data_multi_depth(
    #     img_path_name=img_path_name,
    #     depth_tensor=merged_depth,
    #     main_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/03/img_main',
    #     background_save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/03/img_background',
    #     lower_bound=0.0,
    #     higher_bound=1.4
    # )