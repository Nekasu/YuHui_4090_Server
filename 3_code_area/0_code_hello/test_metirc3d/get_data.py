'''
本文件将metric3d的模型与自己写的各种工具结合使用, 能快速获得深度分离的图像.

img_path路径为需要处理的图像的路径
main_save_path为存储图像主体信息的路径
background_save_path为存储图像背景信息的路径

修改上述三个路径即可快速运行
'''


from torch import tensor
from utils import use_metric3d
from utils import engine
from utils import mask_tensor
from utils import mask_origin_image
from utils import tranparent


def main():
    img_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/01_cameraman.png'
    depth_tensor = use_metric3d.use_metric3d(rgb_file_path=img_path)
    # engine.display_save_depth(pred_depth=depth_tensor, path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/01_cameraman_pred_depth.png')
    # engine.write_depth(file_path="/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/depth.txt", pred_depth=depth_tensor)
    
    merged_depth = engine.merge_depth(pred_depth=depth_tensor, t=0)
    # engine.display_save_depth(pred_depth=merged_depth, path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/01_cameraman_pred_depth_merged0.png')
    # engine.write_depth(file_path="/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/depth_merged.txt", pred_depth=merged_depth)
    
    
    main_save_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_main'
    background_save_path = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_background'
    
    
    for i in range(1,13):
        print(f"processing the {i}th depth info...")
        
        # 处理主体信息
        main_save_name = main_save_path + '/' + 'main_cameraman' + str(i) + '.png'
        masked_tensor = mask_tensor.get_mask_tensor(depth_tensor=merged_depth, t=i, mode='tensor')
        mask_origin_image.show_save_masked_img(
            masked_depth = masked_tensor,
            origin_path = img_path,
            save_path= main_save_name,
            isshow=False
        )
        
        # 处理背景信息
        background_save_name = background_save_path + '/' + 'background_cameraman' + str(i) + '.png'
        negative_masked_tensor = mask_tensor.get_negative_mask_tensor(depth_tensor=merged_depth, t=i, mode='tensor')
        mask_origin_image.show_save_masked_img(
            masked_depth=negative_masked_tensor,
            origin_path=img_path,
            save_path=background_save_name,
            isshow=False
        )
        
    # 将深度信息进行整合, 深度为5~25的全部整合为一张图像
    list_t = [x for x in range(5,26)]
    
    masked_tensor = mask_tensor.get_range_mask_tensor(depth_tensor=merged_depth, list_t=list_t, mode='tensor')
    negative_masked_tensor = mask_tensor.get_range_negative_mask_tensor(depth_tensor=merged_depth, list_t=list_t, mode='tensor')
        
    mask_origin_image.show_save_masked_img(
        masked_depth = masked_tensor,
        origin_path = img_path,
        save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_main/main_cameraman_5_25.png',
        isshow=False
    )
    
    mask_origin_image.show_save_masked_img(
        masked_depth=negative_masked_tensor,
        origin_path=img_path,
        save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_background/background_cameraman5_25.png',
        isshow=False
    )
    
    # 将整合后的图像透明化
    tranparent.make_black_transparent(
        img_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_main/main_cameraman_5_25.png',
        save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_main/main_cameraman_5_25_transparent.png'
    )
        
    tranparent.make_black_transparent(
        img_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_background/background_cameraman5_25.png',
        save_path='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/cameraman/img_background/background_cameraman5_25_transparent.png'
    )
            
    
if __name__ == '__main__':
    main()