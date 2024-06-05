import torch
from . import mask_tensor
from . import use_metric3d

def isblack(depth_tensor: torch.Tensor) -> bool:
    # 直接返回布尔值
    return torch.all(depth_tensor == 0).item()

def main():
    depth_tensor = use_metric3d.use_metric3d(
        rgb_file_path_name='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/02/02.png'
    )

    mask: torch.Tensor = mask_tensor.get_mask_tensor(depth_tensor=depth_tensor, t=0, mode='tensor')

    print(isblack(mask))

if __name__ == '__main__':
    main()
