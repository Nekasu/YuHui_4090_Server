# this is a file to test python and torch

from test_metirc3d.utils import mask_tensor
from test_metirc3d.utils import use_metric3d
import torch
from test_metirc3d.utils import mask_tensor
from test_metirc3d.utils import isblack

depth_tensor = use_metric3d.use_metric3d(rgb_file_path_name='/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/02/02.png')

mask:torch.Tensor = mask_tensor.get_mask_tensor(depth_tensor=depth_tensor, t=0, mode='tensor')

print(isblack.isblack(mask))