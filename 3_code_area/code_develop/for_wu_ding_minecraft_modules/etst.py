# %% [markdown]
# # 导入包

# %%
from matplotlib.font_manager import weight_dict
from click import style
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import numbers
import math
import cv2
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from torchvision import utils as vutils
import time

# %% [markdown]
# # 计时器设置

# %%
t0 = time.time()

# %% [markdown]
# # 导入预设模型

# %%
model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).cuda()
batch_size = 1

for params in model.parameters():
    params.requires_grad = False

model.eval()

# %% [markdown]
# # 标准化参数

# %%
mu = torch.Tensor([0.485, 0.456, 0.406]).unsqueeze(dim=-1).unsqueeze(-1).cuda()
print(mu.size())

std = torch.Tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1).cuda()
unnormalize = lambda x: x*std + mu
normalize = lambda x: (x-mu)/std

# %% [markdown]
# # 图像上采样模块

# %%
# 函数定义, 将32*32的风格图像放大到512*512, 原图中的每个像素向右下扩散16*16个像素
def super_resolution_to_512(img):
    img = img.convert('RGB')
    img_torch = ToTensor()(img)
    # print(img_torch.size())
    img_512 = torch.zeros((3, 512, 512))
    # print(img_512.size())
    for i in range(0,img_torch.size()[0],):
        for j in range(0, img_torch.size()[1]):
            for k in range(0, img_torch.size()[2]):
                img_512[i, 16*j:16*(j+1), 16*k:16*(k+1)] = img_torch[i, j, k]
    # print(img_512.size())
    return img_512    

# %%
# 上述函数的测试
if __name__ == '__main__':
    style_img_path = '/home/zxt/Python_area/for_wu_ding_minecraft_modules/datas/1_origin_data/bamboo_large_leaves.png'
    style_img = Image.open(style_img_path)
    img = super_resolution_to_512(style_img)
    vutils.save_image(img, 'test.png')

# %%
transform_test = Compose([
    Resize(size=(512,512)),
    ToTensor(),
])

# %% [markdown]
# # 输入内容图像与风格图像的路径

# %%
content_img_path = '/home/zxt/Python_area/for_wu_ding_minecraft_modules/datas/2_preprocessed_data/preprocessed_tomatoes_edge.png'
style_img_path = '/home/zxt/Python_area/for_wu_ding_minecraft_modules/datas/1_origin_data/bamboo_large_leaves.png'

# %% [markdown]
# ## 处理内容图像与风格图像

# %%
content_img = Image.open(content_img_path)
content_img = content_img.convert('RGB')
image_size = content_img.size
content_img = transform_test(content_img)
content_img = content_img.squeeze(0).cuda()
print(f"内容图像的维度是：{content_img.size()}")

style_img = Image.open(fp=style_img_path)
style_img = style_img.convert('RGB')
# style_img = transform_test(style_img)
style_img = super_resolution_to_512(style_img)
# print(style_img.size())
style_img = style_img.squeeze(0).cuda()
print(f"风格图像的维度是：{style_img.size()}")

var_img = content_img.clone()
#var_img = torch.rand_like(content_img)
var_img.requires_grad=True

# %%
class ShuntModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.module = model.features.cuda().eval()
        self.con_layers = [22]
        self.sty_layers = [1,6,11,20,29]
        for name, layer in self.module.named_children():
            if isinstance(layer, nn.MaxPool2d):
                self.module[int(name)] = nn.AvgPool2d(kernel_size = 2, stride = 2)

    def forward(self, tensor: torch.Tensor) -> dict:
        sty_feat_maps = []; con_feat_maps = [];
        x = normalize(tensor)
        for name, layer in self.module.named_children():
            x = layer(x);
            if int(name) in self.con_layers: con_feat_maps.append(x)
            if int(name) in self.sty_layers: sty_feat_maps.append(x)
        return {"Con_features": con_feat_maps, "Sty_features": sty_feat_maps}

# %%
model = ShuntModel(model)
sty_target = model(style_img)["Sty_features"]
con_target = model(content_img)["Con_features"]
gram_target = []

# %%
print(type(sty_target))
print(len(sty_target))
print(sty_target[0].size())

# %%
for i in range(len(sty_target)):
    c, h, w  = sty_target[i].size()
    tensor_ = sty_target[i].view(1 * c, h * w)
    gram_i = torch.mm(tensor_, tensor_.t()).div(1*c*h*w)
    gram_target.append(gram_i)

optimizer = torch.optim.Adam([var_img], lr = 0.01, betas = (0.9,0.999), eps = 1e-8)
lam1 = 1e-3; lam2 = 1e7; lam3 = 5e-3

# %%
for itera in range(20001):
    optimizer.zero_grad()
    output = model(var_img)
    sty_output = output["Sty_features"]
    con_output = output["Con_features"]
    
    con_loss = torch.tensor([0]).cuda().float()
    for i in range(len(con_output)):
        con_loss = con_loss + F.mse_loss(con_output[i], con_target[i])
    
    sty_loss = torch.tensor([0]).cuda().float()
    for i in range(len(sty_output)):
        c, h, w  = sty_output[i].size()
        tensor_ = sty_output[i].view(1 * c, h * w)
        gram_i = torch.mm(tensor_, tensor_.t()).div(1*c*h*w)
        sty_loss = sty_loss + F.mse_loss(gram_i, gram_target[i])
    
    c, h, w  = style_img.size()
    TV_loss = (torch.sum(torch.abs(style_img[ :, :, :-1] - style_img[ :, :, 1:])) +
                torch.sum(torch.abs(style_img[ :, :-1, :] - style_img[ :, 1:, :])))/(1*c*h*w)
    
    loss = con_loss * lam1 + sty_loss * lam2 + TV_loss * lam3
    loss.backward()
    var_img.data.clamp_(0, 1)
    optimizer.step()
    if itera%100==0:
        print('itera: %d, con_loss: %.4f, sty_loss: %.4f, TV_loss: %.4f'%(itera,
              con_loss.item()*lam1,sty_loss.item()*lam2,TV_loss.item()*lam3),'\n\t total loss:',loss.item())
        print('var_img mean:%.4f, std:%.4f'%(var_img.mean().item(),var_img.std().item()))
        print('time: %.2f seconds'%(time.time()-t0))
    if itera%1000==0:    
        save_img = var_img.clone()
        save_img = torch.clamp(save_img,0,1)
        # save_img = save_img[0].permute(1,2,0).data.cpu().numpy()*255
        vutils.save_image(img, '/home/zxt/Python_area/for_wu_ding_minecraft_modules/datas/999_output1/transfer%d.jpg'%itera)




