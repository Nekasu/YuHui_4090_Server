# 数据集导入类的文件, 其目的在于将风格图像与内容图像分别导入, 并确保两者的数据量一致

from typing import Dict
from path import Path
import glob
# import torch
import torch.nn as nn
# import pandas as pd
# import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, Compose, Resize, Normalize, RandomCrop
import random

Image.MAX_IMAGE_PIXELS = 1000000000

class DataSplit(nn.Module):
    """
    本文件定义了一个DataSplit类. 
    用于从文件夹中获取数据, 并确保 风格图像数据量 与 内容图像数据量保持一致.
    在两者数量不同时, 调整 风格图像数据量, 使其与 内容图像数据量保持一致
    get_data函数是一个内部函数, 不应该在外部调用这个函数
    魔法函数__init__为构造函数;
    魔法函数__len__已定义；
    魔法函数__getitem__已定义；
    
    构造函数参数列表中的config是“cConfig.py”中的Config类的实例
    """
    def __init__(self, config, phase='train'):
        super(DataSplit, self).__init__()

        self.transform = Compose([Resize(size=[config.load_size, config.load_size]),
                                # RandomCrop用于随机剪裁, 剪裁的大小由传入的size参数确定
                                RandomCrop(size=(config.crop_size, config.crop_size)),
                                ToTensor(),
                                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

        if phase == 'train':
            # Content image data
            img_dir = Path(config.content_dir+'/train') # 从COCO数据集的路径中找到trian文件夹
            self.images = self.get_data(img_dir)  # 利用self.get_data函数获取文件夹中所有['*.jpg', '*.png', '*.jpeg', '*.tif']文件, 返回包含所有图像文件名的列表
            if config.data_num < len(self.images): # 如果设定的数据量小于搜索到的图像数据
                self.images = random.sample(population=self.images, k=config.data_num) # 则进行随机采样, 从列表self.images这个列表中获取config.data_num个名称

            # Style image data
            sty_dir = Path(config.style_dir+'/train') # 从WikiArt数据集的路径中找到trian文件夹
            self.style_images = self.get_data(sty_dir) # 利用self.get_data函数获取文件夹中所有['*.jpg', '*.png', '*.jpeg', '*.tif']文件, 返回包含所有图像文件名的列表
            # 以下的if-elif语句用于使 风格数据集中的数据量 与 内容数据集中的数据量 保持一致
            # 如果 内容数据集中的数据量少了, 则使用同样数量的风格数据集
            if len(self.images) < len(self.style_images): # 如果 内容数据集中的数据量 小于 风格数据集中的数据量
                self.style_images = random.sample(self.style_images, len(self.images)) # 则进行随机采样, 从风格图像中用与内容图像数量相同的图像
            # 如果 风格数据集中的数据量少了, 则重复使用风格数据集多次
            elif len(self.images) > len(self.style_images): # 如果 内容数据集中的数据量 大于 风格数据集中的数据量
                ratio = len(self.images) // len(self.style_images) # 计算 内容数据集中的数据量 是 风格数据集中数据量 的几倍, 取整数部分
                bias = len(self.images) - ratio * len(self.style_images) # 计算 除整数倍外, 内容数据集中的数据量 还比 风格数据集中的数据量 多多少
                self.style_images = self.style_images * ratio # 将风格数据集使用ratio次
                self.style_images += random.sample(self.style_images, bias) # 最后还不够的部分使用随机采样补全
            # assert是Python关键词, 为代码添加"断言"功能, 用于在程序运行时检查, 如果不满足条件, 则会触发 AssertionError 异常
            assert len(self.images) == len(self.style_images)
            
        elif phase == 'test':
            # 如果当前为测试模式, 则直接获取从Config.py中获取内容图像与风格图像的存储路径, 并利用get_data函数获取文件名列表       
            img_dir = Path(config.content_dir)
            self.images = self.get_data(img_dir)[:config.data_num]
            
            sty_dir = Path(config.style_dir)
            self.style_images = self.get_data(sty_dir)[:config.data_num]
            
            ################################zxt修改部分begin##########################
             # 以下的if-elif语句用于使 风格数据集中的数据量 与 内容数据集中的数据量 保持一致
            # 如果 内容数据集中的数据量少了, 则使用同样数量的风格数据集
            if len(self.images) < len(self.style_images): # 如果 内容数据集中的数据量 小于 风格数据集中的数据量
                self.style_images = random.sample(self.style_images, len(self.images)) # 则进行随机采样, 从风格图像中用与内容图像数量相同的图像
            # 如果 风格数据集中的数据量少了, 则重复使用风格数据集多次
            elif len(self.images) > len(self.style_images): # 如果 内容数据集中的数据量 大于 风格数据集中的数据量
                ratio = len(self.images) // len(self.style_images) # 计算 内容数据集中的数据量 是 风格数据集中数据量 的几倍, 取整数部分
                bias = len(self.images) - ratio * len(self.style_images) # 计算 除整数倍外, 内容数据集中的数据量 还比 风格数据集中的数据量 多多少
                self.style_images = self.style_images * ratio # 将风格数据集使用ratio次
                self.style_images += random.sample(self.style_images, bias) # 最后还不够的部分使用随机采样补全
            # assert是Python关键词, 为代码添加"断言"功能, 用于在程序运行时检查, 如果不满足条件, 则会触发 AssertionError 异常
            assert len(self.images) == len(self.style_images)
            ################################zxt修改部分end##########################
            
        
        print('content dir:', img_dir)
        print('style dir:', sty_dir)
            
    def __len__(self):
        """
        返回内容数据集中的图像数量, 风格数据集中的数量需要与这个值保持一致
        """
        return len(self.images)
    
    def get_data(self, img_dir):  
        """
        获取文件夹中所有['*.jpg', '*.png', '*.jpeg', '*.tif']文件, 返回包含所有图像文件名的列表
        """
        file_type = ['*.jpg', '*.png', '*.jpeg', '*.tif']
        imgs = []
        for ft in file_type:
            imgs += sorted(img_dir.glob(ft))
        images = sorted(imgs)
        return images

    def __getitem__(self, index):
        """
        从 self.images 列表中获取对应编号的内容与风格图片的名称, 并使用定义的Transform模块处理图像, 包括Resize,
        """
        cont_img = self.images[index] # 从 self.images 列表中获取对应编号的内容图片名称
        cont_img = Image.open(cont_img).convert('RGB') # 使用PIL中Image包以RGB的模式打开对应的内容图像

        sty_img = self.style_images[index]# 从 self.style_images 列表中获取对应编号的风格图片名称
        sty_img = Image.open(sty_img).convert(mode='RGB',colors=0) # 使用PIL中Image包以RGB的模式打开对应的风格图像
        sty_img = self.transform(sty_img) # 使用定义的Transform模块处理图像, 包括Resize, RandomCrop, ToTensor与Normalize

        img_dict: Dict[str, Image.Image] = {'content_img': cont_img, 'style_img': sty_img} # 返回一个字典
        
        return img_dict
