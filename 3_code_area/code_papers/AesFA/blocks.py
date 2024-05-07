##########begin参数说明############
# lf_ch_in: short for "low frequency channel in", 表示输入的低频通道数
# hf_ch_in: short for "high frequency channel in"

##########end参数说明##############



import os
import glob
from path import Path
import math
import torch
from torch import nn
import torch.nn.functional as F

# lr_scheduler用于根据epoch调整学习率
from torch.optim import lr_scheduler
from blocks import *

# 用于保存模型的参数的文件
def model_save(ckpt_dir, model, optim_E, optim_S, optim_G, epoch, itr):
    """
    Save the model parameters to the directory "ckpt_dir", with the iteration number "itr" and the epoch number "epoch".
    The saved file name will be model_iter_<itr>_epoch_<epoch>.pth
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({
        'netE': model.netE.state_dict(),
        'netS': model.netS.state_dict(),
        'netG': model.netG.state_dict(),
        'optim_E': optim_E.state_dict(),
        'optim_S': optim_S.state_dict(),
        'optim_G': optim_G.state_dict()
    }, '%s/model_iter_%d_epoch_%d.pth' % (ckpt_dir, itr+1, epoch+1))

# 用于读取保存的权重参数
def model_load(checkpoint, ckpt_dir, model, optim_E, optim_S, optim_G):
    # 如果不存在路径, 则返回epch=-1, 无法运行
    if not os.path.exists(ckpt_dir):
        epoch = -1
        return model, optim_E, optim_S, optim_G, epoch
    
    # chpt_path is for checkpoint_path
    ckpt_path = Path(ckpt_dir)

    if checkpoint:    # 如果checkpoint 不为空, 
        model_ckpt = ckpt_path + '/' + checkpoint   #则与ckpt_path拼接, 获取checkpoint文件的位置作为model_ckpt
    else: # 若chekppoint为空, 则获取最新的checkpoint文件作为model_ckpt
        ckpt_lst = ckpt_path.glob('model_iter_*')        # 查找文件名中带有"model_iter_*"的文件, 该操作将返回一个列表
        ckpt_lst.sort(key=lambda x: int(x.split('iter_')[1].split('_epoch')[0]))
        model_ckpt = ckpt_lst[-1]
    
    # model_ckpt文件名的格式为: model_iter_<数字>_epoch_<数字>
    itr = int(model_ckpt.split('iter_')[1].split('_epoch_')[0])    # 从文件名中获取当前的model_ckpt文件是第几个epoch的
    epoch = int(model_ckpt.split('iter_')[1].split('_epoch_')[1].split('.')[0])    # 从文件名中获取当前的model_ckpt文件是第几个epoch的
    print(model_ckpt)

    # 从文件名中获取参数字典文件
    dict_model = torch.load(model_ckpt)

    # 载入参数字典
    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    optim_E.load_state_dict(dict_model['optim_E'])
    optim_S.load_state_dict(dict_model['optim_S'])
    optim_G.load_state_dict(dict_model['optim_G'])

    # 返回加载好的模型model, 与优化器optim
    return model, optim_E, optim_S, optim_G, epoch, itr

# 加载测试用模型
def test_model_load(checkpoint, model):
    dict_model = torch.load(checkpoint)
    model.netE.load_state_dict(dict_model['netE'])
    model.netS.load_state_dict(dict_model['netS'])
    model.netG.load_state_dict(dict_model['netG'])
    return model

def get_scheduler(optimizer, config):   # 本函数用于定义更新学习率的方式, 当config参数为lambda时, 采用第一个if分支
    if config.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + config.n_epoch - config.n_iter) / float(config.n_iter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
        # lr_scheduler是一个用于更新学习率的模块(module), 其中包含了许多种更新学习率的方式.
    elif config.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=config.lr_decay_iters, gamma=0.1)
    elif config.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif config.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.n_iter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', config.lr_policy)
    return scheduler

def update_learning_rate(scheduler, optimizer):
    """
    调用scheduler进行学习率更新
    scheduler: 用于更新学习率的对象(对象的类型由配置文件中的lr_policy决定)
    optimizer: 优化器对象
    """
    # 调用scheduler进行学习率更新
    # scheduler: 用于更新学习率的对象(对象的类型由配置文件中的lr_policy决定)
    # optimizer: 优化器对象
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)


class Oct_Conv_aftup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, pad_type, alpha_in, alpha_out):
        super(Oct_Conv_aftup, self).__init__()
        lf_in = int(in_channels*alpha_in)
        lf_out = int(out_channels*alpha_out)
        hf_in = in_channels - lf_in
        hf_out = out_channels - lf_out

        self.conv_h = nn.Conv2d(in_channels=hf_in, out_channels=hf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
        self.conv_l = nn.Conv2d(in_channels=lf_in, out_channels=lf_out, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type)
    
    def forward(self, x):
        hf, lf = x
        hf = self.conv_h(hf)
        lf = self.conv_l(lf)
        return hf, lf

class Oct_conv_reLU(nn.ReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_reLU, self).forward(hf)
        lf = super(Oct_conv_reLU, self).forward(lf)
        return hf, lf
    
class Oct_conv_lreLU(nn.LeakyReLU):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_lreLU, self).forward(hf)
        lf = super(Oct_conv_lreLU, self).forward(lf)
        return hf, lf

class Oct_conv_up(nn.Upsample):
    def forward(self, x):
        hf, lf = x
        hf = super(Oct_conv_up, self).forward(hf)
        lf = super(Oct_conv_up, self).forward(lf)
        return hf, lf


############## Encoder ##############

class OctConv(nn.Module):  # 八度卷积
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, groups=1, pad_type='reflect', alpha_in=0.5, alpha_out=0.5, oct_conv_type='normal', freq_ratio = [1, 1]):
        super(OctConv, self).__init__()
        self.kernel_size = kernel_size    # 卷积核大小
        self.stride = stride    # 卷积核步长
        self.oct_conv_type = oct_conv_type    # OntConv类中所有的oct_conv_type均为本人修改而来, 原来为type
        self.alpha_in = alpha_in    # alpha_in表示输入通道中, 多少算低频部分
        self.alpha_out = alpha_out    # αlpha_out表示输出通道中, 多少通道算低频部分
        self.freq_ratio = freq_ratio   # 这个变量的含义此处还不清楚, 稍后补充

        ##begin 用于构建卷积核的参数
        hf_ch_in = int(in_channels * (1 - self.alpha_in))    # 根据1-alpha_in参数计算高频通道, 用于构建卷积核
        hf_ch_out = int(out_channels * (1 -self. alpha_out))    # 根据1-alpha_in计算输出的高频通道, 用于构建卷积核
        lf_ch_in = in_channels - hf_ch_in    # 输入的低频通道, 用于构建卷积核
        lf_ch_out = out_channels - hf_ch_out    # 输出的低频通道, 用于构建卷积核
        ##end 用于构建卷积核的参数


        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)    # 池化层, 池化窗口为2*2的大小, 步长为2
        self.upsample = nn.Upsample(scale_factor=2)    # 上采样层, 图像大小变为原来的2倍

        self.is_dw = groups == in_channels    # 什么意思？

        if oct_conv_type == 'first':    # 此处first表示输入通道数为in_channels、 输出通道数为hf_ch_out, lf_ch_out的这种卷积, 应该是第一层的卷积核
            # 下行为构建一个卷积, 从名字上来看, 该卷积(convh)是一个对图像高频部分进行卷积的卷积核
            self.convh = nn.Conv2d(in_channels = in_channels, out_channels = hf_ch_out, 
                                    kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias = False)
            # 下行为构建一个卷积, 从名字上来看, 该卷积(convl)是一个对图像低频部分进行卷积的卷积核
            self.convl = nn.Conv2d(in_channels=in_channels, out_channels = lf_ch_out,
                                   kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        elif oct_conv_type == 'last':    # 此处的last表示输入通道数为hf_ch_in、lf_ch_in、输出通道数为out_channels的这种卷积, 应该是最后一层的卷积
            self.convh = nn.Conv2d(in_channels=hf_ch_in,out_channels=out_channels, 
                                   kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
            self.convl = nn.Conv2d(in_channels=lf_ch_in, out_channels=out_channels, 
                                    kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=pad_type, bias=False)
        else:    # 如果不是first也不是last, 则为中间层的卷积. 
            self.L2L = nn.Conv2d( # L2L表示 OctConv 中的低频到低频卷积核, 该卷积核的输入为lf_ch_in, 输出为lf_ch_out
                in_channels=lf_ch_in, out_channels=lf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(alpha_in * groups), padding_mode=pad_type, bias=False
            )
            if self.is_dw: # 如果是dw, 则将两个卷积核变为None, 不用于接受张量(Tensor)
                self.L2H = None
                self.H2L = None
            else:
                self.L2H = nn.Conv2d( # L2H表示 OctConv 中的低频到高频卷积核, 用于高低频之间的信息交流, 该卷积核的输入为lf_ch_in, 输出为hf_ch_out
                    in_channels=lf_ch_in, out_channels=hf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )
                self.H2L = nn.Conv2d( # H2L表示 OctConv 中的高频到低频卷积核, 用于高低频之间的信息交流, 该卷积核的输入为hf_ch_in, 输出为lf_ch_out
                    in_channels=hf_ch_in, out_channels=lf_ch_out,
                    kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, padding_mode=pad_type, bias=False
                )

            self.H2H = nn.Conv2d( # H2H表示 OctConv 中的高频到高频卷积核, 该卷积核的输入为hf_ch_in, 输出也为hf_ch_out
                in_channels=hf_ch_in, out_channels=hf_ch_out,
                kernel_size=kernel_size, stride=stride, padding=padding, groups=math.ceil(groups - alpha_in * groups), padding_mode=pad_type, bias=False
            )
            
    def forward(self, x):
        if self.oct_conv_type == 'first': # 定义前向过程, 如果是第一层, 则调用first相关的卷积核
            hf = self.convh(x)
            lf = self.avg_pool(x)
            lf = self.convl(lf)
            return hf, lf
        elif self.oct_conv_type == 'last': # 定义前向过程, 如果是最后一层, 则调用last相关的卷积核
            hf, lf = x
            out_h = self.convh(hf)
            out_l = self.convl(self.upsample(lf))
            output = out_h * self.freq_ratio[0] + out_l * self.freq_ratio[1]
            return output, out_h, out_l
        else:# 定义前向过程, 如果是中间层, 则调用其他相关的卷积核
            hf, lf = x
            if self.is_dw:
                hf, lf = self.H2H(hf), self.L2L(lf)
            else:
                hf, lf = self.H2H(hf) + self.L2H(self.upsample(lf)), self.L2L(lf) + self.H2L(self.avg_pool(hf))
            return hf, lf
        
############## Encoder End ##########

############## Decoder ##############
class AdaOctConv(nn.Module):  # 初步判断为使用上述定义的OctConv类, 结合PyTorch自带的AdaConv进行网络模块的搭建
    def __init__(self, in_channels, out_channels, group_div, style_channels, kernel_size,
                 stride, padding, oct_groups, alpha_in, alpha_out, ada_oct_conv_type='normal'):
        super(AdaOctConv, self).__init__()
        self.in_channels = in_channels      # 将输入通道数存在类对象内部
        self.alpha_in = alpha_in    # 输入通道中, 低频所占用的比例
        self.alpha_out = alpha_out  # 输出通道中, 低频通道的比例
        self.ada_oct_conv_type = ada_oct_conv_type # 表示运行的类别, 包括本行的上面四行, 均为传递给OctConv类的参数
        
        h_in = int(in_channels * (1 - self.alpha_in)) # 定义输入的高频通道部分
        l_in = in_channels - h_in # 定义输入的低频通道部分

        n_groups_h = h_in // group_div # 将输入的高频通道部分分成多组
        n_groups_l = l_in // group_div # 将输入的低频通道部分分成多组
        
        style_channels_h = int(style_channels * (1 - self.alpha_in)) # 定义风格图像的高频部分
        style_channels_l = int(style_channels - style_channels_h)   # 定义风格图像的低频部分
        
        kernel_size_h = kernel_size[0]  # 根据传入的kernel_size列表设定不同的kernel_size, kernel_size_h是高频的卷积核大小
        kernel_size_l = kernel_size[1]  # 根据传入的kernel_size列表设定不同的kernel_size, kernel_size_l是低频的卷积核大小
        kernel_size_A = kernel_size[2]  # 根据传入的kernel_size列表设定不同的kernel_size, kernel_size_A是什么还不理解 

        self.kernelPredictor_h = KernelPredictor(in_channels=h_in,
                                              out_channels=h_in,
                                              n_groups=n_groups_h,
                                              style_channels=style_channels_h,
                                              kernel_size=kernel_size_h)  # KernelPredictor是下面定义的类
        self.kernelPredictor_l = KernelPredictor(in_channels=l_in,
                                               out_channels=l_in,
                                               n_groups=n_groups_l,
                                               style_channels=style_channels_l,
                                               kernel_size=kernel_size_l)
        
        self.AdaConv_h = AdaConv2d(in_channels=h_in, out_channels=h_in, n_groups=n_groups_h)
        self.AdaConv_l = AdaConv2d(in_channels=l_in, out_channels=l_in, n_groups=n_groups_l)
        
        self.OctConv = OctConv(in_channels=in_channels,
                            out_channels=out_channels,
                            kernel_size=kernel_size_A, stride=stride, padding=padding, groups=oct_groups,
                            alpha_in=alpha_in, alpha_out=alpha_out, oct_conv_type=ada_oct_conv_type)
        
        self.relu = Oct_conv_lreLU()

    def forward(self, content, style, cond='train'):
        c_hf, c_lf = content
        s_hf, s_lf = style
        h_w_spatial, h_w_pointwise, h_bias = self.kernelPredictor_h(s_hf)
        l_w_spatial, l_w_pointwise, l_bias = self.kernelPredictor_l(s_lf)
        
        if cond == 'train':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l

            output = self.relu(output)

            output = self.OctConv(output)
            if self.ada_oct_conv_type != 'last':
                output = self.relu(output)
            return output
        
        if cond == 'test':
            output_h = self.AdaConv_h(c_hf, h_w_spatial, h_w_pointwise, h_bias)
            output_l = self.AdaConv_l(c_lf, l_w_spatial, l_w_pointwise, l_bias)
            output = output_h, output_l
            output = self.relu(output)
            output = self.OctConv(output)
            if self.ada_oct_conv_type != 'last':
                output = self.relu(output)
            return output

class KernelPredictor(nn.Module):
    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size):
        super(KernelPredictor, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_groups = n_groups
        self.w_channels = style_channels
        self.kernel_size = kernel_size

        padding = (kernel_size - 1) / 2
        self.spatial = nn.Conv2d(style_channels,
                                 in_channels * out_channels // n_groups,
                                 kernel_size=kernel_size,
                                 padding=(math.ceil(padding), math.ceil(padding)),
                                 padding_mode='reflect')
        self.pointwise = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels * out_channels // n_groups,
                      kernel_size=1)
        )
        self.bias = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(style_channels,
                      out_channels,
                      kernel_size=1)
        )

    def forward(self, w):
        w_spatial = self.spatial(w)
        w_spatial = w_spatial.reshape(len(w),
                                      self.out_channels,
                                      self.in_channels // self.n_groups,
                                      self.kernel_size, self.kernel_size)

        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.reshape(len(w),
                                          self.out_channels,
                                          self.out_channels // self.n_groups,
                                          1, 1)
        bias = self.bias(w)
        bias = bias.reshape(len(w), self.out_channels)
        return w_spatial, w_pointwise, bias

class AdaConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, n_groups=None):
        super(AdaConv2d, self).__init__()
        self.n_groups = in_channels if n_groups is None else n_groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        padding = (kernel_size - 1) / 2
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=(kernel_size, kernel_size),
                              padding=(math.ceil(padding), math.floor(padding)),
                              padding_mode='reflect')

    def forward(self, x, w_spatial, w_pointwise, bias):
        assert len(x) == len(w_spatial) == len(w_pointwise) == len(bias)
        x = F.instance_norm(x)

        ys = []
        for i in range(len(x)):
            y = self.forward_single(x[i:i+1], w_spatial[i], w_pointwise[i], bias[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        ys = self.conv(ys)
        return ys

    def forward_single(self, x, w_spatial, w_pointwise, bias):
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (math.ceil(padding), math.floor(padding), math.ceil(padding), math.floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise, groups=self.n_groups, bias=bias)
        return x
