import os
import torch
import numpy as np
import tensorboardX

# 以下的import均为自定义的模块
from Config import Config
from DataSplit import DataSplit
from model import AesFA
from blocks import model_save, model_load, update_learning_rate

from torch.utils.data import RandomSampler

# 生成输出文件夹
def mkoutput_dir(config):
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)

def get_n_params(model):
    """
    This function is used to get the number of parameters in the model

    Parameters:
    ----------
    model : nn.Module
        The pytorch model

    Returns:
    -------
    total_params : int
        The total number of parameters
    net_params : dict
        The number of parameters for each sub-network
    """
    total_params=0
    net_params = {'netE':0, 'netS':0, 'netG':0, 'vgg_loss':0}

    for name, param in model.named_parameters():
        # e.g. netE.conv1.weight
        net = name.split('.')[0]
        # calculate the number of parameters
        nn=1
        for s in list(param.size()):
            nn = nn*s
        # update the number of parameters of the sub-network
        net_params[net] += nn
        # update the total number of parameters
        total_params += nn

    return total_params, net_params

def im_convert(tensor):
    """
    Convert a tensor to numpy image.

    The tensor is first converted to numpy array on CPU,
    and then the ordering of axes is reversed to convert from
    CHW to HWC. The values are then normalized using mean and std
    values of ImageNet. Finally, the clipping is applied to ensure
    that the values remain between 0 and 1.

    Parameters
    ----------
    tensor : torch.Tensor
        The input tensor

    Returns
    -------
    np.ndarray
        The converted image
    """
    image = tensor.to("cpu").clone().detach().numpy()
    image = image.transpose(0, 2, 3, 1)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image


def main():
    """
    Main function to train the AesFA network.

    It loads the configuration, initializes the model, data loader,
    and starts the training.

    """
    config = Config()
    mkoutput_dir(config)

    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('cuda:', config.device)
    print('Version:', config.file_n)
    
    ########## Data Loader ##########
    train_data = DataSplit(config=config, phase='train')
    train_sampler = RandomSampler(train_data)
    data_loader_train = torch.utils.data.DataLoader(train_data, batch_size=config.batch_size,  num_workers=config.num_workers, pin_memory=False, sampler=train_sampler)
    print("Train: ", train_data.__len__(), "images: ", len(data_loader_train), "x", config.batch_size,"(batch size) =", train_data.__len__())

    ########## load model ##########
    model = AesFA(config)
    model.to(config.device)
    
    # # of parameter
    param_num, net_params = get_n_params(model)
    print("# of parameter:", param_num)
    print("parameters of networks:", net_params)

    ########## load saved model - to continue previous learning ##########
    if config.train_continue == 'on':
        model, model.optimizer_E, model.optimizer_S, model.optimizer_G, epoch_start, tot_itr = model_load(checkpoint=None, ckpt_dir=config.ckpt_dir, model=model,
                           optim_E=model.optimizer_E,
                           optim_S=model.optimizer_S,
                           optim_G=model.optimizer_G)
        print(epoch_start, "th epoch ", tot_itr, "th iteration model load")
    else:
        epoch_start = 0
        tot_itr = 0

    train_writer = tensorboardX.SummaryWriter(config.log_dir)

    ########## Training ##########
    # to save ckpt file starting with epoch and iteration 1
    epoch = epoch_start - 1
    tot_itr = tot_itr - 1
    while tot_itr < config.n_iter:
        epoch += 1
        
        for i, data in enumerate(data_loader_train):
            tot_itr += 1
            train_dict = model.train_step(data)

            real_A = im_convert(data['content_img'])
            real_B = im_convert(train_dict['style_img'])
            fake_B = im_convert(train_dict['fake_AtoB'])
            trs_high = im_convert(train_dict['fake_AtoB_high'])
            trs_low = im_convert(train_dict['fake_AtoB_low'])

            ## Tensorboard ##
            # tensorboard - loss
            train_writer.add_scalar('Loss_G', train_dict['G_loss'], tot_itr)
            train_writer.add_scalar('Loss_G_Percept', train_dict['G_Percept'], tot_itr)
            train_writer.add_scalar('Loss_G_Contrast', train_dict['G_Contrast'], tot_itr)

            # tensorboard - images
            train_writer.add_image('Content_Image_A', real_A, tot_itr, dataformats='NHWC')
            train_writer.add_image('Style_Image_B', real_B, tot_itr, dataformats='NHWC')
            train_writer.add_image('Generated_Image_AtoB', fake_B, tot_itr, dataformats='NHWC')
            train_writer.add_image('Translation_AtoB_high', trs_high, tot_itr, dataformats='NHWC')
            train_writer.add_image('Translation_AtoB_low', trs_low, tot_itr, dataformats='NHWC')

            print("Tot_itrs: %d/%d | Epoch: %d | itr: %d/%d | Loss_G: %.5f"%(tot_itr+1, config.n_iter, epoch+1, (i+1), len(data_loader_train), train_dict['G_loss']))

            if (tot_itr + 1) % 10000 == 0:
                model_save(ckpt_dir=config.ckpt_dir, model=model, optim_E=model.optimizer_E, optim_S=model.optimizer_S, optim_G=model.optimizer_G, epoch=epoch, itr=tot_itr)
                print(tot_itr+1, "th iteration model save")

        update_learning_rate(model.E_scheduler, model.optimizer_E)
        update_learning_rate(model.S_scheduler, model.optimizer_S)
        update_learning_rate(model.G_scheduler, model.optimizer_G)

if __name__ == '__main__':
    main()
