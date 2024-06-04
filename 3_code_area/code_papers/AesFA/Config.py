class Config:
    '''
    一个工具类, 里面全是表示配置信息的字符串或整型数. 其他文件将从这个类中获取配置信息
    '''
    phase = 'test'         # You must change the phase into train/test/style_blending
    train_continue = 'off'  # on / off

    data_num = 60000        # Maximum # of training data

    content_dir = '/mnt/sda/zxt/3_code_area/0_code_hello/test_metirc3d/outputs/camera_man/img_background'
    style_dir = '/mnt/sda/zxt/3_code_area/code_papers/AesFA/img/background_style'
    
    file_n = 'main'
    log_dir = './log/' + file_n
    ckpt_dir = './ckpt/' + file_n
    img_dir = './Generated_images/' + file_n
    
    if phase == 'test':
        multi_to_multi = False #True表示整个图像以one content image, N style image的形式进行测试
        test_content_size = 256
        test_style_size = 256
        content_dir = content_dir
        style_dir = style_dir
        img_dir = './output/'+file_n+'/'+ 'cameraman'

    elif phase == 'style_blending':
        blend_load_size = 256
        blend_dir = './blendingDataset/'
        content_img = blend_dir + str(blend_load_size) + '/A.jpg'
        style_high_img = blend_dir + str(blend_load_size) + '/B.jpg'
        style_low_img = blend_dir + str(blend_load_size) + '/C.jpg'
        img_dir = './output/'+file_n+'_blending_' + str(blend_load_size)
        
    # VGG pre-trained model
    vgg_model = '/mnt/sda/zxt/3_code_area/code_papers/AesFA/vgg_model/vgg_normalised.pth'

    ## basic parameters
    n_iter = 160000
    batch_size = 8
    lr = 0.0001
    lr_policy = 'step'
    lr_decay_iters = 50
    beta1 = 0.0

    # preprocess parameters
    load_size = 512
    crop_size = 256

    # model parameters
    input_nc = 3         # of input image channel
    nf = 64              # of feature map channel after Encoder first layer
    output_nc = 3        # of output image channel
    style_kernel = 3     # size of style kernel
    
    # Octave Convolution parameters
    alpha_in = 0.5       # input ratio of low-frequency channel
    alpha_out = 0.5      # output ratio of low-frequency channel
    freq_ratio = [1, 1]  # [high, low] ratio at the last layer

    # Loss ratio
    lambda_percept = 1.0
    lambda_perc_cont = 1.0
    lambda_perc_style = 10.0
    lambda_const_style = 5.0

    # Else
    norm = 'instance'
    init_type = 'normal'
    init_gain = 0.02
    no_dropout = 'store_true'
    num_workers = 4
