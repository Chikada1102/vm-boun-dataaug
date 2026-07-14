from torchvision import transforms
from utils import *
from datetime import datetime


class setting_config:
    """
    VM-UNet + IBR Boundary Refinement 配置。
    IBR = Interior-guided Boundary Refinement。
    """

    network = 'vmunet'
    model_config = {
        'num_classes': 1,
        'input_channels': 3,
        # ----- VM-UNet ----- #
        'depths': [2, 2, 2, 2],
        'depths_decoder': [2, 2, 2, 1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
        # ==================== 新添加的代码开始：模型内部启用 IBR 内部区域引导边界细化 ====================
        'use_ibr_guidance': True,
        'ibr_kernel_size': 5,
        # ==================== 新添加的代码结束：模型内部启用 IBR 内部区域引导边界细化 ====================
    }

    # 【原始代码删除说明】
    # 原始默认数据集：datasets = 'isic18'
    # 这里按你的实验要求默认改为 ISIC2017。
    datasets = 'isic17'
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    # ==================== 新添加的代码开始：训练全局启用 IBR，loss 必须消费 boundary 标签和 IBR 输出 ====================
    use_ibr_guidance = True
    boundary_label_dir = './data/isic2017/train/boundaries/'
    aux_label_fail_silently = False
    criterion = IBRBoundaryRefinementLoss(
        wb=1,
        wd=1,
        coarse_weight=0.25,
        boundary_weight=0.25,
        outer_suppression_weight=0.05,
        # ==================== 新添加的代码开始：启用 background prototype 对比式抑制 loss ====================
        background_suppression_weight=0.05,
        contrast_boundary_weight=0.05,
        # ==================== 新添加的代码结束：启用 background prototype 对比式抑制 loss ====================
    )
    # ==================== 新添加的代码结束：训练全局启用 IBR，loss 必须消费 boundary 标签和 IBR 输出 ====================

    # 【原始代码删除说明】
    # 原始 criterion = BceDiceLoss(wb=1, wd=1)
    # 已替换为 IBRBoundaryRefinementLoss；如果模型不返回 coarse/final/boundary/IBR 标志，会直接报错。

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 8
    prefetch_factor = 4
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 300

    # 【原始代码删除说明】
    # 上一版结果目录：_ibr_boundary_refinement_
    # work_dir = 'results/' + network + '_' + datasets + '_ibr_boundary_refinement_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    # ==================== 新添加的代码开始：区分 background prototype contrast 版本结果目录 ====================
    work_dir = 'results/' + network + '_' + datasets + '_ibr_bgproto_contrast_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'
    # ==================== 新添加的代码结束：区分 background prototype contrast 版本结果目录 ====================

    print_interval = 20
    val_interval = 10
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'

    train_transformer = transforms.Compose([
        myNormalize(datasets, train=True),
        myToTensor(),
        myRandomHorizontalFlip(p=0.5),
        myRandomVerticalFlip(p=0.5),
        myRandomRotation(p=0.5, degree=[0, 360]),
        myResize(input_size_h, input_size_w)
    ])
    test_transformer = transforms.Compose([
        myNormalize(datasets, train=False),
        myToTensor(),
        myResize(input_size_h, input_size_w)
    ])

    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'AdamW':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == 'Adam':
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == 'SGD':
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False
    elif opt == 'Adadelta':
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == 'Adagrad':
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == 'Adamax':
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == 'ASGD':
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == 'RMSprop':
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == 'Rprop':
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)

    sch = 'CosineAnnealingLR'
    if sch == 'CosineAnnealingLR':
        T_max = 300
        eta_min = 0.00001
        last_epoch = -1
    elif sch == 'StepLR':
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == 'ExponentialLR':
        gamma = 0.99
        last_epoch = -1
    elif sch == 'ReduceLROnPlateau':
        mode = 'min'
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = 'rel'
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
