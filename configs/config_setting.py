from torchvision import transforms
from utils import *

from datetime import datetime
# new
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
# new
class setting_config:
    """
    the config of training setting.
    """

    network = 'vmunet_dataaug2'
    model_config = {
        'num_classes': 1, 
        'input_channels': 3, 
        # ----- VM-UNet ----- #
        'depths': [2,2,2,2],
        'depths_decoder': [2,2,2,1],
        'drop_path_rate': 0.2,
        'load_ckpt_path': './pre_trained_weights/vmamba_small_e238_ema.pth',
    }

    datasets = 'isic18' 
    if datasets == 'isic18':
        data_path = './data/isic2018/'
    elif datasets == 'isic17':
        data_path = './data/isic2017/'
    else:
        raise Exception('datasets in not right!')

    criterion = BceDiceLoss(wb=1, wd=1)

    pretrained_path = './pre_trained/'
    num_classes = 1
    input_size_h = 256
    input_size_w = 256
    input_channels = 3
    distributed = False
    local_rank = -1
    num_workers = 0
    seed = 42
    world_size = None
    rank = None
    amp = False
    gpu_id = '0'
    batch_size = 32
    epochs = 300

    work_dir = 'results/' + network + '_' + datasets + '_' + datetime.now().strftime('%A_%d_%B_%Y_%Hh_%Mm_%Ss') + '/'

    print_interval = 20
    val_interval = 30
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = 'PATH_TO_YOUR_BEST_CKPT'
    img_save_path = 'PATH_TO_SAVE_IMAGES'

    # train_transformer = transforms.Compose([
    #     myNormalize(datasets, train=True),
    #     myToTensor(),
    #     myRandomHorizontalFlip(p=0.5),
    #     myRandomVerticalFlip(p=0.5),
    #     myRandomRotation(p=0.5, degree=[0, 360]),
    #     myResize(input_size_h, input_size_w)
    # ])
    # test_transformer = transforms.Compose([
    #     myNormalize(datasets, train=False),
    #     myToTensor(),
    #     myResize(input_size_h, input_size_w)
    # ])
# new2
    class HairArtifact(A.ImageOnlyTransform):
        def __init__(
            self,
            num_hairs=(3, 8),
            hair_length=(30, 120),
            hair_thickness=(1, 2),
            darkness=(20, 80),
            always_apply=False,
            p=0.25
        ):
            super().__init__(always_apply, p)
            self.num_hairs = num_hairs
            self.hair_length = hair_length
            self.hair_thickness = hair_thickness
            self.darkness = darkness

        def apply(self, image, **params):
            img = image.copy()
            h, w = img.shape[:2]

            num = np.random.randint(self.num_hairs[0], self.num_hairs[1] + 1)

            for _ in range(num):
                x1 = np.random.randint(0, w)
                y1 = np.random.randint(0, h)

                length = np.random.randint(self.hair_length[0], self.hair_length[1] + 1)
                angle = np.random.uniform(0, 2 * np.pi)

                x2 = int(x1 + length * np.cos(angle))
                y2 = int(y1 + length * np.sin(angle))

                thickness = np.random.randint(
                    self.hair_thickness[0],
                    self.hair_thickness[1] + 1
                )

                color_value = np.random.randint(
                    self.darkness[0],
                    self.darkness[1] + 1
                )

                color = (color_value, color_value, color_value)

                cv2.line(
                    img,
                    (x1, y1),
                    (x2, y2),
                    color,
                    thickness,
                    lineType=cv2.LINE_AA
                )

            return img


    train_transformer = A.Compose([
        # 更推荐用 RandomResizedCrop 替代单纯 Resize
        # 作用：模拟病灶大小、视野范围变化
        A.RandomResizedCrop(
            size=(input_size_h, input_size_w),
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=0.65
        ),

        # 如果上面的 RandomResizedCrop 没触发，保证尺寸一致
        A.Resize(
            height=input_size_h,
            width=input_size_w,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1.0
        ),

        # 几何增强
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),

        # 比原来的 ShiftScaleRotate 更强一点
        A.OneOf([
            A.ShiftScaleRotate(
                shift_limit=0.08,
                scale_limit=0.15,
                rotate_limit=45,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0
            ),

            # 轻微弹性形变：适合边界不规则的皮肤病灶
            A.ElasticTransform(
                alpha=25,
                sigma=5,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0
            ),

            # 网格形变：模拟局部拉伸
            A.GridDistortion(
                num_steps=5,
                distort_limit=0.18,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                mask_value=0,
                p=1.0
            ),
        ], p=0.45),

        # 颜色 / 光照增强：这里比你原来强
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.22,
                contrast_limit=0.22,
                p=1.0
            ),
            A.HueSaturationValue(
                hue_shift_limit=12,
                sat_shift_limit=18,
                val_shift_limit=15,
                p=1.0
            ),
            A.RandomGamma(
                gamma_limit=(70, 135),
                p=1.0
            ),
            A.RGBShift(
                r_shift_limit=12,
                g_shift_limit=12,
                b_shift_limit=12,
                p=1.0
            ),
            A.CLAHE(
                clip_limit=2.0,
                tile_grid_size=(8, 8),
                p=1.0
            ),
        ], p=0.75),

        # 成像退化：比原来略强，但概率不要太高
        A.OneOf([
            A.GaussNoise(
                var_limit=(5.0, 35.0),
                p=1.0
            ),
            A.GaussianBlur(
                blur_limit=(3, 5),
                p=1.0
            ),
            A.MotionBlur(
                blur_limit=5,
                p=1.0
            ),
            A.ImageCompression(
                quality_lower=65,
                quality_upper=95,
                p=1.0
            ),
        ], p=0.22),
        # 毛发遮挡，只改 image，不改 mask
        HairArtifact(
            num_hairs=(3, 8),
            hair_length=(30, 120),
            hair_thickness=(1, 2),
            darkness=(15, 80),
            p=0.25
        ),

        # 小块遮挡，模拟气泡、标尺、局部反光、遮挡
        # 注意：只建议小概率，不要太大
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_size_range=(0.02, 0.08),
            fill=0,
            p=0.18
        ),
        # ====================== 新添加结束 ======================

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])


    test_transformer = A.Compose([
        A.Resize(
            height=input_size_h,
            width=input_size_w,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST
        ),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),

        ToTensorV2()
    ])
# new2
    opt = 'AdamW'
    assert opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if opt == 'Adadelta':
        lr = 0.01 # default: 1.0 – coefficient that scale delta before it is applied to the parameters
        rho = 0.9 # default: 0.9 – coefficient used for computing a running average of squared gradients
        eps = 1e-6 # default: 1e-6 – term added to the denominator to improve numerical stability 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'Adagrad':
        lr = 0.01 # default: 0.01 – learning rate
        lr_decay = 0 # default: 0 – learning rate decay
        eps = 1e-10 # default: 1e-10 – term added to the denominator to improve numerical stability
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Adam':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability 
        weight_decay = 0.0001 # default: 0 – weight decay (L2 penalty) 
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond
    elif opt == 'AdamW':
        lr = 0.001 # default: 1e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 1e-2 # default: 1e-2 – weight decay coefficient
        amsgrad = False # default: False – whether to use the AMSGrad variant of this algorithm from the paper On the Convergence of Adam and Beyond 
    elif opt == 'Adamax':
        lr = 2e-3 # default: 2e-3 – learning rate
        betas = (0.9, 0.999) # default: (0.9, 0.999) – coefficients used for computing running averages of gradient and its square
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        weight_decay = 0 # default: 0 – weight decay (L2 penalty) 
    elif opt == 'ASGD':
        lr = 0.01 # default: 1e-2 – learning rate 
        lambd = 1e-4 # default: 1e-4 – decay term
        alpha = 0.75 # default: 0.75 – power for eta update
        t0 = 1e6 # default: 1e6 – point at which to start averaging
        weight_decay = 0 # default: 0 – weight decay
    elif opt == 'RMSprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        momentum = 0 # default: 0 – momentum factor
        alpha = 0.99 # default: 0.99 – smoothing constant
        eps = 1e-8 # default: 1e-8 – term added to the denominator to improve numerical stability
        centered = False # default: False – if True, compute the centered RMSProp, the gradient is normalized by an estimation of its variance
        weight_decay = 0 # default: 0 – weight decay (L2 penalty)
    elif opt == 'Rprop':
        lr = 1e-2 # default: 1e-2 – learning rate
        etas = (0.5, 1.2) # default: (0.5, 1.2) – pair of (etaminus, etaplis), that are multiplicative increase and decrease factors
        step_sizes = (1e-6, 50) # default: (1e-6, 50) – a pair of minimal and maximal allowed step sizes 
    elif opt == 'SGD':
        lr = 0.01 # – learning rate
        momentum = 0.9 # default: 0 – momentum factor 
        weight_decay = 0.05 # default: 0 – weight decay (L2 penalty) 
        dampening = 0 # default: 0 – dampening for momentum
        nesterov = False # default: False – enables Nesterov momentum 
    
    sch = 'CosineAnnealingLR'
    if sch == 'StepLR':
        step_size = epochs // 5 # – Period of learning rate decay.
        gamma = 0.5 # – Multiplicative factor of learning rate decay. Default: 0.1
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'MultiStepLR':
        milestones = [60, 120, 150] # – List of epoch indices. Must be increasing.
        gamma = 0.1 # – Multiplicative factor of learning rate decay. Default: 0.1.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'ExponentialLR':
        gamma = 0.99 #  – Multiplicative factor of learning rate decay.
        last_epoch = -1 # – The index of last epoch. Default: -1.
    elif sch == 'CosineAnnealingLR':
        T_max = 50 # – Maximum number of iterations. Cosine function period.
        eta_min = 0.00001 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1.  
    elif sch == 'ReduceLROnPlateau':
        mode = 'min' # – One of min, max. In min mode, lr will be reduced when the quantity monitored has stopped decreasing; in max mode it will be reduced when the quantity monitored has stopped increasing. Default: ‘min’.
        factor = 0.1 # – Factor by which the learning rate will be reduced. new_lr = lr * factor. Default: 0.1.
        patience = 10 # – Number of epochs with no improvement after which learning rate will be reduced. For example, if patience = 2, then we will ignore the first 2 epochs with no improvement, and will only decrease the LR after the 3rd epoch if the loss still hasn’t improved then. Default: 10.
        threshold = 0.0001 # – Threshold for measuring the new optimum, to only focus on significant changes. Default: 1e-4.
        threshold_mode = 'rel' # – One of rel, abs. In rel mode, dynamic_threshold = best * ( 1 + threshold ) in ‘max’ mode or best * ( 1 - threshold ) in min mode. In abs mode, dynamic_threshold = best + threshold in max mode or best - threshold in min mode. Default: ‘rel’.
        cooldown = 0 # – Number of epochs to wait before resuming normal operation after lr has been reduced. Default: 0.
        min_lr = 0 # – A scalar or a list of scalars. A lower bound on the learning rate of all param groups or each group respectively. Default: 0.
        eps = 1e-08 # – Minimal decay applied to lr. If the difference between new and old lr is smaller than eps, the update is ignored. Default: 1e-8.
    elif sch == 'CosineAnnealingWarmRestarts':
        T_0 = 50 # – Number of iterations for the first restart.
        T_mult = 2 # – A factor increases T_{i} after a restart. Default: 1.
        eta_min = 1e-6 # – Minimum learning rate. Default: 0.
        last_epoch = -1 # – The index of last epoch. Default: -1. 
    elif sch == 'WP_MultiStepLR':
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == 'WP_CosineLR':
        warm_up_epochs = 20
