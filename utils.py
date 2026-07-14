import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import numpy as np
import os
import math
import random
import logging
import logging.handlers
from matplotlib import pyplot as plt


# ==================== 原始工具函数：保留 ====================
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True


def get_logger(name, log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if len(logger.handlers) == 0:
        info_name = os.path.join(log_dir, '{}.info.log'.format(name))
        info_handler = logging.handlers.TimedRotatingFileHandler(info_name, when='D', encoding='utf-8')
        info_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        info_handler.setFormatter(formatter)
        logger.addHandler(info_handler)
    return logger


def log_config_info(config, logger):
    config_dict = config.__dict__
    logger.info('#----------Config info----------#')
    for k, v in config_dict.items():
        if k[0] == '_':
            continue
        logger.info(f'{k}: {v},')


def get_optimizer(config, model):
    assert config.opt in ['Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'ASGD', 'RMSprop', 'Rprop', 'SGD'], 'Unsupported optimizer!'
    if config.opt == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                 weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    if config.opt == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                weight_decay=config.weight_decay, amsgrad=config.amsgrad)
    if config.opt == 'SGD':
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum,
                               weight_decay=config.weight_decay, dampening=config.dampening,
                               nesterov=config.nesterov)
    if config.opt == 'Adadelta':
        return torch.optim.Adadelta(model.parameters(), lr=config.lr, rho=config.rho, eps=config.eps,
                                    weight_decay=config.weight_decay)
    if config.opt == 'Adagrad':
        return torch.optim.Adagrad(model.parameters(), lr=config.lr, lr_decay=config.lr_decay, eps=config.eps,
                                   weight_decay=config.weight_decay)
    if config.opt == 'Adamax':
        return torch.optim.Adamax(model.parameters(), lr=config.lr, betas=config.betas, eps=config.eps,
                                  weight_decay=config.weight_decay)
    if config.opt == 'ASGD':
        return torch.optim.ASGD(model.parameters(), lr=config.lr, lambd=config.lambd, alpha=config.alpha,
                                t0=config.t0, weight_decay=config.weight_decay)
    if config.opt == 'RMSprop':
        return torch.optim.RMSprop(model.parameters(), lr=config.lr, momentum=config.momentum, alpha=config.alpha,
                                   eps=config.eps, centered=config.centered, weight_decay=config.weight_decay)
    if config.opt == 'Rprop':
        return torch.optim.Rprop(model.parameters(), lr=config.lr, etas=config.etas, step_sizes=config.step_sizes)
    raise RuntimeError('Unsupported optimizer branch')


def get_scheduler(config, optimizer):
    assert config.sch in ['StepLR', 'MultiStepLR', 'ExponentialLR', 'CosineAnnealingLR', 'ReduceLROnPlateau',
                          'CosineAnnealingWarmRestarts', 'WP_MultiStepLR', 'WP_CosineLR'], 'Unsupported scheduler!'
    if config.sch == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max,
                                                          eta_min=config.eta_min, last_epoch=config.last_epoch)
    if config.sch == 'StepLR':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma,
                                               last_epoch=config.last_epoch)
    if config.sch == 'MultiStepLR':
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma,
                                                    last_epoch=config.last_epoch)
    if config.sch == 'ExponentialLR':
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.gamma, last_epoch=config.last_epoch)
    if config.sch == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config.T_0,
                                                                    T_mult=config.T_mult,
                                                                    eta_min=config.eta_min,
                                                                    last_epoch=config.last_epoch)
    if config.sch == 'WP_MultiStepLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else config.gamma ** len(
            [m for m in config.milestones if m <= epoch])
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    if config.sch == 'WP_CosineLR':
        lr_func = lambda epoch: epoch / config.warm_up_epochs if epoch <= config.warm_up_epochs else 0.5 * (
            math.cos((epoch - config.warm_up_epochs) / (config.epochs - config.warm_up_epochs) * math.pi) + 1)
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    if config.sch == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=config.mode, factor=config.factor,
                                                          patience=config.patience, threshold=config.threshold,
                                                          threshold_mode=config.threshold_mode,
                                                          cooldown=config.cooldown, min_lr=config.min_lr,
                                                          eps=config.eps)
    raise RuntimeError('Unsupported scheduler branch')


def save_imgs(img, msk, msk_pred, i, save_path, datasets, threshold=0.5, test_data_name=None):
    img = img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    img = img / 255. if img.max() > 1.1 else img
    if datasets == 'retinal':
        msk = np.squeeze(msk, axis=0)
        msk_pred = np.squeeze(msk_pred, axis=0)
    else:
        att_msk = np.squeeze(msk_pred, axis=0)
        msk = np.where(np.squeeze(msk, axis=0) > 0.5, 1, 0)
        msk_pred = np.where(np.squeeze(msk_pred, axis=0) > threshold, 1, 0)

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.figure(figsize=(10, 20))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.subplot(4, 1, 1); plt.imshow(img); plt.axis('off'); plt.title('Original Image')
    plt.subplot(4, 1, 2); plt.imshow(msk, cmap='gray'); plt.axis('off'); plt.title('Ground Truth Mask')
    plt.subplot(4, 1, 3); plt.imshow(msk_pred, cmap='gray'); plt.axis('off'); plt.title('Predicted Mask')
    plt.subplot(4, 1, 4); plt.imshow(img); plt.imshow(att_msk, cmap='jet', alpha=0.5); plt.axis('off'); plt.title('Attention Map Overlay')
    if test_data_name is not None:
        save_path = save_path + test_data_name + '_'
    plt.savefig(save_path + str(i) + '.png', bbox_inches='tight', pad_inches=0)
    plt.close()


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.bceloss = nn.BCELoss()

    def forward(self, pred, target):
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        return self.bceloss(pred_, target_)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1
        size = pred.size(0)
        pred_ = pred.view(size, -1)
        target_ = target.view(size, -1)
        intersection = pred_ * target_
        dice_score = (2 * intersection.sum(1) + smooth) / (pred_.sum(1) + target_.sum(1) + smooth)
        return 1 - dice_score.sum() / size


class BceDiceLoss(nn.Module):
    def __init__(self, wb=1, wd=1):
        super(BceDiceLoss, self).__init__()
        self.bce = BCELoss()
        self.dice = DiceLoss()
        self.wb = wb
        self.wd = wd

    def forward(self, pred, target):
        bceloss = self.bce(pred, target)
        diceloss = self.dice(pred, target)
        return self.wd * diceloss + self.wb * bceloss


# ==================== 新添加的代码开始：IBR 边界细化损失，强制检查新增结构被使用 ====================
class IBRBoundaryRefinementLoss(nn.Module):
    """
    IBR-VMUNet 训练损失。

    必须同时使用：
        final segmentation output
        coarse segmentation output
        boundary output
        outer ring map
        stable background map
        foreground/background/contrast similarity maps
        IBR / foreground prototype / background prototype / contrast / interactive attention 标志

    如果模型没有返回这些字段，直接报错，防止“新结构没调用也继续训练”。
    """
    def __init__(self,
                 wb=1,
                 wd=1,
                 coarse_weight=0.25,
                 boundary_weight=0.25,
                 outer_suppression_weight=0.05,
                 # ==================== 新添加的代码开始：background prototype 对比约束权重 ====================
                 background_suppression_weight=0.05,
                 contrast_boundary_weight=0.05):
                 # ==================== 新添加的代码结束：background prototype 对比约束权重 ====================
        super(IBRBoundaryRefinementLoss, self).__init__()
        self.seg_loss = BceDiceLoss(wb, wd)
        self.boundary_loss = BceDiceLoss(wb, wd)
        self.coarse_weight = coarse_weight
        self.boundary_weight = boundary_weight
        self.outer_suppression_weight = outer_suppression_weight
        # ==================== 新添加的代码开始：保存 background prototype 对比约束权重 ====================
        self.background_suppression_weight = background_suppression_weight
        self.contrast_boundary_weight = contrast_boundary_weight
        # ==================== 新添加的代码结束：保存 background prototype 对比约束权重 ====================

    def _check_map(self, outputs, key, ref_shape):
        if key not in outputs:
            raise RuntimeError(f'IBRBoundaryRefinementLoss 要求模型输出 {key}，但没有收到，禁止忽略新增结构。')
        item = outputs[key]
        if item is None:
            raise RuntimeError(f'模型输出 {key}=None，说明新增结构没有有效返回。')
        if item.shape != ref_shape:
            raise RuntimeError(f'{key} 形状错误：{item.shape}，期望 {ref_shape}。')
        if not torch.isfinite(item).all():
            raise RuntimeError(f'{key} 中存在 NaN/Inf，IBR 结构异常。')
        return item

    def forward(self, outputs, target, boundary):
        if not isinstance(outputs, dict):
            raise RuntimeError('IBRBoundaryRefinementLoss 要求模型返回 dict，但收到普通 tensor，说明 IBR 结构没有被调用。')

        # 【原始代码删除说明】
        # 上一版只要求 foreground_similarity_map，无法确认 background prototype 和 contrast guidance 被使用。
        # required_keys = [
        #     'seg', 'coarse_seg', 'boundary',
        #     'interior_map', 'outer_ring_map', 'boundary_candidate_map', 'foreground_similarity_map',
        #     'ibr_guidance_used', 'prototype_guidance_used', 'interactive_attention_used'
        # ]
        # ==================== 新添加的代码开始：loss 强制消费 background/contrast 输出与调用标志 ====================
        required_keys = [
            'seg', 'coarse_seg', 'boundary',
            'interior_map', 'outer_ring_map', 'stable_background_map', 'boundary_candidate_map',
            'foreground_similarity_map', 'background_similarity_map', 'contrast_similarity_map',
            'ibr_guidance_used', 'prototype_guidance_used',
            'background_prototype_guidance_used', 'contrast_guidance_used', 'interactive_attention_used'
        ]
        # ==================== 新添加的代码结束：loss 强制消费 background/contrast 输出与调用标志 ====================
        for key in required_keys:
            if key not in outputs:
                raise RuntimeError(f'模型输出缺少 {key}，禁止忽略 IBR 新增结构。')

        if outputs['ibr_guidance_used'] is not True:
            raise RuntimeError('模型返回 ibr_guidance_used=False，说明 IBR 结构未实际参与 forward。')
        if outputs['prototype_guidance_used'] is not True:
            raise RuntimeError('模型返回 prototype_guidance_used=False，说明 foreground prototype 未实际参与 forward。')
        # ==================== 新添加的代码开始：background prototype 与 contrast guidance 标志检查 ====================
        if outputs['background_prototype_guidance_used'] is not True:
            raise RuntimeError('模型返回 background_prototype_guidance_used=False，说明 background prototype 未实际参与 forward。')
        if outputs['contrast_guidance_used'] is not True:
            raise RuntimeError('模型返回 contrast_guidance_used=False，说明前景-背景对比 guidance 未实际参与 forward。')
        # ==================== 新添加的代码结束：background prototype 与 contrast guidance 标志检查 ====================
        if outputs['interactive_attention_used'] is not True:
            raise RuntimeError('模型返回 interactive_attention_used=False，说明 interactive attention 未实际参与 forward。')

        seg = outputs['seg']
        coarse_seg = outputs['coarse_seg']
        boundary_pred = outputs['boundary']

        if seg.shape != target.shape:
            raise RuntimeError(f'final seg 和 target 形状不一致：seg={seg.shape}, target={target.shape}')
        if coarse_seg.shape != target.shape:
            raise RuntimeError(f'coarse seg 和 target 形状不一致：coarse={coarse_seg.shape}, target={target.shape}')
        if boundary_pred.shape != boundary.shape:
            raise RuntimeError(f'boundary 输出和 boundary 标签形状不一致：pred={boundary_pred.shape}, label={boundary.shape}')

        interior_map = self._check_map(outputs, 'interior_map', target.shape)
        outer_ring_map = self._check_map(outputs, 'outer_ring_map', target.shape)
        # ==================== 新添加的代码开始：检查 background/contrast 结构图确实返回并参与 loss ====================
        stable_background_map = self._check_map(outputs, 'stable_background_map', target.shape)
        # ==================== 新添加的代码结束：检查 background/contrast 结构图确实返回并参与 loss ====================
        boundary_candidate_map = self._check_map(outputs, 'boundary_candidate_map', target.shape)
        foreground_similarity_map = self._check_map(outputs, 'foreground_similarity_map', target.shape)
        # ==================== 新添加的代码开始：检查 background/contrast similarity map ====================
        background_similarity_map = self._check_map(outputs, 'background_similarity_map', target.shape)
        contrast_similarity_map = self._check_map(outputs, 'contrast_similarity_map', target.shape)
        # ==================== 新添加的代码结束：检查 background/contrast similarity map ====================

        # 这些 map 已在 forward 中参与 final mask 生成；这里检查 shape/finite。
        # 不强制要求均值大于 0，因为训练早期 coarse mask 可能接近常数，
        # 但只要字段存在且参与计算，就不允许静默忽略。

        loss_final = self.seg_loss(seg, target)
        loss_coarse = self.seg_loss(coarse_seg, target)
        loss_boundary = self.boundary_loss(boundary_pred, boundary)

        # 【原始代码删除说明】
        # 上一版 outer loss 只利用 foreground_similarity_map：
        # outer_weight = outer_ring_map.detach() * (1.0 - foreground_similarity_map.detach()).clamp(0.0, 1.0)
        # loss_outer = (seg * outer_weight).mean()
        # return loss_final + ... + self.outer_suppression_weight * loss_outer
        # ==================== 新添加的代码开始：前景-背景对比式外环/背景抑制损失 ====================
        # outer ring 中“更像背景、不像前景”的区域对 FP 惩罚更强。
        outer_weight = (
            outer_ring_map.detach()
            * background_similarity_map.detach().clamp(0.0, 1.0)
            * (1.0 - contrast_similarity_map.detach()).clamp(0.0, 1.0)
        )
        loss_outer = (seg * outer_weight).mean()

        # stable background 区域不应被 final seg 激活，用轻量项进一步压背景 FP。
        background_weight = stable_background_map.detach() * background_similarity_map.detach().clamp(0.0, 1.0)
        loss_background = (seg * background_weight).mean()

        # boundary_candidate 区域中，contrast 越低越像背景，预测为前景的代价越高。
        contrast_boundary_weight = boundary_candidate_map.detach() * (1.0 - contrast_similarity_map.detach()).clamp(0.0, 1.0)
        loss_contrast_boundary = (seg * contrast_boundary_weight).mean()

        return (loss_final
                + self.coarse_weight * loss_coarse
                + self.boundary_weight * loss_boundary
                + self.outer_suppression_weight * loss_outer
                + self.background_suppression_weight * loss_background
                + self.contrast_boundary_weight * loss_contrast_boundary)
        # ==================== 新添加的代码结束：前景-背景对比式外环/背景抑制损失 ====================
# ==================== 新添加的代码结束：IBR 边界细化损失，强制检查新增结构被使用 ====================


# ==================== 新添加的代码开始：支持 image/mask/boundary 三元组的同步变换 ====================
def _is_boundary_tuple(data):
    return isinstance(data, (tuple, list)) and len(data) == 3


class myToTensor:
    def __init__(self):
        pass

    def __call__(self, data):
        if _is_boundary_tuple(data):
            image, mask, boundary = data
            return (torch.tensor(image).permute(2, 0, 1).float(),
                    torch.tensor(mask).permute(2, 0, 1).float(),
                    torch.tensor(boundary).permute(2, 0, 1).float())
        # 【原始代码保留】原始版本只处理 image/mask 二元组。
        image, mask = data
        return torch.tensor(image).permute(2, 0, 1).float(), torch.tensor(mask).permute(2, 0, 1).float()


class myResize:
    def __init__(self, size_h=256, size_w=256):
        self.size_h = size_h
        self.size_w = size_w

    def __call__(self, data):
        if _is_boundary_tuple(data):
            image, mask, boundary = data
            return (TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR),
                    TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST),
                    TF.resize(boundary, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST))
        image, mask = data
        return (TF.resize(image, [self.size_h, self.size_w], interpolation=InterpolationMode.BILINEAR),
                TF.resize(mask, [self.size_h, self.size_w], interpolation=InterpolationMode.NEAREST))


class myRandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() >= self.p:
            return data
        if _is_boundary_tuple(data):
            image, mask, boundary = data
            return TF.hflip(image), TF.hflip(mask), TF.hflip(boundary)
        image, mask = data
        return TF.hflip(image), TF.hflip(mask)


class myRandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, data):
        if random.random() >= self.p:
            return data
        if _is_boundary_tuple(data):
            image, mask, boundary = data
            return TF.vflip(image), TF.vflip(mask), TF.vflip(boundary)
        image, mask = data
        return TF.vflip(image), TF.vflip(mask)


class myRandomRotation:
    def __init__(self, p=0.5, degree=[0, 360]):
        self.degree = degree
        self.p = p

    def __call__(self, data):
        if random.random() >= self.p:
            return data
        angle = random.uniform(self.degree[0], self.degree[1])
        if _is_boundary_tuple(data):
            image, mask, boundary = data
            return (TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR),
                    TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST),
                    TF.rotate(boundary, angle, interpolation=InterpolationMode.NEAREST))
        image, mask = data
        return (TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR),
                TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST))


class myNormalize:
    def __init__(self, data_name, train=True):
        if data_name == 'isic18':
            self.mean = 157.561 if train else 149.034
            self.std = 26.706 if train else 32.022
        elif data_name == 'isic17':
            self.mean = 159.922 if train else 148.429
            self.std = 28.871 if train else 25.748
        elif data_name == 'isic18_82':
            self.mean = 156.2899 if train else 149.8485
            self.std = 26.5457 if train else 35.3346
        else:
            raise RuntimeError(f'未知数据集 {data_name}，请在 myNormalize 中补充 mean/std。')

    def __call__(self, data):
        if _is_boundary_tuple(data):
            img, msk, boundary = data
            img_normalized = (img - self.mean) / self.std
            img_normalized = ((img_normalized - np.min(img_normalized)) /
                              (np.max(img_normalized) - np.min(img_normalized) + 1e-8)) * 255.
            return img_normalized, msk, boundary
        img, msk = data
        img_normalized = (img - self.mean) / self.std
        img_normalized = ((img_normalized - np.min(img_normalized)) /
                          (np.max(img_normalized) - np.min(img_normalized) + 1e-8)) * 255.
        return img_normalized, msk
# ==================== 新添加的代码结束：支持 image/mask/boundary 三元组的同步变换 ====================


from thop import profile


class _MainOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, dict):
            if 'seg' not in out:
                raise RuntimeError('cal_params_flops 遇到 dict 输出但缺少 seg。')
            return out['seg']
        return out


def cal_params_flops(model, size, logger):
    input_tensor = torch.randn(1, 3, size, size).cuda()
    # ==================== 新添加的代码开始：thop 只统计主输出，同时保持模型真实 forward 被调用 ====================
    wrapped_model = _MainOutputWrapper(model)
    flops, params = profile(wrapped_model, inputs=(input_tensor,))
    # ==================== 新添加的代码结束：thop 只统计主输出，同时保持模型真实 forward 被调用 ====================
    print('flops', flops / 1e9)
    print('params', params / 1e6)
    total = sum(p.numel() for p in model.parameters())
    print("Total params: %.2fM" % (total / 1e6))
    logger.info(f'flops: {flops/1e9}, params: {params/1e6}, Total params: : {total/1e6:.4f}')
