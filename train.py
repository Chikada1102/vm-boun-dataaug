import torch
from torch.utils.data import DataLoader
import timm
from datasets.dataset import NPY_datasets
from tensorboardX import SummaryWriter
from models.vmunet.vmunet import VMUNet

from engine import *
import os
import sys

from utils import *
from configs.config_setting import setting_config

import warnings
warnings.filterwarnings("ignore")


def _make_dataloader(dataset, config, train=True):
    kwargs = dict(
        batch_size=config.batch_size if train else 1,
        shuffle=train,
        pin_memory=True,
        num_workers=config.num_workers,
        drop_last=False if not train else False,
    )
    # ==================== 新添加的代码开始：num_workers>0 时启用预取，不影响结构但提升读取边界标签速度 ====================
    if config.num_workers > 0:
        kwargs['persistent_workers'] = True
        kwargs['prefetch_factor'] = getattr(config, 'prefetch_factor', 2)
    # ==================== 新添加的代码结束：num_workers>0 时启用预取，不影响结构但提升读取边界标签速度 ====================
    return DataLoader(dataset, **kwargs)


def main(config):
    print('#----------Creating logger----------#')
    sys.path.append(config.work_dir + '/')
    log_dir = os.path.join(config.work_dir, 'log')
    checkpoint_dir = os.path.join(config.work_dir, 'checkpoints')
    resume_model = os.path.join(checkpoint_dir, 'latest.pth')
    outputs = os.path.join(config.work_dir, 'outputs')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(outputs):
        os.makedirs(outputs)

    global logger
    logger = get_logger('train', log_dir)
    global writer
    writer = SummaryWriter(config.work_dir + 'summary')

    log_config_info(config, logger)

    print('#----------GPU init----------#')
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    set_seed(config.seed)
    torch.cuda.empty_cache()

    print('#----------Preparing dataset----------#')
    train_dataset = NPY_datasets(config.data_path, config, train=True)
    train_loader = _make_dataloader(train_dataset, config, train=True)
    val_dataset = NPY_datasets(config.data_path, config, train=False)
    val_loader = _make_dataloader(val_dataset, config, train=False)

    print('#----------Prepareing Model----------#')
    model_cfg = config.model_config
    if config.network == 'vmunet':
        # 【原始代码删除说明】
        # 原始版本没有传 use_ibr_guidance：
        # model = VMUNet(..., load_ckpt_path=model_cfg['load_ckpt_path'])
        # 现在必须显式把配置传入模型；如果模型没有用上，forward/loss 会直接报错。
        model = VMUNet(
            num_classes=model_cfg['num_classes'],
            input_channels=model_cfg['input_channels'],
            depths=model_cfg['depths'],
            depths_decoder=model_cfg['depths_decoder'],
            drop_path_rate=model_cfg['drop_path_rate'],
            load_ckpt_path=model_cfg['load_ckpt_path'],
            # ==================== 新添加的代码开始：把IBR 边界细化开关真正传入模型 ====================
            use_ibr_guidance=model_cfg.get('use_ibr_guidance', False),
            ibr_kernel_size=model_cfg.get('ibr_kernel_size', 5),
            # ==================== 新添加的代码结束：把 IBR 边界细化开关和形态学核真正传入模型 ====================
        )
        model.load_from()
    else:
        raise Exception('network in not right!')

    if getattr(config, 'use_ibr_guidance', False) != model_cfg.get('use_ibr_guidance', False):
        raise RuntimeError('config.use_ibr_guidance 与 model_config[use_ibr_guidance] 不一致，禁止训练配置和模型结构脱节。')

    model = model.cuda()
    cal_params_flops(model, 256, logger)

    print('#----------Prepareing loss, opt, sch and amp----------#')
    criterion = config.criterion
    optimizer = get_optimizer(config, model)
    scheduler = get_scheduler(config, optimizer)

    print('#----------Set other params----------#')
    min_loss = 999
    start_epoch = 1
    min_epoch = 1

    if config.only_test_and_save_figs:
        checkpoint = torch.load(config.best_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint)
        config.work_dir = config.img_save_path
        if not os.path.exists(config.work_dir + 'outputs/'):
            os.makedirs(config.work_dir + 'outputs/')
        loss = test_one_epoch(val_loader, model, criterion, logger, config)
        return

    if os.path.exists(resume_model):
        print('#----------Resume Model and Other params----------#')
        checkpoint = torch.load(resume_model, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        saved_epoch = checkpoint['epoch']
        start_epoch += saved_epoch
        min_loss, min_epoch, loss = checkpoint['min_loss'], checkpoint['min_epoch'], checkpoint['loss']
        log_info = f'resuming model from {resume_model}. resume_epoch: {saved_epoch}, min_loss: {min_loss:.4f}, min_epoch: {min_epoch}, loss: {loss:.4f}'
        logger.info(log_info)

    step = 0
    print('#----------Training----------#')
    for epoch in range(start_epoch, config.epochs + 1):
        torch.cuda.empty_cache()
        step = train_one_epoch(train_loader, model, criterion, optimizer, scheduler, epoch, step, logger, config, writer)
        loss = val_one_epoch(val_loader, model, criterion, epoch, logger, config)

        if loss < min_loss:
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best.pth'))
            min_loss = loss
            min_epoch = epoch

        torch.save({
            'epoch': epoch,
            'min_loss': min_loss,
            'min_epoch': min_epoch,
            'loss': loss,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, os.path.join(checkpoint_dir, 'latest.pth'))

    if os.path.exists(os.path.join(checkpoint_dir, 'best.pth')):
        print('#----------Testing----------#')
        best_weight = torch.load(config.work_dir + 'checkpoints/best.pth', map_location=torch.device('cpu'))
        model.load_state_dict(best_weight)
        loss = test_one_epoch(val_loader, model, criterion, logger, config)
        os.rename(
            os.path.join(checkpoint_dir, 'best.pth'),
            os.path.join(checkpoint_dir, f'best-epoch{min_epoch}-loss{min_loss:.4f}.pth')
        )


if __name__ == '__main__':
    config = setting_config
    main(config)
