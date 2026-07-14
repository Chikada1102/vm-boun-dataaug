from .vmamba import VSSM
import torch
from torch import nn

# ==================== 新添加的代码开始：导入 IBR 边界细化 VSSM ====================
from .vmamba_ibr import IBRRefinementVSSM
# ==================== 新添加的代码结束：导入 IBR 边界细化 VSSM ====================


class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None,
                 # ==================== 新添加的代码开始：IBR 结构开关 ====================
                 use_ibr_guidance=False,
                 ibr_kernel_size=5,
                 # ==================== 新添加的代码结束：IBR 结构开关 ====================
                ):
        super().__init__()

        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        # ==================== 新添加的代码开始：记录 IBR 结构开关 ====================
        self.use_ibr_guidance = use_ibr_guidance
        # ==================== 新添加的代码结束：记录 IBR 结构开关 ====================

        # 【原始代码删除说明】
        # 原始版本直接使用 VSSM：
        # self.vmunet = VSSM(in_chans=input_channels,
        #                    num_classes=num_classes,
        #                    depths=depths,
        #                    depths_decoder=depths_decoder,
        #                    drop_path_rate=drop_path_rate)
        # 现在保留 baseline 分支；当 use_ibr_guidance=True 时必须使用 IBRRefinementVSSM。
        if self.use_ibr_guidance:
            # ==================== 新添加的代码开始：真正调用 IBR 内部区域引导边界细化解码器 ====================
            self.vmunet = IBRRefinementVSSM(in_chans=input_channels,
                                           num_classes=num_classes,
                                           depths=depths,
                                           depths_decoder=depths_decoder,
                                           drop_path_rate=drop_path_rate,
                                           ibr_kernel_size=ibr_kernel_size)
            # ==================== 新添加的代码结束：真正调用 IBR 内部区域引导边界细化解码器 ====================
        else:
            self.vmunet = VSSM(in_chans=input_channels,
                               num_classes=num_classes,
                               depths=depths,
                               depths_decoder=depths_decoder,
                               drop_path_rate=drop_path_rate)

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        raw = self.vmunet(x)

        # ==================== 新添加的代码开始：严格检查 IBR 新结构是否被调用 ====================
        if self.use_ibr_guidance:
            if not isinstance(raw, dict):
                raise RuntimeError('use_ibr_guidance=True，但 VSSM 没有返回 dict，说明 IBR 结构没有被调用。')
            # 【原始代码删除说明】
            # 上一版 required_keys 只要求 foreground_similarity_map：
            # required_keys = [
            #     'final_logits', 'coarse_logits', 'boundary_logits',
            #     'interior_map', 'outer_ring_map', 'boundary_candidate_map', 'foreground_similarity_map',
            #     'ibr_guidance_used', 'prototype_guidance_used', 'interactive_attention_used'
            # ]
            # ==================== 新添加的代码开始：强制检查 background prototype 与 contrast guidance 输出 ====================
            required_keys = [
                'final_logits', 'coarse_logits', 'boundary_logits',
                'interior_map', 'outer_ring_map', 'stable_background_map', 'boundary_candidate_map',
                'foreground_similarity_map', 'background_similarity_map', 'contrast_similarity_map',
                'ibr_guidance_used', 'prototype_guidance_used',
                'background_prototype_guidance_used', 'contrast_guidance_used', 'interactive_attention_used'
            ]
            # ==================== 新添加的代码结束：强制检查 background prototype 与 contrast guidance 输出 ====================
            for key in required_keys:
                if key not in raw:
                    raise RuntimeError(f'use_ibr_guidance=True，但模型输出缺少 {key}，禁止忽略新增结构。')
            if raw['ibr_guidance_used'] is not True:
                raise RuntimeError('IBRRefinementVSSM 返回 ibr_guidance_used=False，说明 IBR 结构未实际使用。')
            if raw['prototype_guidance_used'] is not True:
                raise RuntimeError('foreground prototype guidance 未实际使用。')
            # ==================== 新添加的代码开始：background prototype 与 contrast guidance 调用检查 ====================
            if raw['background_prototype_guidance_used'] is not True:
                raise RuntimeError('background prototype guidance 未实际使用。')
            if raw['contrast_guidance_used'] is not True:
                raise RuntimeError('foreground-background contrast guidance 未实际使用。')
            # ==================== 新添加的代码结束：background prototype 与 contrast guidance 调用检查 ====================
            if raw['interactive_attention_used'] is not True:
                raise RuntimeError('segmentation/boundary interactive attention 未实际使用。')

            final_logits = raw['final_logits']
            coarse_logits = raw['coarse_logits']
            if self.num_classes == 1:
                seg = torch.sigmoid(final_logits)
                coarse_seg = torch.sigmoid(coarse_logits)
            else:
                seg = final_logits
                coarse_seg = coarse_logits

            boundary = torch.sigmoid(raw['boundary_logits'])
            if seg.shape != coarse_seg.shape or seg.shape != boundary.shape:
                raise RuntimeError(f'IBR 输出尺寸不一致：seg={seg.shape}, coarse={coarse_seg.shape}, boundary={boundary.shape}')

            return {
                'seg': seg,
                'logits': final_logits,
                'coarse_seg': coarse_seg,
                'coarse_logits': coarse_logits,
                'boundary': boundary,
                'boundary_logits': raw['boundary_logits'],
                'interior_map': raw['interior_map'],
                'inner_boundary_map': raw.get('inner_boundary_map', None),
                'outer_ring_map': raw['outer_ring_map'],
                # ==================== 新添加的代码开始：向 loss 暴露 background/contrast 结构图 ====================
                'stable_background_map': raw['stable_background_map'],
                # ==================== 新添加的代码结束：向 loss 暴露 background/contrast 结构图 ====================
                'boundary_candidate_map': raw['boundary_candidate_map'],
                'foreground_similarity_map': raw['foreground_similarity_map'],
                # ==================== 新添加的代码开始：向 loss 暴露 background/contrast 相似图 ====================
                'background_similarity_map': raw['background_similarity_map'],
                'contrast_similarity_map': raw['contrast_similarity_map'],
                # ==================== 新添加的代码结束：向 loss 暴露 background/contrast 相似图 ====================
                'ibr_guidance_used': True,
                'prototype_guidance_used': True,
                # ==================== 新添加的代码开始：新增结构调用标志继续向外传递 ====================
                'background_prototype_guidance_used': True,
                'contrast_guidance_used': True,
                # ==================== 新添加的代码结束：新增结构调用标志继续向外传递 ====================
                'interactive_attention_used': True,
            }
        # ==================== 新添加的代码结束：严格检查 IBR 新结构是否被调用 ====================

        logits = raw
        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    def load_from(self):
        if self.load_ckpt_path is not None:
            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_dict = modelCheckpoint['model']
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("encoder loaded finished!")

            model_dict = self.vmunet.state_dict()
            modelCheckpoint = torch.load(self.load_ckpt_path)
            pretrained_odict = modelCheckpoint['model']
            pretrained_dict = {}
            for k, v in pretrained_odict.items():
                if 'layers.0' in k:
                    new_k = k.replace('layers.0', 'layers_up.3')
                    pretrained_dict[new_k] = v
                elif 'layers.1' in k:
                    new_k = k.replace('layers.1', 'layers_up.2')
                    pretrained_dict[new_k] = v
                elif 'layers.2' in k:
                    new_k = k.replace('layers.2', 'layers_up.1')
                    pretrained_dict[new_k] = v
                elif 'layers.3' in k:
                    new_k = k.replace('layers.3', 'layers_up.0')
                    pretrained_dict[new_k] = v
            # 过滤操作
            new_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict.keys()}
            model_dict.update(new_dict)
            # 打印出来，更新了多少的参数
            print('Total model_dict: {}, Total pretrained_dict: {}, update: {}'.format(len(model_dict), len(pretrained_dict), len(new_dict)))
            self.vmunet.load_state_dict(model_dict)

            # 找到没有加载的键(keys)
            not_loaded_keys = [k for k in pretrained_dict.keys() if k not in new_dict.keys()]
            print('Not loaded keys:', not_loaded_keys)
            print("decoder loaded finished!")
