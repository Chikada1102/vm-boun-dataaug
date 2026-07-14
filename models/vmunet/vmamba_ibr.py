# -*- coding: utf-8 -*-
"""
IBR-VMUNet: Interior-guided Boundary Refinement for VM-UNet.

该文件继承原始 models/vmunet/vmamba.py 中的 VSSM，
不改原始 vmamba.py 文件本体，而是在子类里覆盖 forward，保证新增结构被实际调用。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vmamba import VSSM


# ==================== 新添加的代码开始：IBR 内部区域引导边界细化 VMamba 解码器 ====================
class IBRRefinementVSSM(VSSM):
    """
    Interior-guided Boundary Refinement VSSM.

    结构流程：
        1. 原始 VM-UNet decoder 先得到 decoder feature。
        2. 使用原始 final head 得到 coarse mask。
        3. 从 coarse mask 中通过可微形态学提取 interior / boundary / outer ring。
        4. 用 interior 区域在 decoder feature 上生成 foreground prototype。
        5. 计算 boundary feature 与 foreground prototype 的相似性，得到 prototype-aware boundary cue。
        6. segmentation branch 与 boundary branch 进行 interactive attention。
        7. 输出 final mask + boundary map。

    该版本刻意不使用上一版的 skip * (1 + boundary_att)，避免“看到边界就单向增强”。
    新增结构必须返回 ibr_guidance_used=True、prototype_guidance_used=True、interactive_attention_used=True，
    外层 VMUNet 和 loss 会严格检查，防止新增代码写了但没被调用。
    """

    def __init__(self, *args, ibr_kernel_size=5, **kwargs):
        super().__init__(*args, **kwargs)

        if getattr(self, 'num_classes', 1) != 1:
            raise RuntimeError('IBRRefinementVSSM 当前只针对二分类医学分割实现，num_classes 必须为 1。')
        if not hasattr(self, 'dims') or len(self.dims) < 1:
            raise RuntimeError('IBRRefinementVSSM 需要原始 VSSM 暴露 self.dims。')
        if ibr_kernel_size % 2 == 0 or ibr_kernel_size < 3:
            raise RuntimeError(f'ibr_kernel_size 必须是 >=3 的奇数，当前为 {ibr_kernel_size}。')

        self.ibr_kernel_size = int(ibr_kernel_size)
        feat_dim = int(self.dims[0])
        hidden_dim = max(feat_dim // 2, 16)

        self.seg_feature_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )
        self.boundary_feature_proj = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
        )

        # 【原始代码删除说明】
        # 上一版 IBR 的 structure_gate 只输入 5 个结构图：
        # coarse_prob / interior / boundary_candidate / outer_ring / foreground_similarity。
        # self.structure_gate = nn.Sequential(
        #     nn.Conv2d(5, feat_dim, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(feat_dim),
        #     nn.Sigmoid(),
        # )
        # ==================== 新添加的代码开始：加入 background prototype 与前景-背景对比结构图 ====================
        # 当前版本输入 8 个结构图：
        # coarse_prob / interior / boundary_candidate / outer_ring / stable_background
        # foreground_similarity / background_similarity / contrast_similarity。
        self.structure_gate = nn.Sequential(
            nn.Conv2d(8, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.Sigmoid(),
        )
        # ==================== 新添加的代码结束：加入 background prototype 与前景-背景对比结构图 ====================

        # segmentation branch 与 boundary branch 的双向交互注意力。
        self.boundary_to_seg_gate = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.seg_to_boundary_gate = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(feat_dim * 4, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.GELU(),
            nn.Conv2d(feat_dim, feat_dim, kernel_size=1, bias=True),
        )

        self.boundary_head = nn.Sequential(
            nn.Conv2d(feat_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1, bias=True),
        )

        self.prototype_scale = nn.Parameter(torch.tensor(5.0))
        # ==================== 新添加的代码开始：background prototype 对比校正参数 ====================
        self.background_scale = nn.Parameter(torch.tensor(5.0))
        self.contrast_scale = nn.Parameter(torch.tensor(5.0))
        self.background_suppression_strength = nn.Parameter(torch.tensor(0.5))
        # ==================== 新添加的代码结束：background prototype 对比校正参数 ====================
        self.interactive_strength = nn.Parameter(torch.tensor(0.5))

    def _assert_nchw_one_channel(self, name, tensor):
        if tensor.dim() != 4 or tensor.shape[1] != 1:
            raise RuntimeError(f'{name} 形状错误，期望 [B,1,H,W]，实际 {tuple(tensor.shape)}。')

    def _soft_morphology_from_prob(self, prob):
        """从 coarse probability 中可微提取 interior / boundary / outer ring。"""
        self._assert_nchw_one_channel('coarse_prob', prob)
        pad = self.ibr_kernel_size // 2
        dilated = F.max_pool2d(prob, kernel_size=self.ibr_kernel_size, stride=1, padding=pad)
        eroded = 1.0 - F.max_pool2d(1.0 - prob, kernel_size=self.ibr_kernel_size, stride=1, padding=pad)

        interior = eroded.clamp(0.0, 1.0)
        inner_boundary = (prob - eroded).clamp(0.0, 1.0)
        outer_ring = (dilated - prob).clamp(0.0, 1.0)
        boundary_candidate = (dilated - eroded).clamp(0.0, 1.0)
        # ==================== 新添加的代码开始：稳定背景区域，用于 background prototype ====================
        # stable_background 是 dilation 之后仍然确定为背景的区域；它比 outer_ring 更干净，
        # 用来提供“像背景”的 prototype，避免上一版只问边界像不像前景。
        stable_background = (1.0 - dilated).clamp(0.0, 1.0)
        # ==================== 新添加的代码结束：稳定背景区域，用于 background prototype ====================

        return {
            'interior_map': interior,
            'inner_boundary_map': inner_boundary,
            'outer_ring_map': outer_ring,
            'stable_background_map': stable_background,
            'boundary_candidate_map': boundary_candidate,
        }

    def _resize_map(self, tensor, size):
        self._assert_nchw_one_channel('structure_map', tensor)
        return F.interpolate(tensor, size=size, mode='bilinear', align_corners=True)

    def _masked_average_prototype(self, feat, mask):
        if feat.dim() != 4:
            raise RuntimeError(f'foreground prototype 需要 NCHW feature，实际 {tuple(feat.shape)}。')
        self._assert_nchw_one_channel('interior mask', mask)
        if feat.shape[-2:] != mask.shape[-2:]:
            raise RuntimeError(f'feature 与 interior mask 尺寸不一致：feat={tuple(feat.shape)}, mask={tuple(mask.shape)}。')
        denominator = mask.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        prototype = (feat * mask).sum(dim=(2, 3), keepdim=True) / denominator
        return prototype

    def forward(self, x):
        # 原始 VSSM.forward 的逻辑是：
        # x, skip_list = self.forward_features(x)
        # x = self.forward_features_up(x, skip_list)
        # x = self.forward_final(x)
        # return x
        # 这里不删除原始代码，而是在子类中重写 forward，使 IBR 结构真正参与最终分割。

        x, skip_list = self.forward_features(x)
        decoder_feat_nhwc = self.forward_features_up(x, skip_list)

        if decoder_feat_nhwc.dim() != 4:
            raise RuntimeError(f'decoder feature 应为 NHWC 4D tensor，实际 {tuple(decoder_feat_nhwc.shape)}。')

        # 1. coarse mask：先用原始 final head 得到粗分割。
        coarse_logits = self.forward_final(decoder_feat_nhwc)
        coarse_prob = torch.sigmoid(coarse_logits)
        self._assert_nchw_one_channel('coarse_logits', coarse_logits)

        # 2. 从 coarse mask 提取 interior / boundary / outer ring。
        structure_maps_full = self._soft_morphology_from_prob(coarse_prob)

        decoder_feat = decoder_feat_nhwc.permute(0, 3, 1, 2).contiguous()
        feat_size = decoder_feat.shape[-2:]

        coarse_prob_low = self._resize_map(coarse_prob, feat_size)
        interior_low = self._resize_map(structure_maps_full['interior_map'], feat_size)
        boundary_low = self._resize_map(structure_maps_full['boundary_candidate_map'], feat_size)
        outer_low = self._resize_map(structure_maps_full['outer_ring_map'], feat_size)
        # ==================== 新添加的代码开始：读取稳定背景区域，构建 background prototype ====================
        stable_background_low = self._resize_map(structure_maps_full['stable_background_map'], feat_size)
        # ==================== 新添加的代码结束：读取稳定背景区域，构建 background prototype ====================

        # 3. interior feature -> foreground prototype。
        foreground_prototype = self._masked_average_prototype(decoder_feat, interior_low)

        # 【原始代码删除说明】
        # 上一版只计算 boundary feature 与 foreground prototype 的相似性：
        # feat_norm = F.normalize(decoder_feat, dim=1, eps=1e-6)
        # proto_norm = F.normalize(foreground_prototype, dim=1, eps=1e-6)
        # foreground_similarity = (feat_norm * proto_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        # prototype_gate = torch.sigmoid(self.prototype_scale.clamp(1.0, 10.0) * foreground_similarity)
        # 这样仍然容易偏召回，因为边界像不像背景没有被显式建模。
        # ==================== 新添加的代码开始：foreground/background 双 prototype 对比校正 ====================
        # 4. stable background feature -> background prototype。
        background_prototype = self._masked_average_prototype(decoder_feat, stable_background_low)

        # 5. boundary feature 同时与 foreground/background prototypes 比较。
        feat_norm = F.normalize(decoder_feat, dim=1, eps=1e-6)
        fg_proto_norm = F.normalize(foreground_prototype, dim=1, eps=1e-6)
        bg_proto_norm = F.normalize(background_prototype, dim=1, eps=1e-6)

        foreground_similarity = (feat_norm * fg_proto_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        background_similarity = (feat_norm * bg_proto_norm).sum(dim=1, keepdim=True).clamp(-1.0, 1.0)
        contrast_similarity = (foreground_similarity - background_similarity).clamp(-2.0, 2.0)

        prototype_gate = torch.sigmoid(self.prototype_scale.clamp(1.0, 10.0) * foreground_similarity)
        background_gate = torch.sigmoid(self.background_scale.clamp(1.0, 10.0) * background_similarity)
        contrast_gate = torch.sigmoid(self.contrast_scale.clamp(1.0, 10.0) * contrast_similarity)
        # ==================== 新添加的代码结束：foreground/background 双 prototype 对比校正 ====================

        structural_input = torch.cat([
            coarse_prob_low,
            interior_low,
            boundary_low,
            outer_low,
            stable_background_low,
            prototype_gate,
            background_gate,
            contrast_gate,
        ], dim=1)
        if structural_input.shape[1] != 8:
            raise RuntimeError('IBR structural_input 通道数不是 8，说明 background prototype/contrast map 没有参与结构融合。')

        structure_gate = self.structure_gate(structural_input)

        seg_feat = self.seg_feature_proj(decoder_feat)

        # 【原始代码删除说明】
        # 上一版 prototype_mod 只基于 foreground similarity 和 outer ring：
        # prototype_mod = 1.0 + 0.5 * boundary_low * (prototype_gate - 0.5) - 0.5 * outer_low * (1.0 - prototype_gate)
        # prototype_mod = prototype_mod.clamp(0.5, 1.5)
        # boundary_feat = self.boundary_feature_proj(decoder_feat * prototype_mod)
        # 该做法仍然没有显式比较“更像前景还是更像背景”。
        # ==================== 新添加的代码开始：基于前景-背景对比的边界特征双向调制 ====================
        bg_strength = self.background_suppression_strength.clamp(0.0, 1.0)
        prototype_mod = (
            1.0
            + 0.5 * boundary_low * (contrast_gate - 0.5)
            - bg_strength * outer_low * background_gate
            - 0.5 * stable_background_low * background_gate
        )
        prototype_mod = prototype_mod.clamp(0.35, 1.5)
        boundary_feat = self.boundary_feature_proj(decoder_feat * prototype_mod)
        # ==================== 新添加的代码结束：基于前景-背景对比的边界特征双向调制 ====================

        # 5. boundary branch 和 segmentation branch 做 interactive attention。
        b2s = self.boundary_to_seg_gate(boundary_feat)
        s2b = self.seg_to_boundary_gate(seg_feat)
        strength = self.interactive_strength.clamp(0.0, 1.0)

        seg_interactive = seg_feat * (1.0 + strength * (b2s - 0.5)) + decoder_feat * structure_gate
        boundary_interactive = boundary_feat * (1.0 + strength * (s2b - 0.5)) + decoder_feat * (1.0 - structure_gate)

        final_feat = self.final_fusion(torch.cat([
            decoder_feat,
            seg_interactive,
            boundary_interactive,
            decoder_feat * structure_gate,
        ], dim=1))

        final_logits = self.forward_final(final_feat.permute(0, 2, 3, 1).contiguous())
        boundary_logits_low = self.boundary_head(boundary_interactive)
        boundary_logits = F.interpolate(boundary_logits_low, size=final_logits.shape[-2:], mode='bilinear', align_corners=True)

        if final_logits.shape != coarse_logits.shape:
            raise RuntimeError(f'final_logits 与 coarse_logits 尺寸不一致：final={tuple(final_logits.shape)}, coarse={tuple(coarse_logits.shape)}。')
        if boundary_logits.shape != final_logits.shape:
            raise RuntimeError(f'boundary_logits 与 final_logits 尺寸不一致：boundary={tuple(boundary_logits.shape)}, final={tuple(final_logits.shape)}。')

        # 所有结构图返回 full resolution，loss/外层会检查；其中 final_logits 已经由这些结构图参与生成。
        return {
            'final_logits': final_logits,
            'coarse_logits': coarse_logits,
            'boundary_logits': boundary_logits,
            'coarse_prob': coarse_prob,
            'interior_map': structure_maps_full['interior_map'],
            'inner_boundary_map': structure_maps_full['inner_boundary_map'],
            'outer_ring_map': structure_maps_full['outer_ring_map'],
            'boundary_candidate_map': structure_maps_full['boundary_candidate_map'],
            'stable_background_map': structure_maps_full['stable_background_map'],
            'foreground_similarity_map': F.interpolate(prototype_gate, size=final_logits.shape[-2:], mode='bilinear', align_corners=True),
            # ==================== 新添加的代码开始：返回 background/contrast map，供外层和 loss 强制检查 ====================
            'background_similarity_map': F.interpolate(background_gate, size=final_logits.shape[-2:], mode='bilinear', align_corners=True),
            'contrast_similarity_map': F.interpolate(contrast_gate, size=final_logits.shape[-2:], mode='bilinear', align_corners=True),
            # ==================== 新添加的代码结束：返回 background/contrast map，供外层和 loss 强制检查 ====================
            'ibr_guidance_used': True,
            'prototype_guidance_used': True,
            # ==================== 新添加的代码开始：background prototype 和 contrast guidance 被实际使用的标志 ====================
            'background_prototype_guidance_used': True,
            'contrast_guidance_used': True,
            # ==================== 新添加的代码结束：background prototype 和 contrast guidance 被实际使用的标志 ====================
            'interactive_attention_used': True,
        }
# ==================== 新添加的代码结束：IBR 内部区域引导边界细化 VMamba 解码器 ====================
