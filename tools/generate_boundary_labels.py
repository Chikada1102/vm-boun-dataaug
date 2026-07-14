# -*- coding: utf-8 -*-
"""
从 GT mask 生成边界标签。

输出目录默认：data/isic2017/train/boundaries/
文件名与 train/masks 下的 mask 保持一致，Dataset 会按 stem 严格匹配。
"""

import argparse
import os
import numpy as np
from PIL import Image
from scipy import ndimage


# ==================== 新添加的代码开始：GT mask 生成边界标签 ====================
def generate_boundary(mask, kernel_size=3):
    mask = mask > 127
    if mask.sum() == 0:
        return np.zeros(mask.shape, dtype=np.uint8)
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    dilated = ndimage.binary_dilation(mask, structure=structure)
    eroded = ndimage.binary_erosion(mask, structure=structure)
    boundary = np.logical_xor(dilated, eroded)
    return (boundary.astype(np.uint8) * 255)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/isic2017/', help='数据集根目录')
    parser.add_argument('--split', type=str, default='train', choices=['train'], help='当前版本只给 train 生成边界监督')
    parser.add_argument('--mask_dir', type=str, default=None, help='可选：直接指定 mask 目录')
    parser.add_argument('--output_dir', type=str, default=None, help='可选：直接指定输出目录')
    parser.add_argument('--kernel_size', type=int, default=3, help='边界宽度核大小，建议 3 或 5')
    parser.add_argument('--overwrite', action='store_true', help='覆盖已有文件')
    args = parser.parse_args()

    mask_dir = args.mask_dir or os.path.join(args.data_path, args.split, 'masks')
    output_dir = args.output_dir or os.path.join(args.data_path, args.split, 'boundaries')

    if not os.path.isdir(mask_dir):
        raise RuntimeError(f'mask 目录不存在：{mask_dir}')
    os.makedirs(output_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))])
    if len(files) == 0:
        raise RuntimeError(f'mask 目录为空：{mask_dir}')

    for idx, name in enumerate(files):
        out_path = os.path.join(output_dir, name)
        if os.path.exists(out_path) and not args.overwrite:
            continue
        mask = np.array(Image.open(os.path.join(mask_dir, name)).convert('L'))
        boundary = generate_boundary(mask, kernel_size=args.kernel_size)
        Image.fromarray(boundary).save(out_path)
        if idx % 100 == 0:
            print(f'[{idx}/{len(files)}] saved {out_path}')

    print(f'完成：共处理 {len(files)} 个 mask，输出目录：{output_dir}')
# ==================== 新添加的代码结束：GT mask 生成边界标签 ====================


if __name__ == '__main__':
    main()
