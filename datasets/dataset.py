"""
Dataset definitions for VM-UNet.

This version supports the Albumentations pipeline used by the
VM-UNet + IBR configuration.  For ISIC data, image, segmentation mask and
boundary label are passed to Albumentations as named arguments so that all
geometric augmentations remain spatially aligned.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _list_image_files(directory: str) -> List[str]:
    """Return sorted image files from a directory."""
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"Dataset directory does not exist: {directory}")

    files = [
        os.path.join(directory, name)
        for name in os.listdir(directory)
        if Path(name).suffix.lower() in _IMAGE_EXTENSIONS
    ]
    files.sort()
    return files


def _normalised_stem(path: str) -> str:
    """Normalise common ISIC image/mask/boundary filename suffixes."""
    stem = Path(path).stem.lower()
    suffixes = (
        "_segmentation",
        "_mask",
        "_masks",
        "_boundary",
        "_boundaries",
        "_gt",
        "_label",
    )

    changed = True
    while changed:
        changed = False
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
                changed = True
                break
    return stem


def _index_by_stem(files: List[str]) -> Dict[str, str]:
    """Index paths by both exact and normalised stems."""
    index: Dict[str, str] = {}
    for path in files:
        exact_stem = Path(path).stem.lower()
        index.setdefault(exact_stem, path)
        index.setdefault(_normalised_stem(path), path)
    return index


def _match_file(reference_path: str, file_index: Dict[str, str]) -> Optional[str]:
    """Find the image-like file corresponding to a reference filename."""
    exact_stem = Path(reference_path).stem.lower()
    normalised_stem = _normalised_stem(reference_path)
    return file_index.get(exact_stem) or file_index.get(normalised_stem)


def _make_inner_boundary(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Generate an interior boundary target from a binary segmentation mask.

    boundary = mask - erode(mask)

    The output is uint8 with values 0 and 255, matching boundary PNG labels.
    """
    if mask.ndim == 3:
        mask = np.squeeze(mask)

    binary = (mask > 127).astype(np.uint8)
    kernel_size = max(1, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    boundary = binary - eroded
    return (boundary * 255).astype(np.uint8)


def _prepare_binary_target(target) -> torch.Tensor:
    """Convert a mask/boundary target to float tensor with shape [1, H, W]."""
    if not torch.is_tensor(target):
        target = torch.from_numpy(np.asarray(target))

    if target.ndim == 2:
        target = target.unsqueeze(0)
    elif target.ndim == 3:
        # HWC with one channel -> CHW.
        if target.shape[-1] == 1 and target.shape[0] != 1:
            target = target.permute(2, 0, 1)
        # For an unexpected multi-channel mask, retain the first channel.
        elif target.shape[0] != 1:
            target = target[:1]
    else:
        raise ValueError(
            f"Mask/boundary must have 2 or 3 dimensions, got shape {tuple(target.shape)}"
        )

    target = target.float()
    if target.numel() > 0 and target.max().item() > 1.0:
        target = target / 255.0

    # Nearest-neighbour mask transforms should remain binary; thresholding also
    # protects against masks saved with non-standard foreground intensities.
    return (target > 0.5).float().contiguous()


def _prepare_image_without_transform(image: np.ndarray) -> torch.Tensor:
    """Fallback conversion used only when no transformer is configured."""
    image = torch.from_numpy(np.asarray(image).copy())
    if image.ndim != 3:
        raise ValueError(f"RGB image must be HWC, got shape {tuple(image.shape)}")
    image = image.permute(2, 0, 1).float() / 255.0
    return image.contiguous()


class NPY_datasets(Dataset):
    """
    ISIC dataset loader for VM-UNet + IBR.

    Expected layout:

        data_path/
            train/images/
            train/masks/
            train/boundaries/       # required in strict training mode
            val/images/
            val/masks/
            val/boundaries/         # optional; generated from masks if absent

    The training boundary directory can also be supplied through
    ``config.boundary_label_dir``.  A validation override can be supplied
    through ``config.val_boundary_label_dir``.
    """

    def __init__(self, path_Data, config, train=True):
        super().__init__()

        self.path_data = os.path.abspath(os.path.expanduser(path_Data))
        self.config = config
        self.train = bool(train)
        self.split = "train" if self.train else "val"
        self.transformer = (
            config.train_transformer if self.train else config.test_transformer
        )

        self.use_ibr_guidance = bool(
            getattr(config, "use_ibr_guidance", True)
        )
        self.fail_on_missing_train_boundary = not bool(
            getattr(config, "aux_label_fail_silently", False)
        )

        model_config = getattr(config, "model_config", {})
        self.boundary_kernel_size = int(model_config.get("ibr_kernel_size", 5))

        image_dir = os.path.join(self.path_data, self.split, "images")
        mask_dir = os.path.join(self.path_data, self.split, "masks")

        image_files = _list_image_files(image_dir)
        mask_files = _list_image_files(mask_dir)

        if not image_files:
            raise RuntimeError(f"No images found in: {image_dir}")
        if not mask_files:
            raise RuntimeError(f"No masks found in: {mask_dir}")

        mask_index = _index_by_stem(mask_files)

        if self.train:
            configured_boundary_dir = getattr(config, "boundary_label_dir", None)
        else:
            configured_boundary_dir = getattr(config, "val_boundary_label_dir", None)

        if configured_boundary_dir:
            configured_boundary_dir = os.path.expanduser(configured_boundary_dir)
            if not os.path.isabs(configured_boundary_dir):
                configured_boundary_dir = os.path.abspath(configured_boundary_dir)
            boundary_dir = configured_boundary_dir
        else:
            boundary_dir = os.path.join(
                self.path_data,
                self.split,
                "boundaries",
            )

        boundary_files: List[str] = []
        if os.path.isdir(boundary_dir):
            boundary_files = _list_image_files(boundary_dir)
        boundary_index = _index_by_stem(boundary_files)

        if (
            self.train
            and self.use_ibr_guidance
            and self.fail_on_missing_train_boundary
            and not boundary_files
        ):
            raise FileNotFoundError(
                "IBR boundary supervision is enabled, but no training boundary "
                f"labels were found in: {boundary_dir}"
            )

        self.data: List[Tuple[str, str, Optional[str]]] = []
        missing_masks: List[str] = []
        missing_boundaries: List[str] = []

        for image_path in image_files:
            mask_path = _match_file(image_path, mask_index)
            if mask_path is None:
                missing_masks.append(os.path.basename(image_path))
                continue

            boundary_path = _match_file(mask_path, boundary_index)
            if boundary_path is None:
                boundary_path = _match_file(image_path, boundary_index)

            if (
                self.train
                and self.use_ibr_guidance
                and self.fail_on_missing_train_boundary
                and boundary_path is None
            ):
                missing_boundaries.append(os.path.basename(image_path))

            self.data.append((image_path, mask_path, boundary_path))

        if missing_masks:
            preview = ", ".join(missing_masks[:5])
            raise FileNotFoundError(
                f"Could not match masks for {len(missing_masks)} image(s). "
                f"Examples: {preview}"
            )

        if missing_boundaries:
            preview = ", ".join(missing_boundaries[:5])
            raise FileNotFoundError(
                f"Could not match boundary labels for {len(missing_boundaries)} "
                f"training image(s) in {boundary_dir}. Examples: {preview}"
            )

        if not self.data:
            raise RuntimeError(f"No valid samples were assembled for split: {self.split}")

    def __getitem__(self, indx):
        image_path, mask_path, boundary_path = self.data[indx]

        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8)

        if boundary_path is not None:
            boundary = np.asarray(
                Image.open(boundary_path).convert("L"),
                dtype=np.uint8,
            )
        else:
            # Validation boundaries are allowed to be absent. They are derived
            # from the segmentation mask so that the IBR loss still receives a
            # spatially valid target.
            boundary = _make_inner_boundary(
                mask,
                kernel_size=self.boundary_kernel_size,
            )

        if self.transformer is not None:
            # Albumentations must be called with named arguments and returns a
            # dictionary.  ``boundary`` is configured as an additional mask
            # target in the merged config file.
            transformed = self.transformer(
                image=image,
                mask=mask,
                boundary=boundary,
            )

            if not isinstance(transformed, dict):
                raise TypeError(
                    "The configured transformer must return an Albumentations "
                    "dictionary containing image, mask and boundary."
                )

            image = transformed["image"]
            mask = transformed["mask"]
            boundary = transformed["boundary"]
        else:
            image = _prepare_image_without_transform(image)

        if not torch.is_tensor(image):
            image = _prepare_image_without_transform(image)
        else:
            image = image.float().contiguous()

        mask = _prepare_binary_target(mask)
        boundary = _prepare_binary_target(boundary)

        return image, mask, boundary

    def __len__(self):
        return len(self.data)


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(
                image,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=3,
            )
            label = zoom(
                label,
                (self.output_size[0] / x, self.output_size[1] / y),
                order=0,
            )

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        return {"image": image, "label": label.long()}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.sample_list = open(
            os.path.join(list_dir, self.split + ".txt")
        ).readlines()
        self.data_dir = base_dir

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if self.split == "train":
            slice_name = self.sample_list[idx].strip("\n")
            data_path = os.path.join(self.data_dir, slice_name + ".npz")
            data = np.load(data_path)
            image, label = data["image"], data["label"]
        else:
            vol_name = self.sample_list[idx].strip("\n")
            filepath = self.data_dir + "/{}.npy.h5".format(vol_name)
            with h5py.File(filepath, "r") as data:
                image, label = data["image"][:], data["label"][:]

        sample = {"image": image, "label": label}
        if self.transform:
            sample = self.transform(sample)
        sample["case_name"] = self.sample_list[idx].strip("\n")
        return sample
