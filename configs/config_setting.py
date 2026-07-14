"""
VM-UNet + IBR Boundary Refinement + Background Prototype Contrast
+ Albumentations skin-lesion data augmentation configuration.

Important:
    The dataset should call the Albumentations pipeline as follows when a
    boundary label is available:

        transformed = transform(
            image=image,
            mask=mask,
            boundary=boundary,
        )

    Then read:
        image = transformed["image"]
        mask = transformed["mask"]
        boundary = transformed["boundary"]

    This ensures that mask and boundary receive exactly the same geometric
    transformations.
"""

from datetime import datetime
import inspect
import math

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

from utils import *


def _has_parameter(callable_obj, parameter_name):
    """Return whether an Albumentations transform accepts a parameter."""
    try:
        return parameter_name in inspect.signature(callable_obj).parameters
    except (TypeError, ValueError):
        return False


def _constant_border_kwargs(transform_cls):
    """
    Albumentations 1.x uses value/mask_value, while 2.x uses fill/fill_mask.
    """
    if _has_parameter(transform_cls, "fill"):
        return {"fill": 0, "fill_mask": 0}
    return {"value": 0, "mask_value": 0}



def _resize(height, width, p=1.0):
    """Build Resize while preserving nearest-neighbor mask interpolation."""
    kwargs = dict(
        height=height,
        width=width,
        interpolation=cv2.INTER_LINEAR,
        p=p,
    )
    if _has_parameter(A.Resize, "mask_interpolation"):
        kwargs["mask_interpolation"] = cv2.INTER_NEAREST
    return A.Resize(**kwargs)


def _random_resized_crop(height, width, p=0.65):
    """Build RandomResizedCrop for Albumentations 1.x or 2.x."""
    common_kwargs = dict(
        scale=(0.75, 1.0),
        ratio=(0.85, 1.15),
        interpolation=cv2.INTER_LINEAR,
        p=p,
    )

    if _has_parameter(A.RandomResizedCrop, "mask_interpolation"):
        common_kwargs["mask_interpolation"] = cv2.INTER_NEAREST

    if _has_parameter(A.RandomResizedCrop, "size"):
        return A.RandomResizedCrop(size=(height, width), **common_kwargs)

    return A.RandomResizedCrop(height=height, width=width, **common_kwargs)


def _shift_scale_rotate():
    kwargs = dict(
        shift_limit=0.08,
        scale_limit=0.15,
        rotate_limit=45,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0,
    )
    kwargs.update(_constant_border_kwargs(A.ShiftScaleRotate))
    return A.ShiftScaleRotate(**kwargs)


def _elastic_transform():
    kwargs = dict(
        alpha=25,
        sigma=5,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0,
    )
    kwargs.update(_constant_border_kwargs(A.ElasticTransform))
    return A.ElasticTransform(**kwargs)


def _grid_distortion():
    kwargs = dict(
        num_steps=5,
        distort_limit=0.18,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        p=1.0,
    )
    kwargs.update(_constant_border_kwargs(A.GridDistortion))
    return A.GridDistortion(**kwargs)


def _gauss_noise():
    """
    Albumentations 1.x uses var_limit; 2.x uses std_range.
    The 2.x values below approximate variance 5..35 for an 8-bit image.
    """
    if _has_parameter(A.GaussNoise, "var_limit"):
        return A.GaussNoise(var_limit=(5.0, 35.0), p=1.0)

    std_min = math.sqrt(5.0) / 255.0
    std_max = math.sqrt(35.0) / 255.0
    return A.GaussNoise(std_range=(std_min, std_max), p=1.0)


def _image_compression():
    """Build ImageCompression for Albumentations 1.x or 2.x."""
    if _has_parameter(A.ImageCompression, "quality_range"):
        return A.ImageCompression(quality_range=(65, 95), p=1.0)

    return A.ImageCompression(
        quality_lower=65,
        quality_upper=95,
        p=1.0,
    )


def _coarse_dropout(height, width):
    """Build CoarseDropout for Albumentations 1.x or 2.x."""
    if _has_parameter(A.CoarseDropout, "num_holes_range"):
        return A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_size_range=(0.02, 0.08),
            fill=0,
            p=0.18,
        )

    return A.CoarseDropout(
        min_holes=1,
        max_holes=4,
        min_height=max(1, int(height * 0.02)),
        max_height=max(1, int(height * 0.08)),
        min_width=max(1, int(width * 0.02)),
        max_width=max(1, int(width * 0.08)),
        fill_value=0,
        mask_fill_value=None,
        p=0.18,
    )


class HairArtifact(A.ImageOnlyTransform):
    """
    Draw random dark anti-aliased lines on the image only.

    Because this inherits from ImageOnlyTransform, segmentation masks and
    boundary labels are not modified by the artificial hairs.
    """

    def __init__(
        self,
        num_hairs=(3, 8),
        hair_length=(30, 120),
        hair_thickness=(1, 2),
        darkness=(20, 80),
        always_apply=False,
        p=0.25,
    ):
        if always_apply:
            p = 1.0
        super().__init__(p=p)
        self.num_hairs = num_hairs
        self.hair_length = hair_length
        self.hair_thickness = hair_thickness
        self.darkness = darkness

    def apply(self, img, **params):
        image = img.copy()
        height, width = image.shape[:2]

        number_of_hairs = np.random.randint(
            self.num_hairs[0],
            self.num_hairs[1] + 1,
        )

        for _ in range(number_of_hairs):
            x1 = np.random.randint(0, width)
            y1 = np.random.randint(0, height)

            length = np.random.randint(
                self.hair_length[0],
                self.hair_length[1] + 1,
            )
            angle = np.random.uniform(0, 2 * np.pi)

            x2 = int(x1 + length * np.cos(angle))
            y2 = int(y1 + length * np.sin(angle))

            thickness = np.random.randint(
                self.hair_thickness[0],
                self.hair_thickness[1] + 1,
            )
            color_value = np.random.randint(
                self.darkness[0],
                self.darkness[1] + 1,
            )
            color = (color_value, color_value, color_value)

            cv2.line(
                image,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )

        return image

    def get_transform_init_args_names(self):
        return (
            "num_hairs",
            "hair_length",
            "hair_thickness",
            "darkness",
        )


class setting_config:
    """
    VM-UNet + IBR Boundary Refinement configuration.

    IBR = Interior-guided Boundary Refinement.
    """

    network = "vmunet"

    model_config = {
        "num_classes": 1,
        "input_channels": 3,
        # ----- VM-UNet ----- #
        "depths": [2, 2, 2, 2],
        "depths_decoder": [2, 2, 2, 1],
        "drop_path_rate": 0.2,
        "load_ckpt_path": "./pre_trained_weights/vmamba_small_e238_ema.pth",
        # ----- IBR internal-region-guided boundary refinement ----- #
        "use_ibr_guidance": True,
        "ibr_kernel_size": 5,
    }

    datasets = "isic17"
    if datasets == "isic18":
        data_path = "./data/isic2018/"
    elif datasets == "isic17":
        data_path = "./data/isic2017/"
    else:
        raise ValueError("datasets is not right!")

    # ----- IBR training and boundary labels ----- #
    use_ibr_guidance = True
    boundary_label_dir = "./data/isic2017/train/boundaries/"
    aux_label_fail_silently = False

    criterion = IBRBoundaryRefinementLoss(
        wb=1,
        wd=1,
        coarse_weight=0.25,
        boundary_weight=0.25,
        outer_suppression_weight=0.05,
        # Background-prototype contrastive suppression losses.
        background_suppression_weight=0.05,
        contrast_boundary_weight=0.05,
    )

    pretrained_path = "./pre_trained/"
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
    gpu_id = "0"
    batch_size = 32
    epochs = 300

    work_dir = (
        "results/"
        + network
        + "_"
        + datasets
        + "_ibr_bgproto_contrast_dataaug_"
        + datetime.now().strftime("%A_%d_%B_%Y_%Hh_%Mm_%Ss")
        + "/"
    )

    print_interval = 20
    val_interval = 10
    save_interval = 100
    threshold = 0.5
    only_test_and_save_figs = False
    best_ckpt_path = "PATH_TO_YOUR_BEST_CKPT"
    img_save_path = "PATH_TO_SAVE_IMAGES"

    # Albumentations applies all spatial transforms identically to:
    # image, mask, and boundary. HairArtifact affects image only.
    train_transformer = A.Compose(
        [
            # Simulate lesion-scale and field-of-view variation.
            _random_resized_crop(input_size_h, input_size_w, p=0.65),

            # Guarantee a fixed network input size.
            _resize(input_size_h, input_size_w, p=1.0),

            # Geometric augmentation.
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),

            A.OneOf(
                [
                    _shift_scale_rotate(),
                    _elastic_transform(),
                    _grid_distortion(),
                ],
                p=0.45,
            ),

            # Color and illumination augmentation.
            A.OneOf(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=0.22,
                        contrast_limit=0.22,
                        p=1.0,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=12,
                        sat_shift_limit=18,
                        val_shift_limit=15,
                        p=1.0,
                    ),
                    A.RandomGamma(
                        gamma_limit=(70, 135),
                        p=1.0,
                    ),
                    A.RGBShift(
                        r_shift_limit=12,
                        g_shift_limit=12,
                        b_shift_limit=12,
                        p=1.0,
                    ),
                    A.CLAHE(
                        clip_limit=2.0,
                        tile_grid_size=(8, 8),
                        p=1.0,
                    ),
                ],
                p=0.75,
            ),

            # Imaging degradation.
            A.OneOf(
                [
                    _gauss_noise(),
                    A.GaussianBlur(
                        blur_limit=(3, 5),
                        p=1.0,
                    ),
                    A.MotionBlur(
                        blur_limit=5,
                        p=1.0,
                    ),
                    _image_compression(),
                ],
                p=0.22,
            ),

            # Hair occlusion modifies image only.
            HairArtifact(
                num_hairs=(3, 8),
                hair_length=(30, 120),
                hair_thickness=(1, 2),
                darkness=(15, 80),
                p=0.25,
            ),

            # Small image-only occlusions.
            _coarse_dropout(input_size_h, input_size_w),

            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={
            "boundary": "mask",
        },
    )

    test_transformer = A.Compose(
        [
            _resize(input_size_h, input_size_w, p=1.0),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ],
        additional_targets={
            "boundary": "mask",
        },
    )

    opt = "AdamW"
    assert opt in [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "Adamax",
        "ASGD",
        "RMSprop",
        "Rprop",
        "SGD",
    ], "Unsupported optimizer!"

    if opt == "AdamW":
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 1e-2
        amsgrad = False
    elif opt == "Adam":
        lr = 0.001
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0.0001
        amsgrad = False
    elif opt == "SGD":
        lr = 0.01
        momentum = 0.9
        weight_decay = 0.05
        dampening = 0
        nesterov = False
    elif opt == "Adadelta":
        lr = 0.01
        rho = 0.9
        eps = 1e-6
        weight_decay = 0.05
    elif opt == "Adagrad":
        lr = 0.01
        lr_decay = 0
        eps = 1e-10
        weight_decay = 0.05
    elif opt == "Adamax":
        lr = 2e-3
        betas = (0.9, 0.999)
        eps = 1e-8
        weight_decay = 0
    elif opt == "ASGD":
        lr = 0.01
        lambd = 1e-4
        alpha = 0.75
        t0 = 1e6
        weight_decay = 0
    elif opt == "RMSprop":
        lr = 1e-2
        momentum = 0
        alpha = 0.99
        eps = 1e-8
        centered = False
        weight_decay = 0
    elif opt == "Rprop":
        lr = 1e-2
        etas = (0.5, 1.2)
        step_sizes = (1e-6, 50)

    sch = "CosineAnnealingLR"
    if sch == "CosineAnnealingLR":
        T_max = 300
        eta_min = 0.00001
        last_epoch = -1
    elif sch == "StepLR":
        step_size = epochs // 5
        gamma = 0.5
        last_epoch = -1
    elif sch == "MultiStepLR":
        milestones = [60, 120, 150]
        gamma = 0.1
        last_epoch = -1
    elif sch == "ExponentialLR":
        gamma = 0.99
        last_epoch = -1
    elif sch == "ReduceLROnPlateau":
        mode = "min"
        factor = 0.1
        patience = 10
        threshold = 0.0001
        threshold_mode = "rel"
        cooldown = 0
        min_lr = 0
        eps = 1e-08
    elif sch == "CosineAnnealingWarmRestarts":
        T_0 = 50
        T_mult = 2
        eta_min = 1e-6
        last_epoch = -1
    elif sch == "WP_MultiStepLR":
        warm_up_epochs = 10
        gamma = 0.1
        milestones = [125, 225]
    elif sch == "WP_CosineLR":
        warm_up_epochs = 20
