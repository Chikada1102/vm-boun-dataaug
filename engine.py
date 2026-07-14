"""Training, validation and testing loops for VM-UNet + optional IBR.

This version accepts either:
    (image, mask)
or:
    (image, mask, boundary)

When the configured criterion declares a third boundary argument, the boundary
label is passed to it in train, validation and test consistently.
"""

import inspect
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from utils import save_imgs


def _unpack_batch(data: Any) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Unpack a standard VM-UNet or IBR batch."""
    if isinstance(data, Mapping):
        image = data.get("image", data.get("img"))
        mask = data.get("mask", data.get("msk", data.get("label")))
        boundary = data.get("boundary", data.get("boundary_mask"))
        if image is None or mask is None:
            raise KeyError(
                "Dictionary batch must contain image and mask/label entries."
            )
        return image, mask, boundary

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        if len(data) == 2:
            image, mask = data
            return image, mask, None
        if len(data) == 3:
            image, mask, boundary = data
            return image, mask, boundary

    raise ValueError(
        "DataLoader batch must be (image, mask), (image, mask, boundary), "
        "or a dictionary containing equivalent entries."
    )


def _to_cuda(
    image: torch.Tensor,
    mask: torch.Tensor,
    boundary: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Move a segmentation batch to CUDA using float tensors."""
    image = image.cuda(non_blocking=True).float()
    mask = mask.cuda(non_blocking=True).float()
    if boundary is not None:
        boundary = boundary.cuda(non_blocking=True).float()
    return image, mask, boundary


def _criterion_forward_signature(criterion):
    """Return the most informative callable signature for a loss object."""
    forward = getattr(criterion, "forward", None)
    target = forward if callable(forward) else criterion
    try:
        return inspect.signature(target)
    except (TypeError, ValueError):
        return None


def _compute_loss(criterion, output, mask, boundary=None):
    """
    Call a conventional loss or an IBR loss without hiding internal errors.

    Supported forms include:
        loss(output, mask)
        loss(output, mask, boundary)
        loss(output, mask, boundary=boundary)
        loss(output, mask, boundary_target=boundary)
    """
    if boundary is None:
        return criterion(output, mask)

    signature = _criterion_forward_signature(criterion)
    if signature is None:
        # IBR is enabled in the current dataset, so the three-argument form is
        # the safest fallback when introspection is unavailable.
        return criterion(output, mask, boundary)

    parameters = signature.parameters
    boundary_names = (
        "boundary",
        "boundary_target",
        "boundary_gt",
        "boundary_label",
        "edge_target",
        "edge_gt",
    )

    # Respect keyword-only boundary parameters when present.
    for name in boundary_names:
        parameter = parameters.get(name)
        if parameter is not None and parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            return criterion(output, mask, **{name: boundary})

    positional = [
        parameter
        for parameter in parameters.values()
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    accepts_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in parameters.values()
    )

    if accepts_varargs or len(positional) >= 3:
        return criterion(output, mask, boundary)

    # A conventional two-argument loss remains supported for ablation runs.
    return criterion(output, mask)


def _extract_prediction(output):
    """Extract the final segmentation tensor used for metrics and saving."""
    if torch.is_tensor(output):
        return output

    if isinstance(output, Mapping):
        preferred_keys = (
            "final_logits",
            "final",
            "refined_logits",
            "refined",
            "out",
            "logits",
            "prediction",
            "pred",
        )
        for key in preferred_keys:
            if key in output:
                return _extract_prediction(output[key])

        # Fall back to the first tensor-like value, but fail clearly if the
        # model output does not contain a segmentation prediction.
        for value in output.values():
            try:
                return _extract_prediction(value)
            except (TypeError, ValueError):
                continue
        raise ValueError("Model output dictionary contains no tensor prediction.")

    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("Model returned an empty tuple/list.")
        # VM-UNet-style multi-output models conventionally place the final
        # prediction first.
        return _extract_prediction(output[0])

    raise TypeError(f"Unsupported model output type: {type(output).__name__}")


def _append_metric_arrays(preds, gts, output, mask):
    prediction = _extract_prediction(output)

    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(1)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)

    if prediction.ndim != 4 or mask.ndim != 4:
        raise ValueError(
            "Prediction and mask must have shape [B, C, H, W]; got "
            f"{tuple(prediction.shape)} and {tuple(mask.shape)}."
        )

    # Binary segmentation uses the first/only output channel.
    preds.append(prediction[:, 0].detach().cpu().numpy())
    gts.append(mask[:, 0].detach().cpu().numpy())


def _calculate_binary_metrics(preds, gts, threshold):
    if not preds or not gts:
        raise RuntimeError("No predictions were collected for metric calculation.")

    predictions = np.concatenate([item.reshape(-1) for item in preds])
    ground_truth = np.concatenate([item.reshape(-1) for item in gts])

    y_pre = np.where(predictions >= threshold, 1, 0)
    y_true = np.where(ground_truth >= 0.5, 1, 0)

    # Explicit labels guarantee a 2x2 matrix even when a split contains only
    # foreground or only background pixels.
    confusion = confusion_matrix(y_true, y_pre, labels=[0, 1])
    tn, fp, fn, tp = confusion.ravel()

    total = tn + fp + fn + tp
    accuracy = float(tn + tp) / float(total) if total else 0.0
    sensitivity = float(tp) / float(tp + fn) if (tp + fn) else 0.0
    specificity = float(tn) / float(tn + fp) if (tn + fp) else 0.0
    dsc = float(2 * tp) / float(2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
    miou = float(tp) / float(tp + fp + fn) if (tp + fp + fn) else 0.0

    return {
        "miou": miou,
        "f1_or_dsc": dsc,
        "accuracy": accuracy,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "confusion_matrix": confusion,
    }


def train_one_epoch(
    train_loader,
    model,
    criterion,
    optimizer,
    scheduler,
    epoch,
    step,
    logger,
    config,
    writer,
):
    """Train the model for one epoch."""
    model.train()
    loss_list = []

    for iteration, data in enumerate(train_loader):
        # Increment once per optimizer update. The upstream ``step += iter``
        # grows quadratically within an epoch and is unsuitable for TensorBoard.
        step += 1
        optimizer.zero_grad()

        images, targets, boundaries = _unpack_batch(data)
        images, targets, boundaries = _to_cuda(images, targets, boundaries)

        output = model(images)
        loss = _compute_loss(criterion, output, targets, boundaries)

        loss.backward()
        optimizer.step()

        loss_value = float(loss.detach().item())
        loss_list.append(loss_value)

        now_lr = optimizer.param_groups[0]["lr"]
        writer.add_scalar("loss", loss_value, global_step=step)

        if iteration % config.print_interval == 0:
            log_info = (
                f"train: epoch {epoch}, iter:{iteration}, "
                f"loss: {np.mean(loss_list):.4f}, lr: {now_lr}"
            )
            print(log_info)
            logger.info(log_info)

    scheduler.step()
    return step


def val_one_epoch(test_loader, model, criterion, epoch, logger, config):
    """Validate the model with optional IBR boundary supervision."""
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for data in tqdm(test_loader):
            image, mask, boundary = _unpack_batch(data)
            image, mask, boundary = _to_cuda(image, mask, boundary)

            output = model(image)
            loss = _compute_loss(criterion, output, mask, boundary)
            loss_list.append(float(loss.detach().item()))

            _append_metric_arrays(preds, gts, output, mask)

    mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")

    if epoch % config.val_interval == 0:
        metrics = _calculate_binary_metrics(preds, gts, config.threshold)
        log_info = (
            f"val epoch: {epoch}, loss: {mean_loss:.4f}, "
            f"miou: {metrics['miou']}, "
            f"f1_or_dsc: {metrics['f1_or_dsc']}, "
            f"accuracy: {metrics['accuracy']}, "
            f"specificity: {metrics['specificity']}, "
            f"sensitivity: {metrics['sensitivity']}, "
            f"confusion_matrix: {metrics['confusion_matrix']}"
        )
    else:
        log_info = f"val epoch: {epoch}, loss: {mean_loss:.4f}"

    print(log_info)
    logger.info(log_info)
    return mean_loss


def test_one_epoch(
    test_loader,
    model,
    criterion,
    logger,
    config,
    test_data_name=None,
):
    """Test the model with optional IBR boundary supervision."""
    model.eval()
    preds = []
    gts = []
    loss_list = []

    with torch.no_grad():
        for iteration, data in enumerate(tqdm(test_loader)):
            image, mask, boundary = _unpack_batch(data)
            image, mask, boundary = _to_cuda(image, mask, boundary)

            output = model(image)
            loss = _compute_loss(criterion, output, mask, boundary)
            loss_list.append(float(loss.detach().item()))

            prediction = _extract_prediction(output)
            _append_metric_arrays(preds, gts, prediction, mask)

            if iteration % config.save_interval == 0:
                mask_numpy = mask[:, 0].detach().cpu().numpy()
                prediction_numpy = prediction[:, 0].detach().cpu().numpy()
                save_imgs(
                    image,
                    mask_numpy,
                    prediction_numpy,
                    iteration,
                    config.work_dir + "outputs/",
                    config.datasets,
                    config.threshold,
                    test_data_name=test_data_name,
                )

    mean_loss = float(np.mean(loss_list)) if loss_list else float("nan")
    metrics = _calculate_binary_metrics(preds, gts, config.threshold)

    if test_data_name is not None:
        dataset_info = f"test_datasets_name: {test_data_name}"
        print(dataset_info)
        logger.info(dataset_info)

    log_info = (
        f"test of best model, loss: {mean_loss:.4f}, "
        f"miou: {metrics['miou']}, "
        f"f1_or_dsc: {metrics['f1_or_dsc']}, "
        f"accuracy: {metrics['accuracy']}, "
        f"specificity: {metrics['specificity']}, "
        f"sensitivity: {metrics['sensitivity']}, "
        f"confusion_matrix: {metrics['confusion_matrix']}"
    )
    print(log_info)
    logger.info(log_info)
    return mean_loss
