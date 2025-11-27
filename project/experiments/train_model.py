"""
Training utilities for MLP classifiers on synthetic datasets.

This module implements:
    - SGD (with momentum) and Adam optimizers
    - step learning-rate schedules
    - reproducible training via ``TrainingConfig``
    - checkpoint saving (initial, final, and optional mid-training)
    - logging of train/test loss and accuracy per epoch
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from torch.utils.data import DataLoader, TensorDataset

from project.models import MLPClassifier
from project.utils.configs import TrainingConfig
from project.utils.seed import set_global_seed

logger = logging.getLogger(__name__)


def _create_dataloader(
    x: Tensor,
    y: Tensor,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Create a ``DataLoader`` from tensors.

    Args:
        x (Tensor): Feature tensor of shape (N, D).
        y (Tensor): Label tensor of shape (N,).
        batch_size (int): Batch size.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: DataLoader yielding mini-batches.
    """
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _create_optimizer(model: nn.Module, config: TrainingConfig) -> Optimizer:
    """
    Create an optimizer based on the training configuration.

    Args:
        model (nn.Module): Model whose parameters will be optimized.
        config (TrainingConfig): Training configuration.

    Returns:
        Optimizer: Configured optimizer instance.
    """
    if config.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "adam":
        return Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {config.optimizer}")


def _create_scheduler(optimizer: Optimizer, config: TrainingConfig) -> _LRScheduler:
    """
    Create a step learning-rate scheduler.

    Args:
        optimizer (Optimizer): Optimizer whose learning rate will be scheduled.
        config (TrainingConfig): Training configuration.

    Returns:
        _LRScheduler: StepLR scheduler instance.
    """
    return StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)


def _save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    config: TrainingConfig,
    prefix: str,
) -> Path:
    """
    Save a model checkpoint to disk.

    Args:
        model (nn.Module): Trained model.
        optimizer (Optimizer): Optimizer instance.
        epoch (int): Current epoch number.
        config (TrainingConfig): Training configuration.
        prefix (str): Prefix to distinguish checkpoint type (e.g. ``\"init\"``,
            ``\"epoch10\"``, ``\"final\"``).

    Returns:
        Path: Path to the saved checkpoint file.
    """
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = config.checkpoint_dir / f"{prefix}_epoch{epoch}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "training_config": config,
        },
        path,
    )
    logger.info("Saved checkpoint to %s", path)
    return path


def _compute_accuracy(logits: Tensor, targets: Tensor) -> float:
    """
    Compute classification accuracy given logits and integer labels.

    Args:
        logits (Tensor): Model outputs of shape (N, C).
        targets (Tensor): Ground-truth labels of shape (N,).

    Returns:
        float: Accuracy in the range [0, 1].
    """
    predictions = logits.argmax(dim=1)
    correct = (predictions == targets).float().mean().item()
    return float(correct)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    log_interval: int,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.

    Args:
        model (nn.Module): Model to train.
        dataloader (DataLoader): DataLoader providing training batches.
        criterion (nn.Module): Loss function.
        optimizer (Optimizer): Optimizer instance.
        device (torch.device): Device on which to perform computation.
        log_interval (int): Log every ``log_interval`` batches.

    Returns:
        Tuple[float, float]: Average training loss and accuracy.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        acc = _compute_accuracy(outputs.detach(), targets)

        total_loss += float(loss.item()) * batch_size
        total_correct += acc * batch_size
        total_examples += batch_size

        if log_interval > 0 and batch_idx % log_interval == 0:
            logger.info(
                "Train batch %d: loss=%.4f accuracy=%.4f",
                batch_idx,
                loss.item(),
                acc,
            )

    if total_examples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): DataLoader providing evaluation data.
        criterion (nn.Module): Loss function.
        device (torch.device): Device on which to perform computation.

    Returns:
        Tuple[float, float]: Average loss and accuracy on the dataset.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_examples = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            batch_size = targets.size(0)
            acc = _compute_accuracy(outputs, targets)

            total_loss += float(loss.item()) * batch_size
            total_correct += acc * batch_size
            total_examples += batch_size

    if total_examples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples
    return avg_loss, avg_acc


def train_model(
    model: MLPClassifier,
    x_train: Tensor,
    y_train: Tensor,
    x_test: Tensor,
    y_test: Tensor,
    config: TrainingConfig,
    checkpoint_epochs: Optional[Sequence[int]] = None,
) -> Tuple[MLPClassifier, List[dict]]:
    """
    Train an MLP classifier according to the provided configuration.

    Args:
        model (MLPClassifier): Model instance to train.
        x_train (Tensor): Training features of shape (N_train, D).
        y_train (Tensor): Training labels of shape (N_train,).
        x_test (Tensor): Test features of shape (N_test, D).
        y_test (Tensor): Test labels of shape (N_test,).
        config (TrainingConfig): Training configuration.
        checkpoint_epochs (Optional[Sequence[int]]): Optional list of epoch
            indices at which to save intermediate checkpoints for downstream
            analyses (e.g. PCA trajectories).

    Returns:
        Tuple[MLPClassifier, List[dict]]:
            The trained model and a list of per-epoch metric dictionaries
            containing training and test loss/accuracy.
    """
    device = torch.device(config.device)
    set_global_seed(config.seed)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = _create_optimizer(model, config)
    scheduler = _create_scheduler(optimizer, config)

    train_loader = _create_dataloader(x_train, y_train, batch_size=config.batch_size, shuffle=True)
    test_loader = _create_dataloader(x_test, y_test, batch_size=config.batch_size, shuffle=False)

    metrics: List[dict] = []

    logger.info("Starting training for %d epochs on device %s", config.epochs, device)

    # Save initial checkpoint before any training.
    _save_checkpoint(model, optimizer, epoch=0, config=config, prefix="init")

    checkpoint_set = set(checkpoint_epochs or [])

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            log_interval=config.log_interval,
        )
        test_loss, test_acc = evaluate_model(
            model=model,
            dataloader=test_loader,
            criterion=criterion,
            device=device,
        )

        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        metrics.append(epoch_metrics)

        logger.info(
            "Epoch %d/%d: train_loss=%.4f train_acc=%.4f "
            "test_loss=%.4f test_acc=%.4f lr=%.6f",
            epoch,
            config.epochs,
            train_loss,
            train_acc,
            test_loss,
            test_acc,
            epoch_metrics["learning_rate"],
        )

        if epoch in checkpoint_set:
            _save_checkpoint(model, optimizer, epoch=epoch, config=config, prefix=f"epoch{epoch}")

    # Save final checkpoint after training completes.
    _save_checkpoint(model, optimizer, epoch=config.epochs, config=config, prefix="final")

    return model, metrics

