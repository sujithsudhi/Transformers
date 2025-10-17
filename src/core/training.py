from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_CONFIG_PATH = _PROJECT_ROOT / "configs" / "training.json"


"""Return fallback training defaults merged with user overrides.

Args:
    path: Absolute path to the defaults JSON file.

Returns:
    Dictionary containing merged configuration values.
"""
def _load_training_defaults(path: Path) -> Dict[str, Any]:
    """Return fallback training defaults merged with user overrides.

    Args:
        path: Absolute path to the defaults JSON file.

    Returns:
        Dictionary containing merged configuration values.
    """
    fallback = {
        "epochs": 10,
        "device": "auto",
        "gradient_clip_norm": None,
        "gradient_accumulation_steps": 1,
        "use_amp": "auto",
        "log_interval": 50,
        "non_blocking": True,
    }
    if not path.exists():
        return fallback
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    fallback.update(payload)
    return fallback


"""Resolve truthy configuration values with optional auto fallback.

Args:
    value: Raw configuration entry.
    default: Fallback boolean when value requests auto behaviour.

Returns:
    Normalised boolean flag.
"""
def _resolve_bool(value: Any, default: bool) -> bool:

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return default
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return bool(value)


"""Resolve device configuration to a concrete torch device string.

Args:
    value: Raw device entry, possibly 'auto'.

Returns:
    Device string compatible with torch.device.
"""
def _resolve_device(value: Any) -> str:
    """Resolve device configuration to a concrete torch device string.

    Args:
        value: Raw device entry, possibly 'auto'.

    Returns:
        Device string compatible with torch.device.
    """
    if value is None:
        value = "auto"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value.strip()
    return str(value)


"""Return a positive integer, applying defaults and floor of 1.

Args:
    value: Raw integer-like input.
    default: Default integer used when value is missing.

Returns:
    Positive integer value greater than or equal to 1.
"""
def _resolve_positive_int(value: Any, default: int) -> int:
    """Return a positive integer, applying defaults and floor of 1.

    Args:
        value: Raw integer-like input.
        default: Default integer used when value is missing.

    Returns:
        Positive integer value >= 1.
    """
    candidate = default if value is None else int(value)
    return max(1, candidate)


"""Normalise gradient clipping thresholds.

Args:
    value: Clip magnitude or sentinel values disabling clipping.

Returns:
    Positive float clip value or None when disabled.
"""
def _resolve_gradient_clip_norm(value: Any) -> float | None:
    """Normalise gradient clipping thresholds.

    Args:
        value: Clip magnitude or sentinel values disabling clipping.

    Returns:
        Positive float clip value or None when disabled.
    """
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "off", "false"}:
            return None
        clip = float(value)
    else:
        clip = float(value)
    return clip if clip > 0 else None


"""Decide whether automatic mixed precision should be enabled.

Args:
    value: Raw AMP flag, potentially 'auto'.

Returns:
    True when AMP should be active on supported hardware.
"""
def _resolve_use_amp(value: Any) -> bool:
    """Decide whether automatic mixed precision should be enabled.

    Args:
        value: Raw AMP flag, potentially 'auto'.

    Returns:
        True when AMP should be active on supported hardware.
    """
    default = torch.cuda.is_available()
    resolved = _resolve_bool(value, default)
    return resolved and torch.cuda.is_available()


_TRAINING_DEFAULTS = _load_training_defaults(_TRAINING_CONFIG_PATH)


"""Provide default epoch count drawn from training defaults."""
def _default_epochs() -> int:
    """Provide default epoch count drawn from training defaults."""
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("epochs"), 10)


"""Provide default device string drawn from training defaults."""
def _default_device() -> str:
    """Provide default device string drawn from training defaults."""
    return _resolve_device(_TRAINING_DEFAULTS.get("device"))


"""Provide default gradient clip norm drawn from training defaults."""
def _default_gradient_clip_norm() -> float | None:
    """Provide default gradient clip norm drawn from training defaults."""
    return _resolve_gradient_clip_norm(_TRAINING_DEFAULTS.get("gradient_clip_norm"))


"""Provide default gradient accumulation steps drawn from training defaults."""
def _default_gradient_accumulation_steps() -> int:
    """Provide default gradient accumulation steps drawn from training defaults."""
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("gradient_accumulation_steps"), 1)


"""Provide default AMP flag drawn from training defaults."""
def _default_use_amp() -> bool:
    """Provide default AMP flag drawn from training defaults."""
    return _resolve_use_amp(_TRAINING_DEFAULTS.get("use_amp"))


"""Provide default logging interval drawn from training defaults."""
def _default_log_interval() -> int:
    """Provide default logging interval drawn from training defaults."""
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("log_interval"), 50)


"""Provide default non-blocking transfer flag drawn from training defaults."""
def _default_non_blocking() -> bool:
    """Provide default non-blocking transfer flag drawn from training defaults."""
    return _resolve_bool(_TRAINING_DEFAULTS.get("non_blocking"), True)


_DATACLASS_KWARGS: Dict[str, Any] = {}
if sys.version_info >= (3, 10):
    _DATACLASS_KWARGS["slots"] = True


@dataclass(**_DATACLASS_KWARGS)
class TrainingConfig:
    """Container for training hyperparameters resolved from configuration."""
    epochs: int = field(default_factory=_default_epochs)
    device: str = field(default_factory=_default_device)
    gradient_clip_norm: float | None = field(default_factory=_default_gradient_clip_norm)
    gradient_accumulation_steps: int = field(default_factory=_default_gradient_accumulation_steps)
    use_amp: bool | str | None = field(default_factory=_default_use_amp)  # type: ignore[assignment]
    log_interval: int = field(default_factory=_default_log_interval)
    non_blocking: bool = field(default_factory=_default_non_blocking)

    # Resolve and validate dynamic defaults after instantiation.
    def __post_init__(self) -> None:
        """Resolve and validate dynamic defaults after instantiation."""
        self.epochs = _resolve_positive_int(self.epochs, 10)
        self.device = _resolve_device(self.device)
        self.gradient_clip_norm = _resolve_gradient_clip_norm(self.gradient_clip_norm)
        self.gradient_accumulation_steps = _resolve_positive_int(self.gradient_accumulation_steps, 1)
        self.use_amp = _resolve_use_amp(self.use_amp)
        self.log_interval = _resolve_positive_int(self.log_interval, 50)
        self.non_blocking = _resolve_bool(self.non_blocking, True)


"""Load a training configuration file and construct TrainingConfig.

Args:
    path: Optional path to a training JSON configuration.

Returns:
    Fully-populated TrainingConfig instance.
"""
def load_training_config(path: Path | str | None = None) -> TrainingConfig:
    """Load a training configuration file and construct TrainingConfig.

    Args:
        path: Optional path to a training JSON configuration.

    Returns:
        Fully-populated TrainingConfig instance.
    """
    resolved = Path(path).expanduser().resolve() if path else _TRAINING_CONFIG_PATH
    if not resolved.exists():
        raise FileNotFoundError(f"Training config not found: {resolved}")
    with resolved.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    combined = dict(_TRAINING_DEFAULTS)
    combined.update(payload)
    valid_keys = {
        "epochs",
        "device",
        "gradient_clip_norm",
        "gradient_accumulation_steps",
        "use_amp",
        "log_interval",
        "non_blocking",
    }
    kwargs = {key: combined.get(key) for key in valid_keys}
    return TrainingConfig(**kwargs)


"""Recursively move batch tensors to the selected device.

Args:
    batch: Arbitrary batch structure from a dataloader.
    device: Target torch device.
    non_blocking: Whether to request non-blocking transfers.

Returns:
    Batch mirrored on the target device.
"""
def _move_to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    """Recursively move batch tensors to the selected device.

    Args:
        batch: Arbitrary batch structure from a dataloader.
        device: Target torch device.
        non_blocking: Whether to request non-blocking transfers.

    Returns:
        Batch mirrored on the target device.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device, non_blocking) for v in batch)
    if isinstance(batch, list):
        return [_move_to_device(v, device, non_blocking) for v in batch]
    return batch


"""Extract model inputs and targets from a dataloader batch.

Args:
    batch: Batch emitted by the dataloader.

Returns:
    Tuple of (inputs, targets).

Raises:
    ValueError: When the batch lacks required elements.
    TypeError: When the batch structure is unsupported.
"""
def _split_batch(batch: Any) -> tuple[Any, Any]:
    """Extract model inputs and targets from a dataloader batch.

    Args:
        batch: Batch emitted by the dataloader.

    Returns:
        Tuple of (inputs, targets).

    Raises:
        ValueError: When the batch does not expose at least two elements.
        TypeError: When the batch structure is unsupported.
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Expected (inputs, targets) from the dataloader.")
        return batch[0], batch[1]
    if isinstance(batch, Mapping):
        if "inputs" in batch and "targets" in batch:
            return batch["inputs"], batch["targets"]
        if "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
    raise TypeError("Unsupported batch structure; provide (inputs, targets).")


"""Count effective examples contained in a batch.

Args:
    batch: Batch emitted by the dataloader.

Returns:
    Number of examples inferred from the leading tensor dimension.
"""
def _count_batch_items(batch: Any) -> int:
    """Count effective examples contained in a batch.

    Args:
        batch: Batch emitted by the dataloader.

    Returns:
        Number of examples inferred from the leading tensor dimension.
    """
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    if isinstance(batch, Mapping):
        for value in batch.values():
            return _count_batch_items(value)
        return 0
    if isinstance(batch, (list, tuple)) and batch:
        return _count_batch_items(batch[0])
    return 0


"""Execute a single training epoch with optional gradient scaling.

Args:
    model: Model being trained.
    dataloader: Iterable yielding training batches.
    optimizer: Optimizer responsible for weight updates.
    loss_fn: Loss function applied to model outputs.
    config: Training configuration.
    scaler: Optional gradient scaler for AMP.

Returns:
    Dictionary of aggregated training metrics.
"""
def train_one_epoch(model: nn.Module,
                    dataloader: Iterable[Any],
                    optimizer: Optimizer,
                    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    config: TrainingConfig,
                    scaler: Optional[GradScaler] = None,) -> Dict[str, float]:
    """Execute a single training epoch with optional gradient scaling.

    Args:
        model: Model being trained.
        dataloader: Iterable yielding training batches.
        optimizer: Optimizer responsible for weight updates.
        loss_fn: Loss function applied to model outputs.
        config: Training configuration.
        scaler: Optional gradient scaler for AMP.

    Returns:
        Dictionary of aggregated training metrics.
    """
    device = torch.device(config.device)
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = scaler or GradScaler(enabled=config.use_amp and torch.cuda.is_available())

    accum_steps = max(config.gradient_accumulation_steps, 1)
    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    for step, raw_batch in enumerate(dataloader, start=1):
        batch = _move_to_device(raw_batch, device, config.non_blocking)
        inputs, targets = _split_batch(batch)
        with autocast(enabled=scaler.is_enabled()):
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss_to_backward = loss / accum_steps

        if scaler.is_enabled():
            scaler.scale(loss_to_backward).backward()
        else:
            loss_to_backward.backward()

        should_step = step % accum_steps == 0
        if should_step:
            if config.gradient_clip_norm is not None:
                if scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)

            if scaler.is_enabled():
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_examples += _count_batch_items(raw_batch)
        total_steps += 1

    if total_steps % accum_steps != 0:
        if config.gradient_clip_norm is not None:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm)
        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    return {
        "loss": total_loss / max(total_steps, 1),
        "loss_sum": total_loss,
        "batches": total_steps,
        "examples": total_examples,
    }


"""Evaluate a model on the provided dataloader without gradient tracking.

Args:
    model: Model under evaluation.
    dataloader: Iterable yielding evaluation batches.
    loss_fn: Loss function applied to model outputs.
    device: Device used for inference.
    non_blocking: Whether to request non-blocking transfers.

Returns:
    Dictionary of aggregated evaluation metrics.
"""
def evaluate(model: nn.Module,
            dataloader: Iterable[Any],
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            device: str | torch.device,
            non_blocking: bool = True,
        ) -> Dict[str, float]:
    """Evaluate a model on the provided dataloader without gradient tracking.

    Args:
        model: Model under evaluation.
        dataloader: Iterable yielding evaluation batches.
        loss_fn: Loss function applied to model outputs.
        device: Device used for inference.
        non_blocking: Whether to request non-blocking transfers.

    Returns:
        Dictionary of aggregated evaluation metrics.
    """
    device = torch.device(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    with torch.no_grad():
        for raw_batch in dataloader:
            batch = _move_to_device(raw_batch, device, non_blocking)
            inputs, targets = _split_batch(batch)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            total_loss += loss.item()
            total_examples += _count_batch_items(raw_batch)
            total_steps += 1

    return {
        "loss": total_loss / max(total_steps, 1),
        "loss_sum": total_loss,
        "batches": total_steps,
        "examples": total_examples,
    }


class Trainer:
    """High-level training loop managing epochs, validation, and schedulers."""
    def __init__(self,
                model: nn.Module,
                optimizer: Optimizer,
                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                train_loader: Iterable[Any],
                config: TrainingConfig,
                val_loader: Optional[Iterable[Any]] = None,
                scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
            ) -> None:
        self.model = model.to(torch.device(config.device))
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=config.use_amp and torch.cuda.is_available())
        self.history: list[Dict[str, Any]] = []

    """Run the configured number of epochs and collect metrics.

    Returns:
        List of epoch-level metric dictionaries.
    """
    def fit(self) -> list[Dict[str, Any]]:
        """Run the configured number of epochs and collect metrics.

        Returns:
            List of epoch-level metric dictionaries.
        """
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = train_one_epoch(self.model,
                                            self.train_loader,
                                            self.optimizer,
                                            self.loss_fn,
                                            self.config,
                                            scaler=self.scaler,
            )

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = evaluate(self.model,
                                       self.val_loader,
                                       self.loss_fn,
                                       self.config.device,
                                       self.config.non_blocking,
                )

            self._step_scheduler(val_metrics)
            self.history.append(
                {
                    "epoch": epoch,
                    "train": train_metrics,
                    "val": val_metrics,
                    "lr": self.optimizer.param_groups[0]["lr"],
                }
            )

        return self.history

    """Advance the learning-rate scheduler when supplied.

    Args:
        val_metrics: Validation metrics required by some schedulers.
    """
    def _step_scheduler(self, val_metrics: Optional[Dict[str, float]]) -> None:
        """Advance the learning-rate scheduler when supplied.

        Args:
            val_metrics: Validation metrics required by some schedulers.
        """
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_metrics is None:
                raise ValueError("Validation metrics required for ReduceLROnPlateau scheduler.")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()


"""Convenience wrapper creating a Trainer and executing fit().

Args:
    model: Model being trained.
    optimizer: Optimizer instance.
    loss_fn: Loss function.
    train_loader: Training dataloader.
    config: Training configuration.
    val_loader: Optional validation dataloader.
    scheduler: Optional LR scheduler.

Returns:
    Training history returned by Trainer.fit().
"""
def fit(model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: Iterable[Any],
        config: TrainingConfig,
        val_loader: Optional[Iterable[Any]] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
    ) -> list[Dict[str, Any]]:
    """Convenience wrapper creating a Trainer and executing fit().

    Args:
        model: Model being trained.
        optimizer: Optimizer instance.
        loss_fn: Loss function.
        train_loader: Training dataloader.
        config: Training configuration.
        val_loader: Optional validation dataloader.
        scheduler: Optional LR scheduler.

    Returns:
        Training history returned by Trainer.fit().
    """
    trainer = Trainer(model=model,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     train_loader=train_loader,
                     config=config,
                     val_loader=val_loader,
                     scheduler=scheduler)
    return trainer.fit()
