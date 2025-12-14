"""General training utilities built on PyTorch."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import json
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau

try:
    from tqdm.auto import tqdm  # type: ignore
except ImportError:  # pragma: no cover - tqdm is optional
    tqdm = None  # type: ignore


_TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs"                       : 5,
    "device"                       : "auto",
    "gradient_clip_norm"           : None,
    "gradient_accumulation_steps"  : 1,
    "use_amp"                      : "auto",
    "log_interval"                 : 50,
    "non_blocking"                 : True,
    "early_stopping_patience"      : 10,
    "lr_reduction_patience"        : 5,
    "lr_reduction_factor"          : 0.5,
}


''' Function: _resolve_bool
    Description: Convert various input types to boolean value with auto-detection support.
    Args:
        value   : Input value to convert.
        default : Default boolean value if auto is specified.
    Returns:
        Resolved boolean value.
'''
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


''' Function: _resolve_device
    Description: Resolve device string with automatic CUDA detection.
    Args:
        value : Device specification string or object.
    Returns:
        Resolved device string ('cuda' or 'cpu').
'''
def _resolve_device(value: Any) -> str:
    if value is None:
        value = "auto"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value.strip()
    return str(value)


''' Function: _resolve_positive_int
    Description: Convert value to positive integer with minimum bound of 1.
    Args:
        value   : Input value to convert.
        default : Default value if input is None.
    Returns:
        Positive integer value.
'''
def _resolve_positive_int(value: Any, default: int) -> int:
    candidate = default if value is None else int(value)
    return max(1, candidate)


def _resolve_early_stopping_patience(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"", "none", "null", "off", "false"}:
            return None
        value = lowered
    try:
        patience = int(value)
    except (TypeError, ValueError):
        patience = int(float(value))
    return patience if patience > 0 else None


def _resolve_lr_reduction_patience(value: Any) -> int | None:
    return _resolve_early_stopping_patience(value)


def _resolve_lr_reduction_factor(value: Any) -> float:
    factor = 0.5 if value is None else float(value)
    if factor <= 0 or factor >= 1:
        raise ValueError("lr_reduction_factor must be between 0 and 1 (exclusive).")
    return factor


def _resolve_gradient_clip_norm(value: Any) -> float | None:
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


def _resolve_use_amp(value: Any) -> bool:
    default = torch.cuda.is_available()
    resolved = _resolve_bool(value, default)
    return resolved and torch.cuda.is_available()


def _default_epochs() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("epochs"), 5)


def _default_device() -> str:
    return _resolve_device(_TRAINING_DEFAULTS.get("device"))


def _default_gradient_clip_norm() -> float | None:
    return _resolve_gradient_clip_norm(_TRAINING_DEFAULTS.get("gradient_clip_norm"))


def _default_gradient_accumulation_steps() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("gradient_accumulation_steps"), 1)


def _default_use_amp() -> bool:
    return _resolve_use_amp(_TRAINING_DEFAULTS.get("use_amp"))


def _default_log_interval() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("log_interval"), 50)


def _default_non_blocking() -> bool:
    return _resolve_bool(_TRAINING_DEFAULTS.get("non_blocking"), True)


def _default_early_stopping_patience() -> int | None:
    return _resolve_early_stopping_patience(_TRAINING_DEFAULTS.get("early_stopping_patience"))


def _default_lr_reduction_patience() -> int | None:
    return _resolve_lr_reduction_patience(_TRAINING_DEFAULTS.get("lr_reduction_patience"))


def _default_lr_reduction_factor() -> float:
    return _resolve_lr_reduction_factor(_TRAINING_DEFAULTS.get("lr_reduction_factor"))


@dataclass
class TrainingConfig:
    epochs                       : int                = field(default_factory=_default_epochs)
    device                       : str                = field(default_factory=_default_device)
    gradient_clip_norm           : float | None       = field(default_factory=_default_gradient_clip_norm)
    gradient_accumulation_steps  : int                = field(default_factory=_default_gradient_accumulation_steps)
    use_amp                      : bool | str | None  = field(default_factory=_default_use_amp)  # type: ignore[assignment]
    log_interval                 : int                = field(default_factory=_default_log_interval)
    non_blocking                 : bool               = field(default_factory=_default_non_blocking)
    early_stopping_patience      : int | None         = field(default_factory=_default_early_stopping_patience)
    lr_reduction_patience        : int | None         = field(default_factory=_default_lr_reduction_patience)
    lr_reduction_factor          : float              = field(default_factory=_default_lr_reduction_factor)

    def __post_init__(self) -> None:
        self.epochs = _resolve_positive_int(self.epochs, 5)
        self.device = _resolve_device(self.device)
        self.gradient_clip_norm = _resolve_gradient_clip_norm(self.gradient_clip_norm)
        self.gradient_accumulation_steps = _resolve_positive_int(
            self.gradient_accumulation_steps, 1
        )
        self.use_amp = _resolve_use_amp(self.use_amp)
        self.log_interval = _resolve_positive_int(self.log_interval, 50)
        self.non_blocking = _resolve_bool(self.non_blocking, True)
        self.early_stopping_patience = _resolve_early_stopping_patience(
            self.early_stopping_patience
        )
        self.lr_reduction_patience = _resolve_lr_reduction_patience(
            self.lr_reduction_patience
        )
        self.lr_reduction_factor = _resolve_lr_reduction_factor(self.lr_reduction_factor)


def load_training_config(
    source: Path | str | Mapping[str, Any] | TrainingConfig | None = None,
) -> TrainingConfig:
    if isinstance(source, TrainingConfig):
        return source
    if isinstance(source, Mapping):
        payload = dict(_TRAINING_DEFAULTS)
        payload.update(source)
    elif source is None:
        payload = dict(_TRAINING_DEFAULTS)
    else:
        resolved = Path(source).expanduser().resolve()
        if not resolved.exists():
            raise FileNotFoundError(f"Training config not found: {resolved}")
        with resolved.open("r", encoding="utf-8") as handle:
            payload = dict(_TRAINING_DEFAULTS)
            payload.update(json.load(handle))
    valid_keys = {
        "epochs",
        "device",
        "gradient_clip_norm",
        "gradient_accumulation_steps",
        "use_amp",
        "log_interval",
        "non_blocking",
        "early_stopping_patience",
        "lr_reduction_patience",
        "lr_reduction_factor",
    }
    kwargs = {key: payload.get(key) for key in valid_keys}
    return TrainingConfig(**kwargs)


''' Function: _move_to_device
    Description: Recursively move batch data to specified device.
    Args:
        batch        : Batch data (tensor, dict, list, or tuple).
        device       : Target device for data.
        non_blocking : Whether to use non-blocking transfer.
    Returns:
        Batch data on target device.
'''
def _move_to_device(batch: Any, device: torch.device, non_blocking: bool) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, Mapping):
        return {k: _move_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, tuple):
        return tuple(_move_to_device(v, device, non_blocking) for v in batch)
    if isinstance(batch, list):
        return [_move_to_device(v, device, non_blocking) for v in batch]
    return batch


def _split_batch(batch: Any) -> tuple[Any, Any]:
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


def _count_batch_items(batch: Any) -> int:
    if isinstance(batch, torch.Tensor):
        return batch.size(0)
    if isinstance(batch, Mapping):
        for value in batch.values():
            return _count_batch_items(value)
        return 0
    if isinstance(batch, (list, tuple)) and batch:
        return _count_batch_items(batch[0])
    return 0


def _try_len(iterable: Iterable[Any]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


def _progress_iter(iterable: Iterable[Any], desc: str) -> tuple[Iterable[Any], Optional[Any]]:
    if tqdm is None:
        return iterable, None
    total = _try_len(iterable)
    bar = tqdm(iterable, desc=desc, total=total, leave=True)
    return bar, bar


def _forward_model(model: nn.Module, inputs: Any) -> torch.Tensor:
    if isinstance(inputs, Mapping):
        return model(**inputs)
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


def train_one_epoch(model         : nn.Module,
                    dataloader    : Iterable[Any],
                    optimizer     : Optimizer,
                    loss_fn       : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    config        : TrainingConfig,
                    scaler        : Optional[GradScaler] = None,
                    progress_desc : Optional[str] = None,
                   ) -> Dict[str, float]:
    
    device = torch.device(config.device)
    model.to(device)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    scaler = scaler or GradScaler(enabled=config.use_amp and torch.cuda.is_available())

    accum_steps = max(config.gradient_accumulation_steps, 1)
    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    iterator, progress_bar = _progress_iter(dataloader, progress_desc or "Train")
    try:
        for step, raw_batch in enumerate(iterator, start=1):
            batch = _move_to_device(raw_batch, device, config.non_blocking)
            inputs, targets = _split_batch(batch)
            with autocast(device_type=device.type, enabled=scaler.is_enabled()):
                outputs = _forward_model(model, inputs)
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
            if progress_bar is not None:
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()

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
        "loss"     : total_loss / max(total_steps, 1),
        "loss_sum" : total_loss,
        "batches"  : total_steps,
        "examples" : total_examples,
    }


''' Function: evaluate
    Description: Evaluate model on validation/test data without gradient computation.
    Args:
        model         : Neural network model to evaluate.
        dataloader    : Evaluation data iterator.
        loss_fn       : Loss function for computing evaluation loss.
        device        : Device for model and data.
        non_blocking  : Whether to use non-blocking data transfer.
        progress_desc : Description for progress bar display.
    Returns:
        Dictionary with evaluation metrics (loss, examples, batches).
'''
def evaluate(model         : nn.Module,
             dataloader    : Iterable[Any],
             loss_fn       : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             device        : str | torch.device,
             non_blocking  : bool = True,
             progress_desc : Optional[str] = None,
            ) -> Dict[str, float]:
    device = torch.device(device)
    model.to(device)
    model.eval()

    total_loss = 0.0
    total_examples = 0
    total_steps = 0

    iterator, progress_bar = _progress_iter(dataloader, progress_desc or "Eval")
    with torch.no_grad():
        try:
            for raw_batch in iterator:
                batch = _move_to_device(raw_batch, device, non_blocking)
                inputs, targets = _split_batch(batch)
                outputs = _forward_model(model, inputs)
                loss = loss_fn(outputs, targets)

                total_loss += loss.item()
                total_examples += _count_batch_items(raw_batch)
                total_steps += 1
                if progress_bar is not None:
                    progress_bar.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)
        finally:
            if progress_bar is not None:
                progress_bar.close()

    return {
        "loss"     : total_loss / max(total_steps, 1),
        "loss_sum" : total_loss,
        "batches"  : total_steps,
        "examples" : total_examples,
    }


class Trainer:
    ''' Function: __init__
        Description: Initialize trainer with model, optimizer, and training components.
        Args:
            model        : Neural network model to train.
            optimizer    : Optimizer for parameter updates.
            loss_fn      : Loss function for training.
            train_loader : Training data loader.
            config       : Training configuration.
            val_loader   : Optional validation data loader.
            scheduler    : Optional learning rate scheduler.
        Returns:
            None
    '''
    def __init__(self,
                 model        : nn.Module,
                 optimizer    : Optimizer,
                 loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 train_loader : Iterable[Any],
                 config       : TrainingConfig,
                 val_loader   : Optional[Iterable[Any]] = None,
                 scheduler    : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
                 logger       : Optional[Callable[[Dict[str, Any]], None]] = None,
                ) -> None:
        self.model        = model.to(torch.device(config.device))
        self.optimizer    = optimizer
        self.loss_fn      = loss_fn
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.config       = config
        self.scheduler    = scheduler
        self.scaler = GradScaler(enabled=config.use_amp and torch.cuda.is_available())
        self.history: list[Dict[str, Any]] = []
        self.logger = logger
        self.best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        self.best_val_loss: float = float("inf")
        self.best_epoch: Optional[int] = None

    ''' Function: fit
        Description: Train model for configured number of epochs with validation.
        Args:
            None
        Returns:
            List of dictionaries containing training history per epoch.
    '''
    def fit(self) -> list[Dict[str, Any]]:

        patience = self.config.early_stopping_patience
        monitor_early_stop = (
            patience is not None and patience > 0 and self.val_loader is not None
        )
        lr_patience = self.config.lr_reduction_patience
        monitor_lr_reduction = (
            lr_patience is not None and lr_patience > 0 and self.val_loader is not None
        )
        lr_factor = self.config.lr_reduction_factor
        stagnant_epochs = 0
        stagnant_lr_epochs = 0

        for epoch in range(1, self.config.epochs + 1):
            iterator, _ = _progress_iter(self.train_loader, f"Epoch {epoch}")
            train_metrics = train_one_epoch(self.model,
                                            self.train_loader,
                                            self.optimizer,
                                            self.loss_fn,
                                            self.config,
                                            scaler        = self.scaler,
                                            progress_desc = f"[train]",
                                           )

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = evaluate(self.model,
                                       self.val_loader,
                                       self.loss_fn,
                                       self.config.device,
                                       self.config.non_blocking,
                                       progress_desc = f"Epoch {epoch} [val]",
                                      )

            should_stop_after_epoch = False
            lr_reduced = False
            record_best = False
            if (monitor_early_stop or monitor_lr_reduction) and val_metrics is not None:
                current_loss = val_metrics.get("loss")
                if current_loss is not None:
                    val_loss = float(current_loss)
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        stagnant_epochs = 0
                        stagnant_lr_epochs = 0
                        self.best_state_dict = {key: tensor.detach().cpu().clone()
                                                for key, tensor in self.model.state_dict().items()
                                               }
                        self.best_epoch = epoch
                        record_best = True
                    else:
                        stagnant_epochs += 1
                        stagnant_lr_epochs += 1
                        if monitor_lr_reduction and lr_patience is not None and stagnant_lr_epochs >= lr_patience:
                            lr_reduced = self._reduce_learning_rate(lr_factor)
                            stagnant_lr_epochs = 0
                        if monitor_early_stop and patience is not None and stagnant_epochs >= patience:
                            should_stop_after_epoch = True

            self._step_scheduler(val_metrics)
            record = {
                "epoch" : epoch,
                "train" : train_metrics,
                "val"   : val_metrics,
                "lr"    : self.optimizer.param_groups[0]["lr"],
            }
            if lr_reduced:
                record["lr_reduced"] = True
            if record_best:
                record["best_checkpoint"] = True
            if should_stop_after_epoch:
                record["early_stop_triggered"] = True
            self.history.append(record)
            if self.logger is not None:
                self.logger(record)
            if should_stop_after_epoch:
                break

        return self.history

    ''' Function: _step_scheduler
        Description: Step learning rate scheduler based on validation metrics.
        Args:
            val_metrics : Optional validation metrics dictionary.
        Returns:
            None
    '''
    def _step_scheduler(self, val_metrics: Optional[Dict[str, float]]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_metrics is None:
                raise ValueError("Validation metrics required for ReduceLROnPlateau scheduler.")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()

    def _reduce_learning_rate(self, factor: float) -> bool:
        if factor <= 0 or factor >= 1:
            return False
        updated = False
        for group in self.optimizer.param_groups:
            current_lr = group.get("lr")
            if current_lr is None:
                continue
            group["lr"] = current_lr * factor
            updated = True
        return updated

    def best_model_state_dict(self) -> Dict[str, torch.Tensor]:
        if self.best_state_dict is not None:
            return copy.deepcopy(self.best_state_dict)
        return self.model.state_dict()


''' Function: fit
    Description: Convenience function to train model using Trainer class.
    Args:
        model        : Neural network model to train.
        optimizer    : Optimizer for parameter updates.
        loss_fn      : Loss function for training.
        train_loader : Training data loader.
        config       : Training configuration.
        val_loader   : Optional validation data loader.
        scheduler    : Optional learning rate scheduler.
    Returns:
        List of dictionaries containing training history per epoch.
'''
def fit(model        : nn.Module,
        optimizer    : Optimizer,
        loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader : Iterable[Any],
        config       : TrainingConfig,
        val_loader   : Optional[Iterable[Any]] = None,
        scheduler    : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
        logger       : Optional[Callable[[Dict[str, Any]], None]] = None,
       ) -> list[Dict[str, Any]]:
    trainer = Trainer(model        = model,
                      optimizer    = optimizer,
                      loss_fn      = loss_fn,
                      train_loader = train_loader,
                      config       = config,
                      val_loader   = val_loader,
                      scheduler    = scheduler,
                      logger       = logger,
                     )
    return trainer.fit()
