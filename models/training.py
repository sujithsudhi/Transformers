from __future__ import annotations

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


# Default hyperparameters shared across training configs.
_TRAINING_DEFAULTS: Dict[str, Any] = {
    "epochs": 5,
    "device": "auto",
    "gradient_clip_norm": None,
    "gradient_accumulation_steps": 1,
    "use_amp": "auto",
    "log_interval": 50,
    "non_blocking": True,
}


# Function: _resolve_bool
# Description: Convert assorted truthy inputs into a boolean with auto fallback.
# Args:
#   value: Candidate input that may represent a boolean.
#   default: Value returned when 'auto' is encountered.
# Returns:
#   Boolean indicating resolved truth value.
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


# Function: _resolve_device
# Description: Determine target device string from potentially auto value.
# Args:
#   value: Raw device specification or 'auto'.
# Returns:
#   Device string compatible with torch.device.
def _resolve_device(value: Any) -> str:
    if value is None:
        value = "auto"
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return value.strip()
    return str(value)


# Function: _resolve_positive_int
# Description: Resolve integer inputs ensuring value is positive.
# Args:
#   value: Proposed integer value or None.
#   default: Fallback value when input is missing.
# Returns:
#   Positive integer greater than or equal to one.
def _resolve_positive_int(value: Any, default: int) -> int:
    candidate = default if value is None else int(value)
    return max(1, candidate)


# Function: _resolve_gradient_clip_norm
# Description: Normalise gradient clipping configuration.
# Args:
#   value: Raw clip magnitude or sentinel.
# Returns:
#   Positive float clip value or None when disabled.
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


# Function: _resolve_use_amp
# Description: Decide whether automatic mixed precision should be enabled.
# Args:
#   value: AMP configuration supporting boolean or 'auto'.
# Returns:
#   Boolean indicating AMP usage.
def _resolve_use_amp(value: Any) -> bool:
    default = torch.cuda.is_available()
    resolved = _resolve_bool(value, default)
    return resolved and torch.cuda.is_available()


# Function: _default_epochs
# Description: Provide default epoch count from shared defaults.
def _default_epochs() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("epochs"), 5)


# Function: _default_device
# Description: Provide default device string from shared defaults.
def _default_device() -> str:
    return _resolve_device(_TRAINING_DEFAULTS.get("device"))


# Function: _default_gradient_clip_norm
# Description: Provide default gradient clip norm from shared defaults.
def _default_gradient_clip_norm() -> float | None:
    return _resolve_gradient_clip_norm(_TRAINING_DEFAULTS.get("gradient_clip_norm"))


# Function: _default_gradient_accumulation_steps
# Description: Provide default gradient accumulation steps.
def _default_gradient_accumulation_steps() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("gradient_accumulation_steps"), 1)


# Function: _default_use_amp
# Description: Provide default AMP setting.
def _default_use_amp() -> bool:
    return _resolve_use_amp(_TRAINING_DEFAULTS.get("use_amp"))


# Function: _default_log_interval
# Description: Provide default logging interval.
def _default_log_interval() -> int:
    return _resolve_positive_int(_TRAINING_DEFAULTS.get("log_interval"), 50)


# Function: _default_non_blocking
# Description: Provide default non-blocking transfer setting.
def _default_non_blocking() -> bool:
    return _resolve_bool(_TRAINING_DEFAULTS.get("non_blocking"), True)


@dataclass
class TrainingConfig:
    epochs: int = field(default_factory=_default_epochs)
    device: str = field(default_factory=_default_device)
    gradient_clip_norm: float | None = field(default_factory=_default_gradient_clip_norm)
    gradient_accumulation_steps: int = field(default_factory=_default_gradient_accumulation_steps)
    use_amp: bool | str | None = field(default_factory=_default_use_amp)  # type: ignore[assignment]
    log_interval: int = field(default_factory=_default_log_interval)
    non_blocking: bool = field(default_factory=_default_non_blocking)

    # Function: __post_init__
    # Description: Reconcile dynamic defaults and validate configuration.
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


# Function: load_training_config
# Description: Load training configuration from mapping, path, or dataclass.
# Args:
#   source: Optional mapping, path, or TrainingConfig instance.
# Returns:
#   TrainingConfig with defaults merged.
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
    }
    kwargs = {key: payload.get(key) for key in valid_keys}
    return TrainingConfig(**kwargs)


# Function: _move_to_device
# Description: Recursively move tensors within a batch to target device.
# Args:
#   batch: Arbitrary batch structure.
#   device: Target torch.device.
#   non_blocking: Flag toggling async transfers.
# Returns:
#   Batch mirrored on the target device.
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


# Function: _split_batch
# Description: Extract model inputs and targets from batch.
# Args:
#   batch: Batch structure from dataloader.
# Returns:
#   Tuple containing inputs and targets.
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


# Function: _count_batch_items
# Description: Estimate number of examples contained in a batch.
# Args:
#   batch: Dataloader batch.
# Returns:
#   Integer representing sample count.
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


# Function: _try_len
# Description: Attempt to retrieve length from iterable when available.
def _try_len(iterable: Iterable[Any]) -> Optional[int]:
    try:
        return len(iterable)  # type: ignore[arg-type]
    except (TypeError, AttributeError):
        return None


# Function: _progress_iter
# Description: Wrap iterable with tqdm progress bar when library is available.
def _progress_iter(iterable: Iterable[Any], desc: str) -> tuple[Iterable[Any], Optional[Any]]:
    if tqdm is None:
        return iterable, None
    total = _try_len(iterable)
    bar = tqdm(iterable, desc=desc, total=total, leave=False)
    return bar, bar


# Function: train_one_epoch
# Description: Execute model training over a single epoch with gradient scaling support.
# Args:
#   model: Neural network under training.
#   dataloader: Iterable yielding training batches.
#   optimizer: Optimizer used for weight updates.
#   loss_fn: Loss function computing training loss.
#   config: Training configuration parameters.
#   scaler: Optional gradient scaler enabling AMP.
#   progress_desc: Optional string displayed on progress bar.
# Returns:
#   Dictionary containing aggregated training metrics.
def train_one_epoch(model        : nn.Module,
                    dataloader   : Iterable[Any],
                    optimizer    : Optimizer,
                    loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    config       : TrainingConfig,
                    scaler       : Optional[GradScaler] = None,
                    progress_desc: Optional[str] = None,
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
                # Perform gradient clipping and weight update when accumulation completes.
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
        "loss": total_loss / max(total_steps, 1),
        "loss_sum": total_loss,
        "batches": total_steps,
        "examples": total_examples,
    }


# Function: evaluate
# Description: Evaluate model on validation or test set without gradient updates.
# Args:
#   model: Model to evaluate.
#   dataloader: Iterable producing evaluation batches.
#   loss_fn: Loss function applied to predictions.
#   device: Target device for evaluation.
#   non_blocking: Whether to request non-blocking transfers.
#   progress_desc: Optional label for progress bar.
# Returns:
#   Dictionary containing aggregated evaluation metrics.
def evaluate(model        : nn.Module,
             dataloader   : Iterable[Any],
             loss_fn      : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             device       : str | torch.device,
             non_blocking : bool = True,
             progress_desc: Optional[str] = None,
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
                outputs = model(inputs)
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
        "loss": total_loss / max(total_steps, 1),
        "loss_sum": total_loss,
        "batches": total_steps,
        "examples": total_examples,
    }


class Trainer:
    # Function: __init__
    # Description: Construct trainer managing training loop and optional scheduler.
    def __init__(
        self,
        model     : nn.Module,
        optimizer : Optimizer,
        loss_fn   : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: Iterable[Any],
        config    : TrainingConfig,
        val_loader: Optional[Iterable[Any]] = None,
        scheduler : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
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

    # Function: fit
    # Description: Perform full training run across configured epochs.
    def fit(self) -> list[Dict[str, Any]]:
        for epoch in range(1, self.config.epochs + 1):
            train_metrics = train_one_epoch(model        = self.model,
                                            dataloader   = self.train_loader,
                                            optimizer    = self.optimizer,
                                            loss_fn      = self.loss_fn,
                                            config       = self.config,
                                            scaler       = self.scaler,
                                            progress_desc= f"Epoch {epoch} [train]",
                                           )

            val_metrics = None
            if self.val_loader is not None:
                val_metrics = evaluate(model        = self.model,
                                       dataloader   = self.val_loader,
                                       loss_fn      = self.loss_fn,
                                       device       = self.config.device,
                                       non_blocking = self.config.non_blocking,
                                       progress_desc= f"Epoch {epoch} [val]",
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

    # Function: _step_scheduler
    # Description: Advance learning-rate scheduler considering validation metrics.
    def _step_scheduler(self, val_metrics: Optional[Dict[str, float]]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_metrics is None:
                raise ValueError("Validation metrics required for ReduceLROnPlateau scheduler.")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()


# Function: fit
# Description: Convenience helper to instantiate trainer and run training loop.
# Args mirror Trainer initialisation.
def fit(model     : nn.Module,
        optimizer : Optimizer,
        loss_fn   : Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: Iterable[Any],
        config    : TrainingConfig,
        val_loader: Optional[Iterable[Any]] = None,
        scheduler : Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
       ) -> list[Dict[str, Any]]:
    trainer = Trainer(model       = model,
                      optimizer   = optimizer,
                      loss_fn     = loss_fn,
                      train_loader= train_loader,
                      config      = config,
                      val_loader  = val_loader,
                      scheduler   = scheduler,
                     )
    return trainer.fit()
