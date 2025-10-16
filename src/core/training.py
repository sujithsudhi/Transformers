from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Union

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau


@dataclass(slots=True)
class TrainingConfig:
    epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    gradient_clip_norm: float | None = None
    gradient_accumulation_steps: int = 1
    use_amp: bool = torch.cuda.is_available()
    log_interval: int = 50
    non_blocking: bool = True


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


def train_one_epoch(model: nn.Module,
                    dataloader: Iterable[Any],
                    optimizer: Optimizer,
                    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                    config: TrainingConfig,
                    scaler: Optional[GradScaler] = None,) -> Dict[str, float]:
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


def evaluate(model: nn.Module,
            dataloader: Iterable[Any],
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
            device: str | torch.device,
            non_blocking: bool = True,
        ) -> Dict[str, float]:
    
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

    def fit(self) -> list[Dict[str, Any]]:
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

    def _step_scheduler(self, val_metrics: Optional[Dict[str, float]]) -> None:
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, ReduceLROnPlateau):
            if val_metrics is None:
                raise ValueError("Validation metrics required for ReduceLROnPlateau scheduler.")
            self.scheduler.step(val_metrics["loss"])
        else:
            self.scheduler.step()


def fit(model: nn.Module,
        optimizer: Optimizer,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        train_loader: Iterable[Any],
        config: TrainingConfig,
        val_loader: Optional[Iterable[Any]] = None,
        scheduler: Optional[Union[_LRScheduler, ReduceLROnPlateau]] = None,
    ) -> list[Dict[str, Any]]:

    trainer = Trainer(model=model,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     train_loader=train_loader,
                     config=config,
                     val_loader=val_loader,
                     scheduler=scheduler)
    return trainer.fit()
