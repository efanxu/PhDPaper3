"""Unified optimizer, loss, scheduler, early stopping and checkpoint loop."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from data.dataset import ForecastBatch
from data.normalization import NormalizationStats
from models.base import ForecastModel

from .checkpoint import save_checkpoint
from .evaluator import EvaluationResult, evaluate
from .losses import ScoreAlignedHybridTerms, score_aligned_hybrid_terms
from .reproducibility import state_dict_hash


@dataclass(frozen=True)
class TrainResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_metric: float
    epochs_completed: int
    initial_state_hash: str
    first_step_loss: float | None
    last_step_loss: float | None


class Trainer:
    def __init__(
        self,
        model: ForecastModel,
        config: Any,
        *,
        device: torch.device | str,
        model_name: str,
        normalization: NormalizationStats,
        output_dir,
        amp_enabled: bool | None = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.model_name = model_name
        self.normalization = normalization
        self.output_dir = output_dir
        training = config.training
        self.amp_enabled = bool(training["amp"] if amp_enabled is None else amp_enabled)
        if self.amp_enabled and self.device.type != "cuda":
            raise RuntimeError("AMP training requires CUDA; CPU fallback is forbidden")
        betas = tuple(float(value) for value in training["betas"])
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(training["learning_rate"]),
            betas=betas,
            eps=float(training["epsilon"]),
            weight_decay=float(training["weight_decay"]),
            foreach=False,
            fused=False,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode=config.evaluation["checkpoint_selection"]["mode"],
            factor=float(training["scheduler_factor"]),
            patience=int(training["scheduler_patience"]),
            threshold=float(training["scheduler_threshold"]),
            threshold_mode=training["scheduler_threshold_mode"],
            min_lr=float(training["scheduler_min_lr"]),
        )
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=True)
            if self.amp_enabled
            else None
        )
        self.first_step_loss: float | None = None
        self.last_step_loss: float | None = None

    def _autocast(self):
        if not self.amp_enabled:
            return nullcontext()
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
            cache_enabled=bool(self.config.training["amp_cache_enabled"]),
        )

    def _update(self, batches: list[ForecastBatch]) -> float:
        if not batches:
            raise ValueError("optimizer update requires at least one micro-batch")
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        terms: ScoreAlignedHybridTerms | None = None
        for batch in batches:
            device_batch = batch.to(self.device)
            with self._autocast():
                prediction = self.model(device_batch.model_input())
                current = score_aligned_hybrid_terms(
                    prediction,
                    device_batch.target,
                    device_batch.target_mask,
                )
                terms = current if terms is None else terms + current
        assert terms is not None
        loss = terms.loss()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            loss.backward()
        training = self.config.training
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            float(training["gradient_clip"]),
            norm_type=float(training["gradient_clip_norm_type"]),
            error_if_nonfinite=bool(training["gradient_clip_error_if_nonfinite"]),
            foreach=bool(training["gradient_clip_foreach"]),
        )
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        value = float(loss.detach().float().cpu())
        if self.first_step_loss is None:
            self.first_step_loss = value
        self.last_step_loss = value
        return value

    def _groups(
        self,
        loader: Iterable[ForecastBatch],
        accumulation_steps: int,
        max_updates: int | None,
    ) -> Iterable[list[ForecastBatch]]:
        group: list[ForecastBatch] = []
        updates = 0
        for batch in loader:
            group.append(batch)
            if len(group) == accumulation_steps:
                yield group
                updates += 1
                group = []
                if max_updates is not None and updates >= max_updates:
                    return
        if group and (max_updates is None or updates < max_updates):
            yield group

    def fit(
        self,
        train_loader: Iterable[ForecastBatch],
        validation_loader: Iterable[ForecastBatch],
        *,
        horizons: tuple[int, ...],
        total_nodes: int,
        epochs: int | None = None,
        max_train_updates: int | None = None,
        max_validation_batches: int | None = None,
        start_epoch: int = 1,
        checkpoint_extra: dict[str, Any] | None = None,
    ) -> TrainResult:
        training = self.config.training
        epochs_to_run = int(training["epochs"] if epochs is None else epochs)
        accumulation_steps = int(training["gradient_accumulation_steps"])
        selection = self.config.evaluation["checkpoint_selection"]
        mode = selection["mode"]
        best_metric = float("inf") if mode == "min" else float("-inf")
        best_epoch = 0
        stale = 0
        history: list[dict[str, Any]] = []
        initial_hash = state_dict_hash(self.model.state_dict())
        for local_epoch in range(epochs_to_run):
            epoch = start_epoch + local_epoch
            losses = [
                self._update(group)
                for group in self._groups(train_loader, accumulation_steps, max_train_updates)
            ]
            if not losses:
                raise ValueError("training loader produced no optimizer updates")
            validation = evaluate(
                self.model,
                validation_loader,
                device=self.device,
                normalization=self.normalization,
                horizons=horizons,
                total_nodes=total_nodes,
                physical_clip=bool(self.config.evaluation["physical_clip"]),
                physical_min_kw=self.config.evaluation["physical_min_kw"],
                physical_max_kw=self.config.evaluation["physical_max_kw"],
                max_batches=max_validation_batches,
            )
            monitor = float(validation.metrics["monitor"])
            self.scheduler.step(monitor)
            min_delta = float(training["early_stopping_min_delta"])
            improved = (
                monitor < best_metric - min_delta
                if mode == "min"
                else monitor > best_metric + min_delta
            )
            if improved:
                best_metric = monitor
                best_epoch = epoch
                stale = 0
                save_checkpoint(
                    self.output_dir / "best.pt",
                    self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    scaler=self.scaler,
                    manifest={
                        "epoch": epoch,
                        "monitor": monitor,
                        "monitor_name": selection["metric"],
                        "checkpoint_selection": selection,
                        "model": self.model_name,
                        **(checkpoint_extra or {}),
                    },
                )
            else:
                stale += 1
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": float(sum(losses) / len(losses)),
                    "validation": validation.metrics,
                    "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
                    "checkpoint_selected": bool(improved),
                    "train_updates": len(losses),
                }
            )
            save_checkpoint(
                self.output_dir / "last.pt",
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                scaler=self.scaler,
                manifest={
                    "epoch": epoch,
                    "monitor": monitor,
                    "monitor_name": selection["metric"],
                    "model": self.model_name,
                    "is_last": True,
                    **(checkpoint_extra or {}),
                },
            )
            if stale >= int(training["early_stopping_patience"]):
                break
        if best_epoch == 0:
            raise RuntimeError("training did not select a best checkpoint")
        return TrainResult(
            history=history,
            best_epoch=best_epoch,
            best_metric=float(best_metric),
            epochs_completed=len(history),
            initial_state_hash=initial_hash,
            first_step_loss=self.first_step_loss,
            last_step_loss=self.last_step_loss,
        )
