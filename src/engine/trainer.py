"""Unified optimizer, resume, early stopping and checkpoint loop."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Iterable, Mapping

import torch

from data.dataset import ForecastBatch
from data.normalization import NormalizationStats
from models.base import ForecastModel

from .checkpoint import save_checkpoint
from .evaluator import evaluate
from .model_execution import (
    ExecutionPlan,
    execute_training_backward,
)
from .precision import PrecisionPolicy, resolve_precision_policy
from .reproducibility import capture_rng_state, state_dict_hash


@dataclass(frozen=True)
class TrainResult:
    history: list[dict[str, Any]]
    best_epoch: int
    best_metric: float
    epochs_completed: int
    initial_state_hash: str
    first_step_loss: float | None
    last_step_loss: float | None
    epoch_seconds: list[float]
    update_seconds: list[float]
    stale_count: int
    train_batch_order: list[list[int]]


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
        dataloader_generators: Mapping[str, torch.Generator] | None = None,
        amp_enabled: bool | None = None,
        execution_plan: ExecutionPlan,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.model_name = model_name
        self.normalization = normalization
        self.output_dir = output_dir
        self.dataloader_generators = dict(dataloader_generators or {})
        self.execution_plan = execution_plan
        training = config.training
        amp_configured = bool(training["amp"] if amp_enabled is None else amp_enabled)
        self.precision: PrecisionPolicy = resolve_precision_policy(
            device=self.device,
            amp_configured=amp_configured,
            amp_dtype=str(training["amp_dtype"]),
            amp_cache_enabled=bool(training["amp_cache_enabled"]),
        )
        self.amp_enabled = self.precision.amp_effective
        if self.precision.amp_configured and self.device.type != "cuda":
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
        self.scaler = torch.amp.GradScaler("cuda", enabled=True) if self.amp_enabled else None
        self.first_step_loss: float | None = None
        self.last_step_loss: float | None = None
        self.epoch_seconds: list[float] = []
        self.update_seconds: list[float] = []
        self.train_batch_order: list[list[int]] = []

    def _autocast(self):
        return self.precision.autocast()

    def _update(self, batches: list[ForecastBatch]) -> float:
        if not batches:
            raise ValueError("optimizer update requires at least one micro-batch")
        self.train_batch_order.extend(
            [[int(value) for value in batch.starts.tolist()] for batch in batches]
        )
        update_started = time.perf_counter()
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        loss_name = str(self.config.training["loss"])
        backward = (
            (lambda contribution: self.scaler.scale(contribution).backward())
            if self.scaler is not None
            else (lambda contribution: contribution.backward())
        )
        execution = execute_training_backward(
            self.model,
            batches,
            device=self.device,
            plan=self.execution_plan,
            loss_name=loss_name,
            autocast=self._autocast,
            backward=backward,
        )
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
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
        value = float(execution.loss)
        if self.first_step_loss is None:
            self.first_step_loss = value
        self.last_step_loss = value
        self.update_seconds.append(time.perf_counter() - update_started)
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

    def _runtime_state(
        self,
        *,
        history: list[dict[str, Any]],
        epoch: int,
        best_epoch: int,
        best_metric: float,
        stale: int,
        initial_hash: str,
    ) -> dict[str, Any]:
        return {
            "rng": capture_rng_state(),
            "dataloader_generators": {
                name: generator.get_state() for name, generator in self.dataloader_generators.items()
            },
            "trainer": {
                "epoch": int(epoch),
                "best_epoch": int(best_epoch),
                "best_monitor": float(best_metric),
                "early_stopping_no_improvement_count": int(stale),
                "history": history,
                "initial_state_hash": initial_hash,
                "first_step_loss": self.first_step_loss,
                "last_step_loss": self.last_step_loss,
                "epoch_seconds": self.epoch_seconds,
                "update_seconds": self.update_seconds,
                "train_batch_order": self.train_batch_order,
            },
        }

    def _checkpoint_manifest(
        self,
        *,
        epoch: int,
        monitor: float,
        selection: Mapping[str, Any],
        best_epoch: int,
        best_metric: float,
        stale: int,
        history: list[dict[str, Any]],
        initial_hash: str,
        checkpoint_extra: dict[str, Any],
        is_last: bool,
    ) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "monitor": float(monitor),
            "monitor_name": selection["metric"],
            "checkpoint_selection": dict(selection),
            "model": self.model_name,
            "is_last": bool(is_last),
            "best_epoch": int(best_epoch),
            "best_monitor": float(best_metric),
            "early_stopping_no_improvement_count": int(stale),
            "history": history,
            "current_learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "initial_state_hash": initial_hash,
            "cli_overrides": checkpoint_extra.get("cli_overrides", {}),
            "resolved_config_identity": checkpoint_extra.get("resolved_config"),
            "model_config_identity": checkpoint_extra.get("model_config"),
            **checkpoint_extra,
        }

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
        resume_state: Mapping[str, Any] | None = None,
        checkpoint_extra: dict[str, Any] | None = None,
    ) -> TrainResult:
        training = self.config.training
        total_epochs = int(training["epochs"] if epochs is None else epochs)
        accumulation_steps = int(training["gradient_accumulation_steps"])
        selection = self.config.evaluation["checkpoint_selection"]
        mode = selection["mode"]
        state = dict(resume_state or {})
        best_metric = float(state.get("best_monitor", float("inf") if mode == "min" else float("-inf")))
        best_epoch = int(state.get("best_epoch", 0))
        stale = int(state.get("early_stopping_no_improvement_count", 0))
        history: list[dict[str, Any]] = list(state.get("history", []))
        self.first_step_loss = state.get("first_step_loss")
        self.last_step_loss = state.get("last_step_loss")
        self.epoch_seconds = [float(value) for value in state.get("epoch_seconds", [])]
        self.update_seconds = [float(value) for value in state.get("update_seconds", [])]
        self.train_batch_order = [
            [int(value) for value in batch]
            for batch in state.get("train_batch_order", [])
        ]
        initial_hash = str(state.get("initial_state_hash") or state_dict_hash(self.model.state_dict()))
        extra = dict(checkpoint_extra or {})
        extra["initial_state_hash"] = initial_hash

        for epoch in range(int(start_epoch), total_epochs + 1):
            epoch_started = time.perf_counter()
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
                execution_plan=self.execution_plan,
                physical_clip=bool(self.config.evaluation["physical_clip"]),
                physical_min_kw=self.config.evaluation["physical_min_kw"],
                physical_max_kw=self.config.evaluation["physical_max_kw"],
                max_batches=max_validation_batches,
            )
            monitor = float(validation.metrics["monitor"])
            self.scheduler.step(monitor)
            min_delta = float(training["early_stopping_min_delta"])
            improved = monitor < best_metric - min_delta if mode == "min" else monitor > best_metric + min_delta
            if improved:
                best_metric = monitor
                best_epoch = epoch
                stale = 0
            else:
                stale += 1
            self.epoch_seconds.append(time.perf_counter() - epoch_started)
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
            runtime_state = self._runtime_state(
                history=history,
                epoch=epoch,
                best_epoch=best_epoch,
                best_metric=best_metric,
                stale=stale,
                initial_hash=initial_hash,
            )
            manifest = self._checkpoint_manifest(
                epoch=epoch,
                monitor=monitor,
                selection=selection,
                best_epoch=best_epoch,
                best_metric=best_metric,
                stale=stale,
                history=history,
                initial_hash=initial_hash,
                checkpoint_extra=extra,
                is_last=False,
            )
            if improved:
                save_checkpoint(self.output_dir / "best.pt", self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, manifest=manifest, runtime_state=runtime_state)
            last_manifest = dict(manifest)
            last_manifest["is_last"] = True
            save_checkpoint(self.output_dir / "last.pt", self.model, optimizer=self.optimizer, scheduler=self.scheduler, scaler=self.scaler, manifest=last_manifest, runtime_state=runtime_state)
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
            epoch_seconds=self.epoch_seconds,
            update_seconds=self.update_seconds,
            stale_count=stale,
            train_batch_order=self.train_batch_order,
        )
