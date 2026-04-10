"""Highly optimized training loop with async loading, minimal sync points, and full monitoring."""

from __future__ import annotations

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
try:
    from torch.amp import GradScaler  # PyTorch 2.x+
    _SCALER_DEVICE: tuple = ("cuda",)
except ImportError:
    from torch.cuda.amp import GradScaler  # type: ignore[assignment]  # pre-2.x
    _SCALER_DEVICE = ()
from torch.utils.data import DataLoader

from ..autotuner.profiler import TrainingProfiler
from ..autotuner.vram24 import VRAM24Optimizer
from ..checkpointing.checkpoint import CheckpointManager
from ..logging.metrics import MetricsLogger
from ..logging.sqlite_writer import SQLiteMetricsWriter, TrainingPhase
from ..sampling.sampler import TrainingSampler
from .convergence import ConvergenceDetector, ConvergenceState
from .ema import EMAModel
from .optimizer import build_optimizer
from .scheduler import build_scheduler

_MODEL_PREFIX = {
    "wan22":      "wan",
    "wan21":      "wan",
    "wan_i2v":    "wan",
    "wan22_dual": "wan",
    "qwen_edit":  "qwen",
    "qwen":       "qwen",
    "ltx2":       "ltx2",
}


def _training_phase(
    step: int,
    warmup_steps: int,
    max_steps: int,
    conv_state: ConvergenceState,
) -> TrainingPhase:
    if step < warmup_steps:
        return TrainingPhase.WARMUP
    burnin_end = warmup_steps + max(warmup_steps, max_steps // 10)
    if step < burnin_end:
        return TrainingPhase.BURNIN
    if conv_state.plateau_detected:
        return TrainingPhase.CONVERGENCE
    return TrainingPhase.TRAINING

logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator.

    Handles:
    - Training loop with gradient accumulation
    - Mixed precision via GradScaler
    - EMA tracking
    - Step-level profiling and bottleneck detection
    - Convergence detection and auto-stop
    - Checkpointing at regular intervals
    - WandB/TensorBoard logging
    """

    def __init__(
        self,
        cfg: DictConfig,
        model_adapter,
        dataloader: DataLoader,
        rank: int = 0,
        world_size: int = 1,
    ):
        self.cfg = cfg
        self.adapter = model_adapter
        self.dataloader = dataloader
        self.rank = rank
        self.world_size = world_size
        self.is_main = rank == 0

        self.device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
        self.global_step = 0
        self._vram24 = VRAM24Optimizer()

        # Built lazily in setup()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler = None
        self.scaler: Optional[GradScaler] = None
        self.ema: Optional[EMAModel] = None
        self.convergence: Optional[ConvergenceDetector] = None
        self.profiler: Optional[TrainingProfiler] = None
        self.ckpt_manager: Optional[CheckpointManager] = None
        self.metrics_logger: Optional[MetricsLogger] = None
        self.sqlite_writer: Optional[SQLiteMetricsWriter] = None
        self.sampler: Optional[TrainingSampler] = None

    def setup(self) -> None:
        """Initialize all training components."""
        cfg_t = self.cfg.training
        cfg_p = self.cfg.performance

        trainable_params = self.adapter.get_trainable_params()

        self.optimizer = build_optimizer(
            trainable_params,
            optimizer_type="adamw",
            lr=float(cfg_t.learning_rate),
            weight_decay=1e-2,
            fused=cfg_p.get("fused_optimizer", True),
        )

        self.scheduler = build_scheduler(
            self.optimizer,
            scheduler_type=cfg_t.lr_scheduler,
            num_warmup_steps=int(cfg_t.lr_warmup_steps),
            num_training_steps=int(cfg_t.max_steps),
        )

        # Mixed precision
        use_fp16 = self.cfg.model.dtype == "fp16"
        self.scaler = GradScaler(*_SCALER_DEVICE, enabled=use_fp16)

        # EMA
        if cfg_t.get("ema", False):
            self.ema = EMAModel(
                trainable_params,
                decay=float(cfg_t.get("ema_decay", 0.9999)),
                device=self.device,
            )

        # Convergence detector
        self.convergence = ConvergenceDetector(
            patience=int(cfg_t.get("convergence_patience", 200)),
            keep_top_k=int(cfg_t.get("keep_top_k_checkpoints", 3)),
            auto_stop=bool(cfg_t.get("auto_stop", False)),
            warmup_steps=int(cfg_t.get("lr_warmup_steps", 0)),
        )

        # Profiler
        batch_size = int(cfg_t.get("batch_size", 1))
        self.profiler = TrainingProfiler(batch_size=batch_size)

        # Checkpoint manager
        output_dir = self.cfg.logging.get("output_dir", "./output")
        self.ckpt_manager = CheckpointManager(
            output_dir=output_dir,
            cfg=self.cfg,
            keep_last_n=int(cfg_t.get("keep_top_k_checkpoints", 3)),
            save_lora_only=True,
        )

        # Metrics logger
        self.metrics_logger = MetricsLogger(
            cfg=self.cfg,
            output_dir=output_dir,
            enabled=self.is_main,
        )
        self.metrics_logger.setup()

        # SQLite metrics DB (main process only, async background writer)
        if self.is_main:
            _prefix = _MODEL_PREFIX.get(self.cfg.model.type, self.cfg.model.type)
            _name = self.cfg.get("name", "run")
            db_path = str(Path(output_dir) / f"{_prefix}-{_name}.db")
            self.sqlite_writer = SQLiteMetricsWriter(db_path)
            self.sqlite_writer.write_meta("total_steps", str(int(cfg_t.max_steps)))
            self.sqlite_writer.write_meta("model_type", self.cfg.model.type)

        # Training-time sampler (main process only)
        self.sampler = TrainingSampler(self.cfg, self.adapter, output_dir=output_dir)

    def train(self) -> None:
        """Run the full training loop."""
        if self.optimizer is None:
            self.setup()

        cfg_t = self.cfg.training
        max_steps = int(cfg_t.max_steps)
        grad_accum = int(cfg_t.get("gradient_accumulation_steps", 1))
        save_every = int(cfg_t.get("save_every", 500))
        log_every = int(cfg_t.get("log_every", 10))
        profile_steps = int(self.cfg.logging.get("profile_steps", 0))

        dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "fp32": torch.float32,
        }.get(self.cfg.model.dtype, torch.float32)

        self.adapter.model.train()

        logger.info(f"Starting training: max_steps={max_steps}, grad_accum={grad_accum}")
        logger.info(f"Trainable params: {self.adapter.trainable_param_count/1e6:.2f}M")

        data_iter = self._infinite_loader()
        self.optimizer.zero_grad(set_to_none=True)

        for step in range(self.global_step, max_steps):
            self.profiler.start_data()
            batch = next(data_iter)
            self.profiler.end_data()

            batch = self._to_device(batch, dtype)

            # Enable PyTorch profiler for first N steps
            profile_ctx = self._maybe_profile(step, profile_steps)

            with profile_ctx:
                loss, metrics = self._train_step(batch, grad_accum, dtype, step)

            self.global_step += 1
            self.profiler.record_step(step)

            if self.ema is not None:
                self.ema.step()

            # Convergence check
            state = self.convergence.update(step, loss)
            if state.is_best and self.is_main:
                self.ckpt_manager.save(
                    step=step + 1,
                    model=self.adapter.model,
                    injector=getattr(self.adapter, "_lora_injector", None),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={"loss": loss},
                    tag="best",
                )

            # Regular checkpoint
            if self.is_main and (step + 1) % save_every == 0:
                self.ckpt_manager.save(
                    step=step + 1,
                    model=self.adapter.model,
                    injector=getattr(self.adapter, "_lora_injector", None),
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics={"loss": loss},
                )
                # Clear allocator fragmentation after serialisation-heavy save
                if self.cfg.performance.get("vram_fragmentation_reduction", True):
                    self._vram24.clear_fragmentation()

            # Inference samples
            if self.is_main and self.sampler.should_sample(step + 1):
                self.sampler.run(step + 1)
                if self.cfg.performance.get("vram_fragmentation_reduction", True):
                    self._vram24.clear_fragmentation()

            # SQLite: write every step (non-blocking, background thread)
            if self.is_main and self.sqlite_writer is not None:
                phase = _training_phase(
                    step=step,
                    warmup_steps=int(self.cfg.training.lr_warmup_steps),
                    max_steps=max_steps,
                    conv_state=state,
                )
                self.sqlite_writer.log(
                    step=step + 1,
                    loss=loss,
                    lr=self.scheduler.get_last_lr()[0],
                    phase=phase,
                    wall_time=time.time(),
                    grad_norm=metrics.get("grad_norm"),
                )
                self.sqlite_writer.log_convergence(
                    step=step + 1,
                    is_best=state.is_best,
                    best_loss=self.convergence._best_loss if self.convergence._best_loss != float("inf") else None,
                    best_step=self.convergence._best_step,
                    steps_since_best=state.steps_since_best,
                    plateau=state.plateau_detected,
                    overfit=state.overfit_detected,
                )

            # Logging
            if self.is_main and (step + 1) % log_every == 0:
                prof_summary = self.profiler.summarize()
                log_metrics = {
                    "loss": loss,
                    "lr": self.scheduler.get_last_lr()[0],
                    "samples_per_sec": prof_summary.samples_per_sec,
                    "gpu_util": prof_summary.mean_gpu_util,
                    **{k: v for k, v in metrics.items() if k != "loss"},
                }
                self.metrics_logger.log(step, log_metrics)

                if prof_summary.suggestions:
                    for suggestion in prof_summary.suggestions:
                        logger.warning(f"  Bottleneck: {suggestion}")

                logger.info(
                    f"[{step+1}/{max_steps}] loss={loss:.4f} "
                    f"lr={log_metrics['lr']:.2e} "
                    f"{prof_summary.samples_per_sec:.1f} samples/s"
                )

            # Auto-stop
            if self.convergence.should_stop():
                logger.info(
                    f"Auto-stopping at step {step + 1}: convergence detected. "
                    f"Best checkpoint: step {self.convergence.best_step}"
                )
                break

        # Final save
        if self.is_main:
            self.ckpt_manager.save(
                step=self.global_step,
                model=self.adapter.model,
                injector=getattr(self.adapter, "_lora_injector", None),
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                metrics={},
                tag="final",
            )
            logger.info(f"\n{self.convergence.summary()}")

        self.metrics_logger.close()
        if self.sqlite_writer is not None:
            self.sqlite_writer.close()

    def _train_step(
        self,
        batch: Dict[str, Any],
        grad_accum: int,
        dtype: torch.dtype,
        step: int,
    ) -> tuple:
        """Single forward+backward step with gradient accumulation."""
        use_autocast = dtype in (torch.bfloat16, torch.float16)
        amp_dtype = dtype if use_autocast else None

        self.profiler.start_forward()
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_autocast):
            outputs = self.adapter.forward(batch)
            loss = outputs["loss"]
            scaled_loss = loss / grad_accum
        self.profiler.end_forward()

        self.profiler.start_backward()
        self.scaler.scale(scaled_loss).backward()
        self.profiler.end_backward()

        # Gradient accumulation: step every N micro-steps
        metrics = {k: v.item() if hasattr(v, "item") else v
                   for k, v in outputs.items() if k != "pred"}

        is_accum_step = (step + 1) % grad_accum == 0
        if is_accum_step:
            self.profiler.start_optimizer()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.adapter.get_trainable_params(), max_norm=1.0
            ).item()
            metrics["grad_norm"] = grad_norm
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.profiler.end_optimizer()

        return loss.item(), metrics

    def _to_device(self, batch: Dict[str, Any], dtype: torch.dtype) -> Dict[str, Any]:
        """Move batch tensors to device with correct dtype, non-blocking."""
        out = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                if v.is_floating_point():
                    out[k] = v.to(self.device, dtype=dtype, non_blocking=True)
                else:
                    out[k] = v.to(self.device, non_blocking=True)
            else:
                out[k] = v
        return out

    def _infinite_loader(self) -> Iterator:
        """Infinite iterator over the dataloader."""
        while True:
            for batch in self.dataloader:
                yield batch

    def _maybe_profile(self, step: int, profile_steps: int) -> contextlib.AbstractContextManager:
        if profile_steps <= 0 or not self.is_main:
            return contextlib.nullcontext()

        profile_dir = self.cfg.logging.get("profile_output", "./profiler")
        Path(profile_dir).mkdir(parents=True, exist_ok=True)

        # Use schedule: skip=1 warmup, wait=1, warmup=1, active=profile_steps
        schedule = torch.profiler.schedule(wait=1, warmup=1, active=profile_steps, repeat=1)

        return torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=schedule,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(profile_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        )

    def resume(self, checkpoint_path: str) -> None:
        """Resume training from a checkpoint."""
        state = self.ckpt_manager.load(
            checkpoint_path,
            model=self.adapter.model,
            injector=getattr(self.adapter, "_lora_injector", None),
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )
        self.global_step = state.get("step", 0)
        logger.info(f"Resumed from step {self.global_step}")
