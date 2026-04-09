"""Lightweight metrics logging: TensorBoard + WandB with negligible overhead."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Unified metrics logger supporting TensorBoard and WandB.

    Designed to be low-overhead: logging writes happen asynchronously
    and are only materialized on .log() calls, not every step.
    """

    def __init__(self, cfg: DictConfig, output_dir: str, enabled: bool = True):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.enabled = enabled

        self._tb_writer = None
        self._wandb_run = None
        self._backend: str = cfg.logging.get("backend", "tensorboard")

    def setup(self) -> None:
        if not self.enabled:
            return

        log_dir = self.cfg.logging.get("log_dir", str(self.output_dir / "logs"))
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        if self._backend in ("tensorboard", "both"):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(log_dir=log_dir)
                logger.info(f"TensorBoard logging: {log_dir}")
            except ImportError:
                logger.warning("tensorboard not installed; skipping")

        if self._backend in ("wandb", "both"):
            wandb_project = self.cfg.logging.get("wandb_project")
            if wandb_project:
                try:
                    import wandb
                    run_name = self.cfg.logging.get("wandb_run_name")
                    self._wandb_run = wandb.init(
                        project=wandb_project,
                        name=run_name,
                        config=self.cfg,
                        resume="allow",
                    )
                    logger.info(f"WandB logging: project={wandb_project}")
                except ImportError:
                    logger.warning("wandb not installed; skipping")
                except Exception as e:
                    logger.warning(f"WandB init failed: {e}")

    def log(self, step: int, metrics: Dict[str, Any]) -> None:
        if not self.enabled:
            return

        if self._tb_writer is not None:
            for k, v in metrics.items():
                try:
                    self._tb_writer.add_scalar(k, float(v), global_step=step)
                except (TypeError, ValueError):
                    pass

        if self._wandb_run is not None:
            try:
                self._wandb_run.log(
                    {k: float(v) for k, v in metrics.items() if _is_scalar(v)},
                    step=step,
                )
            except Exception:
                pass

    def log_image(self, step: int, key: str, image) -> None:
        """Log a sample image (PIL or tensor)."""
        if not self.enabled:
            return
        if self._tb_writer is not None:
            try:
                import torchvision
                if hasattr(image, "shape"):
                    self._tb_writer.add_image(key, image, global_step=step)
            except Exception:
                pass
        if self._wandb_run is not None:
            try:
                import wandb
                import numpy as np
                self._wandb_run.log({key: wandb.Image(image)}, step=step)
            except Exception:
                pass

    def close(self) -> None:
        if self._tb_writer is not None:
            self._tb_writer.close()
        if self._wandb_run is not None:
            try:
                self._wandb_run.finish()
            except Exception:
                pass


def _is_scalar(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False
