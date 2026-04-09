"""Training-time sampler: run inference at checkpoint intervals to monitor LoRA quality.

Sampling logic lives in each model adapter (adapter.sample()). This class is a thin
coordinator that checks whether sampling should occur, delegates to the adapter, and
saves the returned results to disk.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class TrainingSampler:
    """Runs inference at regular training intervals and saves outputs.

    Delegates to adapter.sample() — each adapter owns its pipeline construction
    and GPU movement, returning a list of result dicts. This class owns saving.

    Result dict format:
        {"mime": "video/mp4", "data": [PIL.Image, ...]}   # video frames
        {"mime": "image/png", "data": PIL.Image}           # single image

    adapter.sample() returns None if the adapter doesn't support sampling.
    """

    def __init__(self, cfg: DictConfig, adapter: Any, output_dir: str) -> None:
        self.cfg = cfg
        self.adapter = adapter
        self.samples_dir = Path(output_dir) / "samples"
        self.sample_cfg = cfg.get("sample", {})

    def should_sample(self, step: int) -> bool:
        s = self.sample_cfg
        if not s or not s.get("prompts"):
            return False
        if not s.get("enabled", True):
            return False
        sample_every = s.get("sample_every", None)
        if sample_every is None:
            sample_every = int(self.cfg.training.get("save_every", 500))
        return step % int(sample_every) == 0

    def run(self, step: int) -> None:
        logger.info("TrainingSampler: generating samples at step %d", step)
        try:
            results = self.adapter.sample(self.sample_cfg)
            if results is None:
                logger.debug("TrainingSampler: adapter returned None, sampling not supported")
                return
            self.samples_dir.mkdir(parents=True, exist_ok=True)
            for i, result in enumerate(results):
                self._save(result, i, step)
        except Exception as e:
            logger.error("TrainingSampler: sampling failed at step %d: %s", step, e, exc_info=True)

    def _save(self, result: dict, idx: int, step: int) -> None:
        mime = result["mime"]
        data = result["data"]
        if mime == "video/mp4":
            path = self.samples_dir / f"sample_{idx:02d}_{step}.mp4"
            self._save_video(data, path)
        elif mime == "image/png":
            path = self.samples_dir / f"sample_{idx:02d}_{step}.png"
            data.save(path)
            logger.info("Saved sample: %s", path)
        else:
            logger.warning("TrainingSampler: unknown mime type %s, skipping", mime)

    def _save_video(self, frames: list, path: Path, fps: int = 16) -> None:
        try:
            import imageio
            imageio.mimwrite(str(path), [np.array(f) for f in frames], fps=fps)
            logger.info("Saved sample: %s", path)
        except Exception as e:
            logger.warning("imageio unavailable (%s); saving individual PNG frames instead", e)
            stem = path.with_suffix("")
            for i, frame in enumerate(frames):
                frame.save(f"{stem}_frame{i:04d}.png")
