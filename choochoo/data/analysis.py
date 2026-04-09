"""Dataset intelligence: analysis, auto-bucketing, repeat calculation, smart sampling."""

from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class DatasetAnalyzer:
    """Analyze a dataset directory to inform training decisions.

    Auto-detects:
    - Number of samples
    - Resolution distribution
    - Aspect ratio distribution
    - Video sequence lengths
    - Recommended repeat count
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def analyze(
        self,
        target_steps: int = 2000,
        lr: float = 1e-4,
        rank: int = 16,
        batch_size: int = 1,
        target_exposure: float = 100.0,
    ) -> Dict[str, Any]:
        """Run full analysis. Returns summary dict."""
        image_files = self._find_files(IMAGE_EXTS)
        video_files = self._find_files(VIDEO_EXTS)

        result: Dict[str, Any] = {
            "num_images": len(image_files),
            "num_videos": len(video_files),
            "total_samples": len(image_files) + len(video_files),
        }

        if image_files:
            img_analysis = self._analyze_images(image_files[:500])  # Sample up to 500
            result.update(img_analysis)

        if video_files:
            vid_analysis = self._analyze_videos(video_files[:100])
            result.update(vid_analysis)

        result["recommended_repeats"] = self._calc_repeats(
            result["total_samples"], target_steps,
            lr=lr, rank=rank, batch_size=batch_size,
            target_exposure=target_exposure,
        )
        result["bucket_distribution"] = result.get("resolution_distribution", {})

        logger.info(
            f"Dataset analysis: {result['total_samples']} samples, "
            f"repeats={result['recommended_repeats']}"
        )
        return result

    def _find_files(self, exts: set) -> List[Path]:
        files = []
        for f in self.data_dir.rglob("*"):
            if f.suffix.lower() in exts:
                files.append(f)
        return sorted(files)

    def _analyze_images(self, files: List[Path]) -> Dict[str, Any]:
        resolutions = []
        aspect_ratios = []
        errors = 0
        for f in files:
            try:
                with Image.open(f) as img:
                    w, h = img.size
                    resolutions.append((h, w))
                    aspect_ratios.append(round(w / h, 2))
            except Exception:
                errors += 1

        if not resolutions:
            return {}

        # Bucket by 64-step resolution
        res_buckets = Counter(
            (round(h / 64) * 64, round(w / 64) * 64) for h, w in resolutions
        )
        top_buckets = dict(res_buckets.most_common(10))

        heights = [r[0] for r in resolutions]
        widths = [r[1] for r in resolutions]

        return {
            "resolution_distribution": top_buckets,
            "mean_height": sum(heights) / len(heights),
            "mean_width": sum(widths) / len(widths),
            "min_resolution": (min(heights), min(widths)),
            "max_resolution": (max(heights), max(widths)),
            "aspect_ratio_distribution": dict(Counter(aspect_ratios).most_common(10)),
            "load_errors": errors,
        }

    def _analyze_videos(self, files: List[Path]) -> Dict[str, Any]:
        frame_counts = []
        for f in files:
            try:
                import cv2
                cap = cv2.VideoCapture(str(f))
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_counts.append(count)
                cap.release()
            except Exception:
                pass

        if not frame_counts:
            return {"video_frame_counts": []}

        return {
            "video_frame_counts": frame_counts,
            "mean_video_frames": sum(frame_counts) / len(frame_counts),
            "max_video_frames": max(frame_counts),
            "min_video_frames": min(frame_counts),
        }

    @staticmethod
    def _calc_repeats(
        dataset_size: int,
        target_steps: int,
        lr: float = 1e-4,
        rank: int = 16,
        batch_size: int = 1,
        target_exposure: float = 100.0,
        base_lr: float = 1e-4,
        base_rank: int = 16,
    ) -> int:
        """LR/rank-aware dataset repeat count (from example_repeat_algo.py).

        Scales target exposure by the effective update magnitude:
          rank_scale  = sqrt(rank / base_rank)   — larger rank → more stable gradients
          lr_scale    = lr / (base_lr / rank_scale)
          adjusted    = target_exposure * lr_scale
          repeats     = (adjusted * dataset_size) / (max_steps * batch_size)
        """
        import math
        if dataset_size == 0:
            return 1
        rank_scale = math.sqrt(rank / base_rank)
        lr_scale = lr / (base_lr / rank_scale)
        adjusted_exposure = target_exposure * lr_scale
        repeats = (adjusted_exposure * dataset_size) / (target_steps * batch_size)
        logger.debug(
            "_calc_repeats: lr=%.2e rank=%d batch=%d → rank_scale=%.3f lr_scale=%.3f "
            "adjusted_exposure=%.1f → repeats=%d (raw=%.2f)",
            lr, rank, batch_size, rank_scale, lr_scale, adjusted_exposure,
            max(1, int(round(repeats))), repeats,
        )
        return max(1, int(round(repeats)))
