"""Video frame sequence dataset for WAN 2.2 temporal training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .dataset import BaseDataset

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class VideoDataset(BaseDataset):
    """Video dataset that loads frame sequences as temporal tensors.

    Supports:
    - MP4/AVI/MOV/MKV/WEBM videos
    - Frame stride sampling
    - Temporal bucketing by sequence length
    - Caption loading from .txt or metadata.jsonl
    """

    def __init__(
        self,
        data_dir: str,
        num_frames: int = 16,
        frame_stride: int = 1,
        target_size: Tuple[int, int] = (512, 512),
        cache_dir: Optional[str] = None,
        repeats: int = 1,
        default_caption: str = "",
        random_start: bool = True,
    ):
        super().__init__(data_dir, cache_dir, repeats)
        self.num_frames = num_frames
        self.frame_stride = frame_stride
        self.target_size = target_size
        self.default_caption = default_caption
        self.random_start = random_start
        self._captions: Dict[str, str] = {}

        self._load_items(VIDEO_EXTS)
        self._load_captions()

    def _load_captions(self) -> None:
        jsonl = self.data_dir / "metadata.jsonl"
        if jsonl.exists():
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    self._captions[entry.get("file_name", "")] = entry.get("caption", "")
            return
        for vid_path in self._items:
            txt = vid_path.with_suffix(".txt")
            if txt.exists():
                self._captions[vid_path.name] = txt.read_text().strip()

    def _load_item(self, path: Path) -> Dict[str, Any]:
        frames = self._extract_frames(path)
        caption = self._captions.get(path.name, self.default_caption)
        return {
            "pixel_values": frames,         # [C, T, H, W]
            "source_frame": frames[:, 0],   # [C, H, W] — first frame for I2V conditioning
            "caption": caption,
            "file_path": str(path),
            "num_frames": frames.shape[1],
        }

    def _extract_frames(self, path: Path) -> torch.Tensor:
        """Extract num_frames frames from video, return [C, T, H, W] tensor."""
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            needed = self.num_frames * self.frame_stride
            if total_frames <= needed:
                start = 0
            elif self.random_start:
                start = random.randint(0, total_frames - needed)
            else:
                start = 0

            cap.set(cv2.CAP_PROP_POS_FRAMES, start)

            frames = []
            frame_idx = 0
            while len(frames) < self.num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % self.frame_stride == 0:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil = Image.fromarray(frame)
                    pil = pil.resize(
                        (self.target_size[1], self.target_size[0]), Image.LANCZOS
                    )
                    t = TF.to_tensor(pil) * 2.0 - 1.0
                    frames.append(t)
                frame_idx += 1
            cap.release()

            # Pad if needed
            while len(frames) < self.num_frames:
                frames.append(frames[-1] if frames else torch.zeros(3, *self.target_size))

            stacked = torch.stack(frames, dim=1)  # [C, T, H, W]
            return stacked

        except Exception as e:
            # Return zeros if video cannot be read
            return torch.zeros(3, self.num_frames, *self.target_size)
