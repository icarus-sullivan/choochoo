"""Image dataset for LoRA training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from .dataset import BaseDataset

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImageDataset(BaseDataset):
    """Image + caption dataset for diffusion LoRA training.

    Expects:
        data_dir/
            image1.png
            image1.txt   (optional caption)
            image2.jpg
            ...

    Or a metadata.jsonl file with {file_name, caption} entries.
    """

    def __init__(
        self,
        data_dir: str,
        target_size: Tuple[int, int] = (1024, 1024),
        cache_dir: Optional[str] = None,
        repeats: int = 1,
        default_caption: str = "",
        center_crop: bool = True,
        random_flip: bool = False,
    ):
        super().__init__(data_dir, cache_dir, repeats)
        self.target_size = target_size
        self.default_caption = default_caption
        self.center_crop = center_crop
        self.random_flip = random_flip
        self._captions: Dict[str, str] = {}

        self._load_items(IMAGE_EXTS)
        self._load_captions()

    def _load_captions(self) -> None:
        """Load captions from .txt files or metadata.jsonl."""
        import json
        jsonl = self.data_dir / "metadata.jsonl"
        if jsonl.exists():
            with open(jsonl) as f:
                for line in f:
                    entry = json.loads(line.strip())
                    self._captions[entry.get("file_name", "")] = entry.get("caption", "")
            return

        for img_path in self._items:
            txt_path = img_path.with_suffix(".txt")
            if txt_path.exists():
                self._captions[img_path.name] = txt_path.read_text().strip()

    def _load_item(self, path: Path) -> Dict[str, Any]:
        img = Image.open(path).convert("RGB")
        original_size = (img.height, img.width)

        img = self._resize_and_crop(img, self.target_size)

        if self.random_flip and torch.rand(1).item() > 0.5:
            img = TF.hflip(img)

        # To tensor: [C, H, W] in [-1, 1]
        tensor = TF.to_tensor(img) * 2.0 - 1.0

        caption = self._captions.get(path.name, self.default_caption)

        return {
            "pixel_values": tensor,
            "caption": caption,
            "original_size": original_size,
            "target_size": self.target_size,
            "file_path": str(path),
        }

    def _resize_and_crop(
        self, img: Image.Image, target: Tuple[int, int]
    ) -> Image.Image:
        th, tw = target
        if self.center_crop:
            # Resize shortest side, center crop
            scale = max(tw / img.width, th / img.height)
            new_w = int(img.width * scale)
            new_h = int(img.height * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            left = (new_w - tw) // 2
            top = (new_h - th) // 2
            img = img.crop((left, top, left + tw, top + th))
        else:
            img = img.resize((tw, th), Image.LANCZOS)
        return img
