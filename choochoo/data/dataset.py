"""Base dataset class with caching, bucketing integration, and smart sampling."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    """Base dataset supporting:
    - RAM and disk latent caching
    - Dataset repeats
    - Extensible item loading
    """

    def __init__(
        self,
        data_dir: str,
        cache_dir: Optional[str] = None,
        repeats: int = 1,
        transform: Optional[Callable] = None,
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.repeats = max(1, repeats)
        self.transform = transform

        self._items: List[Path] = []
        self._cache: Dict[int, Any] = {}
        self._disk_cache_enabled = cache_dir is not None

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_items(self, extensions: set) -> None:
        """Scan data_dir for files with given extensions."""
        self._items = sorted(
            f for f in self.data_dir.rglob("*") if f.suffix.lower() in extensions
        )
        if not self._items:
            raise RuntimeError(f"No files found in {self.data_dir} with exts {extensions}")

    def __len__(self) -> int:
        return len(self._items) * self.repeats

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        real_idx = idx % len(self._items)

        # Check RAM cache first
        if real_idx in self._cache:
            return self._cache[real_idx]

        # Check disk cache
        if self._disk_cache_enabled:
            cached = self._load_disk_cache(real_idx)
            if cached is not None:
                self._cache[real_idx] = cached
                return cached

        item = self._load_item(self._items[real_idx])
        if self.transform is not None:
            item = self.transform(item)

        if self._disk_cache_enabled:
            self._save_disk_cache(real_idx, item)
        self._cache[real_idx] = item
        return item

    def _load_item(self, path: Path) -> Dict[str, Any]:
        raise NotImplementedError

    def _cache_key(self, idx: int) -> str:
        return hashlib.md5(str(self._items[idx]).encode()).hexdigest()

    def _disk_cache_path(self, idx: int) -> Path:
        return self.cache_dir / f"{self._cache_key(idx)}.pt"

    def _load_disk_cache(self, idx: int) -> Optional[Dict[str, Any]]:
        p = self._disk_cache_path(idx)
        if p.exists():
            try:
                return torch.load(p, map_location="cpu", weights_only=True)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _save_disk_cache(self, idx: int, item: Dict[str, Any]) -> None:
        p = self._disk_cache_path(idx)
        try:
            torch.save(item, p)
        except Exception:
            pass

    def prefetch_all(self, num_workers: int = 4) -> None:
        """Pre-cache all items in parallel using ThreadPoolExecutor."""
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import tqdm

        indices = list(range(len(self._items)))
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(self.__getitem__, i): i for i in indices}
            for _ in tqdm.tqdm(
                as_completed(futures), total=len(futures), desc="Prefetching"
            ):
                pass
