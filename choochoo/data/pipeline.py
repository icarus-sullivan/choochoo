"""Data pipeline: builds DataLoader with bucketing, caching, and auto-tuned workers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .analysis import DatasetAnalyzer
from .bucketing import BucketedBatchSampler, ResolutionBucketer
from .image import ImageDataset
from .video import VideoDataset

logger = logging.getLogger(__name__)


def _load_encoded_latents(item: Dict[str, Any], cache_dir) -> Dict[str, Any] | None:
    """Load pre-encoded latents from cache if available."""
    if cache_dir is None or "file_path" not in item:
        return None
    import hashlib
    from pathlib import Path
    cache_key = hashlib.md5(item["file_path"].encode()).hexdigest()
    cache_file = Path(cache_dir) / f"{cache_key}.pt"
    if cache_file.exists():
        try:
            return torch.load(cache_file, map_location="cpu", weights_only=True)
        except Exception:
            return None
    return None


def edit_collate_fn(batch, cache_dir=None, adapter=None):
    """Collate function for qwen_edit datasets.

    If items contain pre-encoded latents (target_latents, encoder_hidden_states),
    stack tensors directly. Otherwise, convert ImageDataset-style items
    (pixel_values, caption) to adapter input format via prepare_batch.
    """
    # Try to load pre-encoded latents from disk cache for raw items
    has_latents = isinstance(batch[0], dict) and "target_latents" in batch[0]

    if not has_latents and cache_dir is not None:
        # Check if cached latents exist for any item
        loaded = []
        for item in batch:
            cached = _load_encoded_latents(item, cache_dir)
            if cached is not None:
                loaded.append(cached)
            else:
                loaded.append(item)

        has_loaded_latents = any("target_latents" in x for x in loaded)
        if all("target_latents" in x for x in loaded):
            has_latents = True
            batch = loaded

    if has_latents:
        target_latents = torch.stack([item["target_latents"] for item in batch])

        # encoder_hidden_states have variable sequence length (different captions
        # tokenize to different lengths). Pad to the batch maximum with zeros.
        # The encoder_attention_mask is padded with zeros (0 = ignore, 1 = attend).
        hs_list = [item["encoder_hidden_states"] for item in batch]
        has_mask = batch[0].get("encoder_attention_mask") is not None
        mask_list = [item["encoder_attention_mask"] for item in batch] if has_mask else None

        max_seq = max(h.shape[0] for h in hs_list)
        if any(h.shape[0] != max_seq for h in hs_list):
            padded_hs = []
            padded_mask = [] if mask_list is not None else None
            for idx, h in enumerate(hs_list):
                pad_len = max_seq - h.shape[0]
                if pad_len:
                    h = torch.cat([h, h.new_zeros(pad_len, h.shape[1])], dim=0)
                    if mask_list is not None:
                        m = mask_list[idx]
                        m = torch.cat([m, m.new_zeros(pad_len)], dim=0)
                        padded_mask.append(m)
                else:
                    if mask_list is not None:
                        padded_mask.append(mask_list[idx])
                padded_hs.append(h)
            hs_list = padded_hs
            if mask_list is not None:
                mask_list = padded_mask

        encoder_hidden_states = torch.stack(hs_list)

        # Logit-normal timestep sampling in [0, 1].
        # t = sigmoid(N(0,1)) concentrates mass at mid-trajectory (t≈0.5) where the
        # model has the most to learn, vs uniform which wastes capacity on near-clean
        # (t≈0) and near-noise (t≈1) samples. Clipped to avoid degenerate endpoints.
        b = target_latents.shape[0]
        timesteps = torch.sigmoid(torch.randn(b)).clamp(0.001, 0.999)
        out = {
            "target_latents": target_latents,
            "encoder_hidden_states": encoder_hidden_states,
            "timesteps": timesteps,
        }
        if mask_list is not None:
            out["encoder_attention_mask"] = torch.stack(mask_list)
        # Pass through source_latents if present
        if batch[0].get("source_latents") is not None:
            out["source_latents"] = torch.stack([item["source_latents"] for item in batch])
        return out

    # Raw ImageDataset items: collate then delegate to adapter.prepare_batch
    collated = {k: [item[k] for item in batch] for k in batch[0]}
    if adapter is not None and hasattr(adapter, "prepare_batch"):
        return adapter.prepare_batch(collated)
    return collated




class DataPipeline:
    """Builds and manages the data pipeline for training.

    Handles:
    - Dataset type selection (image/video/edit)
    - Dataset analysis and repeat calculation
    - Resolution bucketing
    - DataLoader construction with optimal worker config
    """

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self._dataset: Optional[Dataset] = None
        self._dataloader: Optional[DataLoader] = None
        self._model_type: str = "wan22"

    def setup(self, model_type: str = "wan22", adapter=None) -> DataLoader:
        """Build and return the training DataLoader."""
        self._model_type = model_type
        cfg_data = self.cfg.data
        cfg_training = self.cfg.training
        data_dir = cfg_data.get("data_dir", "./data")

        # Analyze dataset
        logger.info(f"Analyzing dataset at {data_dir}")
        try:
            analyzer = DatasetAnalyzer(data_dir)
            _lr = float(cfg_training.get("learning_rate", 1e-4))
            _lora_cfg = self.cfg.get("lora", None)
            _rank = int(_lora_cfg.get("rank", 16) if _lora_cfg is not None else 16)
            _bs_val = cfg_training.get("batch_size", 1)
            _batch = 1 if _bs_val == "auto" else int(_bs_val)
            _target_exposure = 0.95 if model_type in ("qwen", "qwen_edit") else 100.0
            analysis = analyzer.analyze(
                target_steps=int(cfg_training.max_steps),
                lr=_lr,
                rank=_rank,
                batch_size=_batch,
                target_exposure=_target_exposure,
            )
            logger.info(
                f"Dataset: {analysis['total_samples']} samples, "
                f"recommended_repeats={analysis['recommended_repeats']} "
                f"(lr={_lr:.2e} rank={_rank} batch={_batch})"
            )
        except Exception as e:
            logger.warning(f"Dataset analysis failed: {e}")
            analysis = {"total_samples": 0, "recommended_repeats": 1}

        repeat_value = cfg_data.get("repeat_dataset", "auto")
        if repeat_value == "auto":
            repeats = analysis.get("recommended_repeats")
            repeats = int(repeats) if repeats is not None else 1
        else:
            repeats = int(repeat_value)

        target_size = (
            int(cfg_data.get("max_resolution", 1024)),
            int(cfg_data.get("max_resolution", 1024)),
        )

        # Build dataset
        self._dataset = self._build_dataset(
            model_type, data_dir, target_size, repeats, cfg_data
        )

        # Pre-encode latents for qwen/qwen_edit (one-pass VAE + text encoder)
        if model_type in ("qwen_edit", "qwen") and cfg_data.get("cache_latents") and adapter is not None:
            has_vae = getattr(adapter, "vae", None) is not None
            if has_vae:
                self._encode_dataset_for_edit(self._dataset, adapter)
            else:
                logger.warning("VAE not loaded — skipping latent pre-encoding")

        # Build DataLoader
        self._dataloader = self._build_dataloader(
            cfg_data, cfg_training, adapter=adapter
        )
        return self._dataloader

    def _encode_dataset_for_edit(self, dataset, adapter):
        """One-pass VAE + text encoder encoding of all dataset items.

        Images are processed in mini-batches so the VAE runs one forward pass per
        batch instead of per item. Image loading is parallelised with a small thread
        pool (I/O only — no model calls inside threads). Text encoding stays serial
        within each batch until the encode_instruction API is fully stable.

        Cached .pt files are written to disk so the training loop never re-runs
        encoding — it loads pre-computed tensors directly.
        """
        import hashlib
        from concurrent.futures import ThreadPoolExecutor

        import torch

        cache_base = self.cfg.data.get("cache_dir") or str(
            Path(self.cfg.logging.get("output_dir", "./output")) / "cache"
        )
        cache_dir = Path(cache_base) / "encoded"
        cache_dir.mkdir(parents=True, exist_ok=True)

        total = len(dataset._items) if hasattr(dataset, "_items") else 0
        if total == 0:
            return

        # Identify which items still need encoding
        uncached = []
        for item_path in dataset._items:
            cache_key = hashlib.md5(str(item_path).encode()).hexdigest()
            if not (cache_dir / f"{cache_key}.pt").exists():
                uncached.append(item_path)

        already_cached = total - len(uncached)
        if not uncached:
            logger.info(f"Pre-encoded latents: {total}/{total} items cached (up to date)")
            return

        batch_size = int(self.cfg.data.get("encode_batch_size", 4))
        num_io_workers = min(batch_size, 4)

        # Move VAE and text_encoder to GPU for encoding.
        # Use the local-rank GPU directly — do NOT follow adapter.model.device, since
        # setup_ddp() may not have run yet (pre-encoding happens before DDP wrapping)
        # and even if it has, the main model already fills most VRAM so we'd OOM.
        encode_device = "cpu"
        _vae_moved = False
        _te_moved = False
        if torch.cuda.is_available():
            try:
                import torch.distributed as _dist
                local_rank = _dist.get_rank() % torch.cuda.device_count() if _dist.is_initialized() else 0
                target_device = torch.device(f"cuda:{local_rank}")
                if adapter.vae is not None and next(adapter.vae.parameters()).device.type == "cpu":
                    adapter.vae = adapter.vae.to(target_device)
                    _vae_moved = True
                    logger.debug("Pre-encode: moved VAE to %s", target_device)
                if adapter.text_encoder is not None and next(adapter.text_encoder.parameters()).device.type == "cpu":
                    adapter.text_encoder = adapter.text_encoder.to(target_device)
                    _te_moved = True
                    logger.debug("Pre-encode: moved text_encoder to %s", target_device)
                encode_device = str(target_device)
            except Exception as e:
                logger.warning("Pre-encode: could not move encoders to GPU (%s) — encoding on CPU", e)

        logger.info(
            f"Pre-encoding {len(uncached)} items "
            f"(batch_size={batch_size}, io_workers={num_io_workers}, device={encode_device}) → {cache_dir}"
        )

        success = already_cached
        fail = 0

        def _load_item(path):
            return path, dataset._load_item(path)

        with ThreadPoolExecutor(max_workers=num_io_workers) as io_pool:
            for batch_start in range(0, len(uncached), batch_size):
                batch_paths = uncached[batch_start : batch_start + batch_size]

                # Load images from disk in parallel (pure I/O, no model calls)
                loaded = list(io_pool.map(_load_item, batch_paths))

                # --- Batch VAE encode (one forward pass for the whole batch) ---
                try:
                    pixel_batch = torch.stack([item["pixel_values"] for _, item in loaded])
                    # encode_image accepts (B,C,H,W) and handles device/dtype internally
                    latents_batch = adapter.encode_image(pixel_batch).cpu()
                    logger.debug(
                        "encode batch %d-%d: pixel_batch=%s → latents=%s",
                        batch_start, batch_start + len(batch_paths) - 1,
                        tuple(pixel_batch.shape), tuple(latents_batch.shape),
                    )
                except Exception as e:
                    logger.warning(
                        f"Batch VAE encode failed for items {batch_start}-"
                        f"{batch_start + len(batch_paths) - 1}: {e}",
                        exc_info=True,
                    )
                    fail += len(batch_paths)
                    continue

                # --- Text encode per item (serial within the batch) ---
                for i, (item_path, item) in enumerate(loaded):
                    cache_key = hashlib.md5(str(item_path).encode()).hexdigest()
                    cache_file = cache_dir / f"{cache_key}.pt"
                    try:
                        caption = item.get("caption", "")
                        pixel_single = pixel_batch[i : i + 1]
                        enc_hs, enc_mask = adapter.encode_instruction(caption, pixel_single)
                        enc_hs = enc_hs.squeeze(0).cpu()
                        cache_item = {
                            "target_latents": latents_batch[i],
                            "source_latents": latents_batch[i],   # identity conditioning: same image as source and target
                            "encoder_hidden_states": enc_hs,
                        }
                        if enc_mask is not None:
                            cache_item["encoder_attention_mask"] = enc_mask.squeeze(0).cpu()
                        torch.save(cache_item, cache_file)
                        success += 1
                    except Exception as e:
                        logger.warning(f"Failed to encode {item_path}: {e}", exc_info=True)
                        fail += 1

        # Move encoders back to CPU — training only needs the transformer on GPU.
        # The pre-encoded latents are loaded from disk during training, so these
        # components are not needed on GPU again.
        if _vae_moved and adapter.vae is not None:
            adapter.vae = adapter.vae.to("cpu")
            logger.debug("Pre-encode: moved VAE back to cpu")
        if _te_moved and adapter.text_encoder is not None:
            adapter.text_encoder = adapter.text_encoder.to("cpu")
            logger.debug("Pre-encode: moved text_encoder back to cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if fail:
            logger.warning(f"Pre-encoding: {success}/{total} succeeded, {fail} failed → {cache_dir}")
        else:
            logger.info(f"Pre-encoded {success}/{total} items → {cache_dir}")

    def _build_dataset(
        self,
        model_type: str,
        data_dir: str,
        target_size: tuple,
        repeats: int,
        cfg_data: DictConfig,
    ) -> Dataset:
        cache_dir = cfg_data.get("cache_dir") or (
            str(Path(self.cfg.logging.get("output_dir", "./output")) / "cache")
            if cfg_data.get("cache_latents") else None
        )

        if model_type in ("wan22", "wan21", "wan_i2v", "ltx2"):
            # Try video first, fall back to image
            try:
                return VideoDataset(
                    data_dir=data_dir,
                    num_frames=int(cfg_data.get("video_max_frames", 16)),
                    frame_stride=int(cfg_data.get("video_frame_stride", 1)),
                    target_size=target_size,
                    cache_dir=cache_dir,
                    repeats=repeats,
                )
            except RuntimeError:
                logger.info("No video files found; falling back to image dataset")
                return ImageDataset(
                    data_dir=data_dir,
                    target_size=target_size,
                    cache_dir=cache_dir,
                    repeats=repeats,
                )
        else:
            return ImageDataset(
                data_dir=data_dir,
                target_size=target_size,
                cache_dir=cache_dir,
                repeats=repeats,
            )

    def _build_dataloader(
        self, cfg_data: DictConfig, cfg_training: DictConfig, adapter=None
    ) -> DataLoader:
        batch_value = cfg_training.get("batch_size", "auto")

        if batch_value == "auto":
            batch_size = 1  # or your computed value
        else:
            batch_size = int(batch_value)
        # qwen/qwen_edit: force num_workers=0 so collation runs in the rank's main process.
        # Forked DataLoader workers cannot re-initialize CUDA (Linux fork limitation),
        # which breaks any fallback encode path. This only affects I/O parallelism,
        # not how many GPUs are used for training.
        if self._model_type in ("qwen_edit", "qwen"):
            num_workers = 0
        else:
            num_workers = int(cfg_data.get("num_workers", 4))
        prefetch_factor = int(cfg_data.get("prefetch_factor", 2))
        pin_memory = bool(cfg_data.get("pin_memory", True))
        persistent = bool(cfg_data.get("persistent_workers", True)) and num_workers > 0

        use_bucketing = cfg_data.get("bucketing", True)

        if use_bucketing and hasattr(self._dataset, "_items"):
            try:
                dl = self._build_bucketed_dataloader(
                    batch_size, num_workers, prefetch_factor, pin_memory, persistent,
                    adapter=adapter,
                )
                logger.info("Using bucketed DataLoader")
                return dl
            except Exception as e:
                logger.warning(f"Bucketed DataLoader failed: {e}. Using standard.")

        collate_fn = adapter.get_collate_fn() if adapter is not None else None

        return DataLoader(
            self._dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def _build_bucketed_dataloader(
        self,
        batch_size: int,
        num_workers: int,
        prefetch_factor: int,
        pin_memory: bool,
        persistent: bool,
        adapter=None,
    ) -> DataLoader:
        # Collect sizes
        sizes = []
        for item_path in self._dataset._items[:1000]:  # Sample first 1000 for speed
            try:
                from PIL import Image
                with Image.open(item_path) as img:
                    sizes.append((img.height, img.width))
            except Exception:
                sizes.append((512, 512))

        bucketer = ResolutionBucketer(
            max_resolution=int(self.cfg.data.get("max_resolution", 1024)),
            min_resolution=int(self.cfg.data.get("min_resolution", 256)),
            step=int(self.cfg.data.get("resolution_steps", 64)),
            no_upscale=bool(self.cfg.data.get("bucket_no_upscale", True)),
        )
        bucket_map = bucketer.assign_buckets(sizes)
        sampler = BucketedBatchSampler(bucket_map, batch_size=batch_size, shuffle=True)

        collate_fn = adapter.get_collate_fn() if adapter is not None else None

        return DataLoader(
            self._dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent,
            collate_fn=collate_fn,
        )
