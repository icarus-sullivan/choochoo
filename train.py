#!/usr/bin/env python3
"""choochoo training CLI.

Usage:
    python train.py --config configs/wan22_lora.yaml
    python train.py --config configs/wan22_lora.yaml --resume ./output/checkpoint-500

Multi-GPU (torchrun):
    torchrun --nproc_per_node=4 train.py --config configs/wan22_lora.yaml
    torchrun --nproc_per_node=8 --nnodes=2 train.py --config configs/wan22_lora.yaml
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import torch


def _configure_logging(rank: int = 0, log_level: str = "INFO") -> None:
    level = getattr(logging, log_level.upper(), logging.INFO)
    if rank == 0:
        logging.basicConfig(
            level=level,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
            handlers=[logging.StreamHandler(sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.WARNING)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="choochoo: High-performance LoRA training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML training config"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to checkpoint directory to resume from"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Override data.data_dir from config"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Override logging.output_dir from config"
    )
    parser.add_argument(
        "--pretrained-path", type=str, default=None,
        help="Override model.pretrained_path from config"
    )
    parser.add_argument(
        "--max-steps", type=int, default=None,
        help="Override training.max_steps"
    )
    parser.add_argument(
        "--no-auto-tune", action="store_true",
        help="Disable AutoTuner (use config values directly)"
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    parser.add_argument(
        "--merge-lora", type=str, default=None,
        help="Merge LoRA from checkpoint into base model and exit"
    )
    parser.add_argument(
        "--export-lora", type=str, default=None,
        help="Export LoRA from checkpoint to safetensors and exit"
    )
    return parser.parse_args()


def apply_cli_overrides(cfg, args) -> None:
    """Apply command-line argument overrides to config."""
    from omegaconf import OmegaConf

    if args.data_dir:
        OmegaConf.update(cfg, "data.data_dir", args.data_dir)
    if args.output_dir:
        OmegaConf.update(cfg, "logging.output_dir", args.output_dir)
    if args.pretrained_path:
        OmegaConf.update(cfg, "model.pretrained_path", args.pretrained_path)
    if args.max_steps is not None:
        OmegaConf.update(cfg, "training.max_steps", args.max_steps)
    if args.no_auto_tune:
        OmegaConf.update(cfg, "training.auto_tune", False)


def _maybe_relaunch_distributed(config_path: str) -> None:
    """Re-exec under torchrun if multiple GPUs are present and we weren't launched by it.

    torchrun sets RANK before spawning each worker process. If RANK is absent we are
    the original `python train.py` invocation — re-exec via `python -m torch.distributed.run`
    so every GPU gets its own process with correct RANK/WORLD_SIZE env vars.

    Skipped when the config specifies a strategy that doesn't benefit from multi-process
    launch (e.g. qwen_edit defaults to single-GPU due to DDP init issues with 20B models).

    Opt-out: set CHOOCHOO_NO_TORCHRUN=1 to stay single-GPU (useful for debugging).
    """
    if os.environ.get("RANK") is not None:
        return  # Already running under torchrun — nothing to do
    if os.environ.get("CHOOCHOO_NO_TORCHRUN"):
        return  # User explicitly opted out
    if not torch.cuda.is_available():
        return
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        return

    # Peek at the config to check if strategy is forced to single — avoid
    # launching N processes only to have them all run single-GPU.
    try:
        import yaml
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f) or {}
        explicit_strategy = (user_cfg.get("distributed") or {}).get("strategy", "auto")
        if explicit_strategy == "single":
            return
    except Exception:
        pass  # If we can't read the config, proceed with relaunch

    # Use `python -m torch.distributed.run` (equivalent to torchrun) so this works in
    # any venv where torch is installed, even if torchrun isn't on PATH.
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={num_gpus}",
        "--standalone",   # single-node; handles master addr/port automatically
        "--",
        *sys.argv,        # pass through train.py + all original arguments unchanged
    ]
    logging.getLogger(__name__).info(
        "Multi-GPU detected (%d GPUs) — relaunching via torchrun", num_gpus
    )
    os.execvp(sys.executable, cmd)
    # os.execvp replaces this process in-place; nothing below here runs


def main() -> None:
    args = parse_args()

    # Reduce allocator fragmentation — especially important for models with large
    # activation tensors (20B+) where small allocation failures trigger unnecessary
    # cache clears. Set before any CUDA tensors are created.
    if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # TF32 gives ~20-30% faster matmuls on Ampere+ with no meaningful accuracy loss
    # for LoRA training. No-op for bf16 compute paths; affects float32 accumulation
    # (optimizer state, loss scalar). Safe on all GPU generations including Blackwell.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # NCCL defaults assume tight GPU coupling (NVLink / shared PCIe switch).
    # On SYS-topology nodes (GPUs on separate NUMA nodes, no NVLink) the default
    # P2P and GDR paths hang or perform poorly. Disable them so NCCL falls back
    # to a path that works across all hardware topologies.
    # Users can override any of these by setting them in the environment beforehand.
    if "NCCL_P2P_DISABLE" not in os.environ:
        os.environ["NCCL_P2P_DISABLE"] = "1"
    if "NCCL_NET_GDR_DISABLE" not in os.environ:
        os.environ["NCCL_NET_GDR_DISABLE"] = "1"

    # Auto-relaunch under torchrun when multiple GPUs are available.
    # Must run before init_distributed() so RANK/WORLD_SIZE are set correctly.
    _maybe_relaunch_distributed(args.config)

    # Initialize distributed (works for single GPU too)
    from choochoo.distributed.setup import init_distributed, cleanup_distributed
    dist_setup = init_distributed()

    _configure_logging(dist_setup.rank, args.log_level)
    logger = logging.getLogger(__name__)

    # Load config
    from choochoo.config import load_config
    cfg = load_config(args.config)
    apply_cli_overrides(cfg, args)

    if dist_setup.rank == 0:
        logger.info(f"choochoo LoRA trainer")
        logger.info(f"Config: {args.config}")
        logger.info(f"Model: {cfg.model.type} @ {cfg.model.pretrained_path}")

    # Set random seed
    seed = int(cfg.training.seed) + dist_setup.rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Auto-tune hardware — must run BEFORE load_model so model.dtype is resolved
    # before the pipeline is instantiated (otherwise prepare_dtype sees "auto").
    from choochoo.autotuner import AutoTuner
    tuner = AutoTuner()

    if cfg.training.auto_tune:
        from choochoo.config.schema import resolve_auto_values
        hw_info = tuner.detect_hardware()
        cfg = resolve_auto_values(cfg, hw_info)

    # Build model adapter
    from choochoo.models import build_adapter
    adapter = build_adapter(cfg)
    model = adapter.load_model()

    # Inject LoRA
    from choochoo.lora import LoRAInjector, LoRAExporter
    injector = LoRAInjector(
        rank=int(cfg.lora.rank),
        alpha=float(cfg.lora.alpha),
        dropout=float(cfg.lora.dropout),
        dual=bool(cfg.lora.get("dual_lora", False)),
    )
    adapter.inject_lora(injector)

    # Handle export/merge modes
    if args.merge_lora or args.export_lora:
        exporter = LoRAExporter(injector)
        ckpt_path = args.merge_lora or args.export_lora

        if dist_setup.rank == 0:
            logger.info(f"Loading LoRA from {ckpt_path}")
            exporter.load(
                Path(ckpt_path) / "lora_weights.safetensors"
                if Path(ckpt_path).is_dir()
                else ckpt_path
            )

        if args.merge_lora:
            merged_model = exporter.merge_into_base(model)
            out_path = Path(args.merge_lora).parent / "merged_model.safetensors"
            from safetensors.torch import save_file
            save_file(
                {k: v.float().contiguous() for k, v in merged_model.state_dict().items()},
                str(out_path),
            )
            logger.info(f"Merged model saved to {out_path}")

        if args.export_lora:
            out_path = Path(args.export_lora) / "lora_export.safetensors"
            exporter.save(out_path)
            logger.info(f"LoRA exported to {out_path}")

        cleanup_distributed()
        return

    # Apply gradient checkpointing
    if cfg.performance.gradient_checkpointing:
        adapter.enable_gradient_checkpointing(
            selective=bool(cfg.performance.get("selective_checkpointing", True))
        )
        logger.info("Gradient checkpointing: enabled")

    # Apply flash attention
    if cfg.performance.flash_attention:
        adapter.enable_flash_attention()

    # Pre-encode latents BEFORE moving the main model to GPU.
    # At this point the transformer is still on CPU, so each rank's GPU is empty
    # and can hold VAE + text_encoder without competing with the main model.
    # Rank 0 does the actual encoding; other ranks wait, then all build the dataloader
    # from the shared cache (fast — no GPU work on ranks 1+).
    if dist_setup.rank == 0:
        logger.info("Building data pipeline (pre-encoding latents if not cached)...")
        from choochoo.data.pipeline import DataPipeline
        pipeline = DataPipeline(cfg)
        dataloader = pipeline.setup(model_type=cfg.model.type, adapter=adapter)
        logger.info("Data pipeline ready")

    import torch.distributed as _dist_mod
    if _dist_mod.is_initialized():
        # Use gloo group — NCCL has a 10-min watchdog timeout that fires if rank 0
        # is still encoding while other ranks wait. Gloo has no timeout.
        _dist_mod.barrier(group=dist_setup.cpu_group)

    if dist_setup.rank != 0:
        from choochoo.data.pipeline import DataPipeline
        pipeline = DataPipeline(cfg)
        dataloader = pipeline.setup(model_type=cfg.model.type, adapter=adapter)

    # After pre-encoding, VAE and text_encoder are on CPU (moved back by pipeline).
    # They stay in CPU memory for use by the training-time sampler.

    # Wrap model for distributed training
    # Note: model.to(device) transfers ~40 GB for a 20B model — this can take
    # several minutes over PCIe. The log line below confirms we are in progress.
    strategy = cfg.distributed.strategy
    logger.info(
        "Moving model to %s (strategy=%s, world_size=%d) — may take several minutes for large models",
        dist_setup.device, strategy, dist_setup.world_size,
    )
    if strategy == "fsdp" and dist_setup.world_size > 1:
        from choochoo.distributed.fsdp import setup_fsdp
        model = setup_fsdp(model, cfg, device_id=dist_setup.local_rank)
        adapter.model = model
    elif strategy == "ddp" and dist_setup.world_size > 1:
        from choochoo.distributed.ddp import setup_ddp
        model = setup_ddp(model, cfg, device_id=dist_setup.local_rank)
        adapter.model = model
    else:
        model = model.to(dist_setup.device)
        adapter.model = model
    logger.info("Model on device: %s", dist_setup.device)

    # Compile if beneficial — skip for models with trust_remote_code custom ops
    # that don't trace cleanly through torch.compile's dynamo frontend.
    # qwen/qwen_edit produce variable-length sequences from bucketed data, causing
    # CUDAGraph thrashing under reduce-overhead mode (9+ distinct shapes → full
    # recompile per step, far slower than eager).
    _no_compile_types = {"qwen_edit", "qwen"}
    if cfg.performance.compile and cfg.model.type not in _no_compile_types:
        if dist_setup.rank == 0:
            logger.info("torch.compile: enabled (mode=reduce-overhead)")
        try:
            adapter.model = torch.compile(
                adapter.model,
                mode=cfg.performance.get("compile_mode", "reduce-overhead"),
                fullgraph=False,
            )
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    elif cfg.model.type in _no_compile_types:
        logger.info("torch.compile: skipped for %s (variable-length sequences incompatible with CUDAGraphs)", cfg.model.type)

    # VAE and text_encoder are on CPU from pre-encoding (moved back by pipeline).
    # They stay in CPU memory for use by the training-time sampler.
    # The dataloader was already built above (before setup_ddp).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Batch size from auto-tune (if resolved)
    batch_size = cfg.training.get("batch_size", 1)
    if batch_size == "auto" or batch_size is None:
        batch_size = 1
    logger.info(f"Batch size: {batch_size}")

    # Build trainer
    from choochoo.training.loop import Trainer
    trainer = Trainer(
        cfg=cfg,
        model_adapter=adapter,
        dataloader=dataloader,
        rank=dist_setup.rank,
        world_size=dist_setup.world_size,
    )
    trainer.setup()

    # Auto-detect resume from output dir if not explicitly specified
    if args.resume is None:
        from choochoo.checkpointing import CheckpointManager
        _latest = CheckpointManager(cfg.logging.output_dir, cfg).latest_checkpoint()
        if _latest is not None:
            if dist_setup.rank == 0:
                logger.info("Auto-resuming from latest checkpoint: %s", _latest)
            args.resume = str(_latest)

    # Resume if requested
    if args.resume:
        trainer.resume(args.resume)

    # Run training
    if dist_setup.rank == 0:
        from omegaconf import OmegaConf
        logger.debug("Resolved config:\n%s", OmegaConf.to_yaml(cfg))
        logger.info("=" * 60)
        logger.info("Starting training")
        logger.info("=" * 60)

    trainer.train()

    cleanup_distributed()


if __name__ == "__main__":
    main()
