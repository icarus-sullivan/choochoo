"""Default configuration values for choochoo training framework."""

DEFAULTS = {
    "name": None,   # REQUIRED — identifies this run; used in output filenames
    "model": {
        "type": "wan22",
        "pretrained_path": None,
        "dtype": "auto",
        # Noise regime for WAN adapters: "high" | "low" | "both" | "auto" (→ both)
        # Controls which portion of the timestep range is sampled during training.
        # Use "high" / "low" for separate WAN 2.2 dual-checkpoint fine-tuning runs.
        "noise_regime": "auto",
        # Dual-model config (wan22_dual only)
        "dual": {
            "high_noise_path": None,   # path to high-noise transformer checkpoint
            "low_noise_path":  None,   # path to low-noise transformer checkpoint
            "noise_boundary":  0.875,  # t2v: 0.875, i2v: 0.9
        },
    },
    "lora": {
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "target_modules": None,  # None → each adapter uses its own DEFAULT_LORA_TARGETS
        "dual_lora": False,
        "high_noise_weight": 1.0,
        "low_noise_weight": 1.0,
    },
    "training": {
        "auto_tune": True,
        "max_steps": 2000,
        "learning_rate": 1e-4,
        "lr_warmup_steps": 100,
        "lr_scheduler": "cosine",
        "batch_size": "auto",
        "gradient_accumulation_steps": "auto",
        "target_vram_utilization": 0.92,
        "mixed_precision": "auto",
        "seed": 42,
        "save_every": 500,
        "log_every": 10,
        "ema": False,
        "ema_decay": 0.9999,
        "auto_stop": False,
        "convergence_patience": 200,
        "convergence_threshold": 0.01,
        "keep_top_k_checkpoints": 3,
    },
    "distributed": {
        "strategy": "auto",
        "fsdp": {
            "auto_wrap": "wan_optimized",
            "min_params": 1e7,
            "sharding_strategy": "FULL_SHARD",
            "cpu_offload": False,
        },
        "ddp": {
            "find_unused_parameters": False,
            "gradient_as_bucket_view": True,
            "static_graph": True,
            "bucket_cap_mb": 200,  # size to hold all LoRA params in one all-reduce (~106M × 2B bf16 ≈ 200MB)
        },
    },
    "performance": {
        "compile": "auto",
        "compile_mode": "reduce-overhead",
        "flash_attention": "auto",
        "xformers": "auto",
        "gradient_checkpointing": "auto",
        "selective_checkpointing": True,
        "channels_last": True,
        "fused_optimizer": True,
        "vram_fragmentation_reduction": True,
    },
    "data": {
        "num_workers": "auto",
        "prefetch_factor": "auto",
        "pin_memory": True,
        "persistent_workers": True,
        "cache_latents": True,
        "cache_dir": None,
        "bucketing": True,
        "bucket_no_upscale": True,
        "max_resolution": 1024,
        "min_resolution": 256,
        "resolution_steps": 64,
        "video_max_frames": 64,
        "video_frame_stride": 1,
        "repeat_dataset": "auto",
        "encode_batch_size": 4,    # items per VAE + I/O batch during latent pre-encoding
    },
    "logging": {
        "backend": "tensorboard",
        "wandb_project": None,
        "wandb_run_name": None,
        "output_dir": "./output",
        "log_dir": "./logs",
        "profile_steps": 0,
        "profile_output": "./profiler",
    },
    "sample": {
        "enabled": True,        # set False to disable even if prompts are listed
        "sample_every": None,   # None → use training.save_every
        "steps": 20,            # inference denoising steps
        "width": 480,
        "height": 480,
        "num_frames": 42,       # video frames (ignored for image models)
        "cfg": 4.0,             # guidance_scale for inference
        "seed": 42,
        "prompts": [],          # empty → sampling disabled
    },
}
