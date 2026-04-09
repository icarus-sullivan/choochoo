from .base import BaseModelAdapter
from .wan22 import WANAdapter, WAN22Adapter
from .wan22_dual import WANDualAdapter
from .wan_i2v import WANi2vAdapter
from .qwen_edit import QwenEditAdapter
from .qwen import QwenAdapter
from .ltx2 import LTX2Adapter

MODEL_REGISTRY = {
    "wan22":      WANAdapter,       # WAN 2.2 T2V (single checkpoint)
    "wan21":      WANAdapter,       # WAN 2.1 T2V (same adapter, different pretrained_path)
    "wan_i2v":    WANi2vAdapter,    # WAN I2V (image-to-video)
    "wan22_dual": WANDualAdapter,   # WAN 2.2 dual-model (high + low noise checkpoints)
    "qwen_edit":  QwenEditAdapter,  # Qwen-Image-Edit (source+target instruction editing)
    "qwen":       QwenAdapter,      # Qwen-Image (pure T2I, caption only)
    "ltx2":       LTX2Adapter,
}


def build_adapter(cfg) -> BaseModelAdapter:
    model_type = cfg.model.type
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[model_type](cfg)


__all__ = [
    "BaseModelAdapter",
    "WANAdapter", "WAN22Adapter", "WANDualAdapter", "WANi2vAdapter",
    "QwenEditAdapter", "QwenAdapter", "LTX2Adapter",
    "build_adapter",
]
