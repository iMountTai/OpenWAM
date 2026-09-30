"""Opt-in training optimizations; all switches default to the original path."""

import logging
import os

logger = logging.getLogger(__name__)

SWITCHES = (
    "OPENWAM_OPT_VAE_CHANNELS_LAST",
    "OPENWAM_OPT_CONSTANT_CACHE",
    "OPENWAM_OPT_TEXT_MASK",
    "OPENWAM_OPT_TEXT_CACHE",
    "OPENWAM_OPT_VIDEO_PREPROCESS",
    "OPENWAM_OPT_FA2_PADDING",
    "OPENWAM_OPT_MOT_SPLIT_ATTN",
    "OPENWAM_OPT_POINTWISE_COMPILE",
    "OPENWAM_OPT_ROPE_FP32",
    "OPENWAM_OPT_ZERO_OVERLAP",
    "OPENWAM_OPT_TF32",
    "OPENWAM_OPT_VAE_HIPDNN",
    "OPENWAM_OPT_VAE_CONCAT_CACHE_WEIGHT",
    "OPENWAM_OPT_VAE_CONCAT_SINGLETON_RESTRIDE",
)


def enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be 0/1, false/true, no/yes or off/on; got {value!r}")


def configure_backends() -> None:
    """Called by scripts/train.py before model construction and device setup."""
    flags = {name: enabled(name) for name in SWITCHES}
    if flags["OPENWAM_OPT_VAE_HIPDNN"] and not flags["OPENWAM_OPT_VAE_CHANNELS_LAST"]:
        raise ValueError("OPENWAM_OPT_VAE_HIPDNN=1 requires OPENWAM_OPT_VAE_CHANNELS_LAST=1")
    if flags["OPENWAM_OPT_VAE_CHANNELS_LAST"]:
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NDHWC", "1")
    if flags["OPENWAM_OPT_TF32"]:
        import torch

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    if flags["OPENWAM_OPT_VAE_HIPDNN"]:
        import torch

        if getattr(torch.version, "hip", None):
            from openwam.optimizations.vae_fused._hipdnn import require_hipdnn

            require_hipdnn()
        else:
            logger.warning("[optimizations] VAE hipDNN inactive: this is not a HIP PyTorch build")
    if os.environ.get("RANK", "0") == "0":
        logger.info("[optimizations] switches=%s", {name: int(value) for name, value in flags.items()})


def configure_zero(config: dict) -> None:
    if enabled("OPENWAM_OPT_ZERO_OVERLAP"):
        config["zero_optimization"].update(overlap_comm=True, contiguous_gradients=True)
        logger.info("[optimizations] ZeRO overlap_comm=True, contiguous_gradients=True")


def prepare_model(architecture) -> None:
    """Convert frozen VAE weights before DeepSpeed partitions parameters."""
    if not enabled("OPENWAM_OPT_VAE_CHANNELS_LAST"):
        return
    import torch

    backbone = getattr(architecture, "video_backbone", None)
    vae = getattr(backbone, "vae", None)
    if vae is None:
        logger.warning("[optimizations] VAE channels-last skipped: no native Wan VAE")
        return
    torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
    torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
    logger.info("[optimizations] native VAE Conv2d/Conv3d weights use channels-last")


def prepare_runtime_constants(architecture) -> None:
    """Stage small, immutable training constants after Accelerate device placement."""
    if not enabled("OPENWAM_OPT_CONSTANT_CACHE"):
        return
    backbone = getattr(architecture, "video_backbone", None)
    if backbone is None:
        return
    device = architecture.device
    for stream in (backbone, getattr(architecture, "action_backbone", None)):
        scheduler = getattr(stream, "scheduler", None)
        for name in ("timesteps", "sigmas", "linear_timesteps_weights"):
            value = getattr(scheduler, name, None)
            if value is not None:
                setattr(scheduler, name, value.to(device=device))
    vae = getattr(backbone, "vae", None)
    if vae is not None and hasattr(vae, "prepare_scale_cache"):
        vae.prepare_scale_cache()
    logger.info("[optimizations] scheduler tables and VAE scales staged on %s", device)


def stage_training_indices(indices, device):
    """Preserve CPU sampling but avoid a synchronizing transfer behind VAE work."""
    import torch

    target = torch.device(device)
    if indices.device.type == "cpu" and target.type == "cuda":
        return indices.pin_memory().to(device=target, non_blocking=True)
    return indices.to(device=target)
