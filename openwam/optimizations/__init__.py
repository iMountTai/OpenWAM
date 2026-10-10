"""Opt-in training optimizations; absent settings preserve the original path."""

import logging
import os

logger = logging.getLogger(__name__)

SWITCHES = (
    "OPENWAM_OPT_VAE_CHANNELS_LAST",
    "OPENWAM_OPT_CONSTANT_CACHE",
    "OPENWAM_OPT_TEXT_CACHE",
    "OPENWAM_OPT_FA2_PADDING",
    "OPENWAM_OPT_MOT_SPLIT_ATTN",
    "OPENWAM_OPT_POINTWISE_COMPILE",
    "OPENWAM_OPT_ROPE_REAL",
    "OPENWAM_OPT_ZERO_OVERLAP",
    "OPENWAM_OPT_LIGHTOP_NORM",
    "OPENWAM_OPT_VAE_POINTWISE_COMPILE",
    "OPENWAM_OPT_GLOBAL_COMPILE",
    "OPENWAM_OPT_PARTIAL_CHECKPOINT",
    "OPENWAM_OPT_UINT8_PREPROCESS",
    "OPENWAM_OPT_T5_PREFIX",
    "OPENWAM_OPT_TEXT_TRIM",
    "OPENWAM_OPT_LINEAR_BIAS2D",
    "OPENWAM_OPT_SAC_FA",
    "OPENWAM_OPT_SAC_FFN_FULL",
    "OPENWAM_OPT_SAC_GEMM",
    "OPENWAM_OPT_VAE_LAYOUT_REPAIR",
    "OPENWAM_OPT_ZERO_REDUCE_SCATTER",
    "OPENWAM_OPT_ONLINE_ENCODE",
)


def enabled(name: str, default: bool = False) -> bool:
    if name == "OPENWAM_OPT_PARTIAL_CHECKPOINT":
        from openwam.optimizations.runtime import checkpoint_skip_layers

        return checkpoint_skip_layers() > 0
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
    from openwam.optimizations.runtime import checkpoint_skip_layers
    from openwam.optimizations.sac import validate

    validate()
    if flags["OPENWAM_OPT_VAE_LAYOUT_REPAIR"]:
        from openwam.optimizations.vae_layout import validate as validate_layout

        validate_layout()
    if flags["OPENWAM_OPT_GLOBAL_COMPILE"]:
        from openwam.optimizations.global_compile import compile_scope

        compile_scope()
    if flags["OPENWAM_OPT_LIGHTOP_NORM"]:
        import torch

        if getattr(torch.version, "hip", None):
            from openwam.optimizations.norms import lightop_ops

            lightop_ops()
    if flags["OPENWAM_OPT_VAE_CHANNELS_LAST"] or flags["OPENWAM_OPT_VAE_LAYOUT_REPAIR"]:
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
        os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NDHWC", "1")
    if os.environ.get("RANK", "0") == "0":
        settings = {name: int(value) for name, value in flags.items()}
        settings["OPENWAM_OPT_PARTIAL_CHECKPOINT"] = checkpoint_skip_layers()
        logger.info("[optimizations] switches=%s", settings)


def configure_zero(config: dict) -> None:
    if enabled("OPENWAM_OPT_ZERO_REDUCE_SCATTER") or enabled("OPENWAM_OPT_ONLINE_ENCODE"):
        zero = config["zero_optimization"]
        if zero.get("stage") != 2:
            raise ValueError("ReduceScatter and online encoding currently require ZeRO-2")
        if enabled("OPENWAM_OPT_ZERO_REDUCE_SCATTER"):
            zero.update(reduce_scatter=True, contiguous_gradients=True, use_multi_rank_bucket_allreduce=True,
                        overlap_comm=True)
    if enabled("OPENWAM_OPT_ZERO_OVERLAP"):
        zero = config["zero_optimization"]
        zero.update(overlap_comm=True, contiguous_gradients=True)
        # Retain the measured ZeRO-2 default and explicit caller configurations.
        # Bucket size is in elements, not bytes.
        if zero.get("stage") == 2:
            zero.setdefault("reduce_bucket_size", 500_000_000)
        logger.info(
            "[optimizations] ZeRO overlap_comm=True, contiguous_gradients=True, reduce_bucket_size=%s",
            zero.get("reduce_bucket_size", "default"),
        )


def prepare_model(architecture) -> None:
    """Prepare VAE layout and lazy compilation before DeepSpeed."""
    from openwam.optimizations.sac import prepare

    prepare(architecture)
    if enabled("OPENWAM_OPT_PARTIAL_CHECKPOINT"):
        if type(architecture).__name__ != "DualSystemSelfAttnArchitecture":
            raise ValueError("OPENWAM_OPT_PARTIAL_CHECKPOINT currently requires dual_system/joint_self_attn")
    layout = enabled("OPENWAM_OPT_VAE_CHANNELS_LAST") or enabled("OPENWAM_OPT_VAE_LAYOUT_REPAIR")
    compile_vae = False
    if enabled("OPENWAM_OPT_GLOBAL_COMPILE"):
        from openwam.optimizations.global_compile import compile_scope

        compile_vae = "vae" in compile_scope()
    if not layout and not compile_vae:
        return
    import torch

    backbone = getattr(architecture, "video_backbone", None)
    vae = getattr(backbone, "vae", None)
    if vae is None:
        logger.warning("[optimizations] VAE preparation skipped: no native Wan VAE")
        return
    if layout:
        torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
        torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
        logger.info("[optimizations] native VAE Conv2d/Conv3d weights use channels-last")
    if compile_vae:
        from openwam.optimizations.global_compile import compile_vae_encoder

        compile_vae_encoder(vae)


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
