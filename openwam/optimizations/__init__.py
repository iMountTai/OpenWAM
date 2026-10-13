"""Opt-in training optimizations; absent settings preserve the original path."""

import logging
import os

from openwam.optimizations.config import configure, option, settings

logger = logging.getLogger(__name__)

SWITCHES = (
    "vae_channels_last",
    "constant_cache",
    "text_cache",
    "fa2_padding",
    "fa2_mask_fused",
    "padding_plan_cpu",
    "mot_split_attn",
    "compile.pointwise",
    "rope_real",
    "zero_overlap",
    "dtk.lightop_norm",
    "compile.vae_pointwise",
    "compile",
    "checkpoint_skip_layers",
    "uint8_preprocess",
    "t5_prefix",
    "text_trim",
    "dtk.linear_bias2d",
    "sac.attention",
    "sac.ffn_full",
    "sac.gemm",
    "zero_reduce_scatter",
    "online_encode",
    "dtk.hw_queues",
)


def enabled(name: str) -> bool:
    if name == "checkpoint_skip_layers":
        return settings().checkpoint_skip_layers > 0
    if name == "dtk.hw_queues":
        return settings().dtk.enabled and settings().dtk.hw_queues is not None
    if name == "compile":
        config = settings().compile
        return any((config.mot, config.vae))
    value = option(name)
    if not isinstance(value, bool):
        raise TypeError(f"optimizations.{name} is not a boolean option")
    return value


def configure_backends(config=None) -> None:
    """Called by scripts/train.py before model construction and device setup."""
    configure(config)
    from openwam.optimizations.dtk import configure as configure_dtk

    configure_dtk()  # Before anything can initialize the HIP runtime.
    if settings().compile.threads is not None:
        os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = str(settings().compile.threads)
    flags = {name: enabled(name) for name in SWITCHES}
    if not any(flags.values()) and not settings().sac.extra_ops:
        return
    from openwam.optimizations.runtime import checkpoint_skip_layers
    from openwam.optimizations.sac import validate

    validate()
    if flags["vae_channels_last"]:
        from openwam.optimizations.vae_layout import validate as validate_layout

        validate_layout()
    if flags["compile"]:
        from openwam.optimizations.global_compile import compile_scope

        compile_scope()
    if os.environ.get("RANK", "0") == "0":
        effective = {name: int(value) for name, value in flags.items()}
        effective["checkpoint_skip_layers"] = checkpoint_skip_layers()
        if flags["dtk.hw_queues"]:
            effective["dtk.hw_queues"] = option("dtk.hw_queues")
        logger.info("[optimizations] settings=%s", effective)


def configure_zero(config: dict) -> None:
    if enabled("zero_reduce_scatter") or enabled("online_encode"):
        zero = config["zero_optimization"]
        if zero.get("stage") != 2:
            raise ValueError("ReduceScatter and online encoding currently require ZeRO-2")
        if enabled("zero_reduce_scatter"):
            zero.update(reduce_scatter=True, contiguous_gradients=True, use_multi_rank_bucket_allreduce=True,
                        overlap_comm=True)
    if enabled("zero_overlap"):
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
    if (enabled("compile.mot") or any(enabled(name) for name in ("sac.ffn_full", "sac.gemm"))
            or settings().sac.extra_ops or enabled("dtk.linear_bias2d")):
        from openwam.model.video_backbone.wan_backbone import WanBase

        backbone = getattr(architecture, "video_backbone", None)
        if isinstance(backbone, WanBase):
            from openwam.optimizations.wan import prepare as prepare_wan

            prepare_wan(backbone)
    if enabled("checkpoint_skip_layers"):
        if type(architecture).__name__ != "DualSystemSelfAttnArchitecture":
            raise ValueError("checkpoint_skip_layers currently requires dual_system/joint_self_attn")
    layout = enabled("vae_channels_last")
    compile_vae = False
    if enabled("compile"):
        from openwam.optimizations.global_compile import compile_scope

        compile_vae = "vae" in compile_scope()
    adapt_vae = any(enabled(name) for name in ("vae_channels_last", "compile.vae_pointwise", "constant_cache"))
    if not adapt_vae and not compile_vae:
        return
    import torch

    backbone = getattr(architecture, "video_backbone", None)
    vae = getattr(backbone, "vae", None)
    if vae is None:
        logger.warning("[optimizations] VAE preparation skipped: no native Wan VAE")
        return
    if adapt_vae:
        from openwam.optimizations.vae import prepare as prepare_vae

        prepare_vae(vae)
    if layout:
        torch.nn.utils.convert_conv3d_weight_memory_format(vae, torch.channels_last_3d)
        torch.nn.utils.convert_conv2d_weight_memory_format(vae, torch.channels_last)
        logger.info("[optimizations] native VAE Conv2d/Conv3d weights use channels-last")
    if compile_vae:
        from openwam.optimizations.global_compile import compile_vae_encoder

        compile_vae_encoder(vae)


def prepare_runtime_constants(architecture) -> None:
    """Stage small, immutable training constants after Accelerate device placement."""
    if not enabled("constant_cache"):
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


def create_trainer(cfg, *, accelerator=None, dataset=None):
    """One training-entry hook; keep the upstream trainer for the all-off route."""
    configure_backends(cfg.get("optimizations"))
    config = settings()
    active = (any(enabled(name) for name in SWITCHES) or bool(config.sac.extra_ops)
              or config.profiling.ranks != "off" or config.diagnostics.checkpoint_stats or config.diagnostics.trace_gc)
    if active:
        from openwam.optimizations.trainer import OptimizationTrainer as trainer
    else:
        from openwam.train.openwam_trainer import OpenWAMTrainer as trainer
    return trainer(cfg, accelerator=accelerator, dataset=dataset)


def mot_driver_type():
    """Select a configured extension without altering the upstream driver class."""
    config = settings()
    active = (config.compile.mot or config.mot_split_attn or config.checkpoint_skip_layers > 0
              or any((config.sac.attention, config.sac.ffn_full, config.sac.gemm, bool(config.sac.extra_ops)))
              or config.diagnostics.checkpoint_stats)
    if active:
        from openwam.optimizations.mot import OptimizationMoTDriver

        return OptimizationMoTDriver
    from openwam.model.architectures.dual_system.mot_driver import DualSystemMoTDriver

    return DualSystemMoTDriver


def stage_training_indices(indices, device):
    """Preserve CPU sampling but avoid a synchronizing transfer behind VAE work."""
    import torch

    target = torch.device(device)
    if indices.device.type == "cpu" and target.type == "cuda":
        return indices.pin_memory().to(device=target, non_blocking=True)
    return indices.to(device=target)
