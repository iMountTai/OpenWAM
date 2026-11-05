"""Opt-in training optimizations; absent settings preserve the original path."""

import logging
import os

logger = logging.getLogger(__name__)

SWITCHES = (
    "OPENWAM_OPT_CONSTANT_CACHE",
    "OPENWAM_OPT_TEXT_CACHE",
    "OPENWAM_OPT_FA2_PADDING",
    "OPENWAM_OPT_FA2_MASK_FUSED",
    "OPENWAM_OPT_PADDING_PLAN_CPU",
    "OPENWAM_OPT_LIGHTOP_NORM",
    "OPENWAM_OPT_LINEAR_BIAS2D",
    "OPENWAM_OPT_SAC_FA",
    "OPENWAM_OPT_SAC_FFN_FULL",
    "OPENWAM_OPT_SAC_GEMM",
    "OPENWAM_SAC_LAYERS",
    "OPENWAM_SAC_EXTRA_OPS",
    "OPENWAM_OPT_ZERO_OVERLAP",
    "OPENWAM_OPT_ZERO_REDUCE_SCATTER",
    "OPENWAM_OPT_ZERO_SCALED_PACK",
    "OPENWAM_OPT_ZERO_NORM_OVERLAP",
    "OPENWAM_OPT_ONLINE_ENCODE",
    "OPENWAM_OPT_VAE_LATENT_CACHE",
    "OPENWAM_OPT_FUSED_ADAMW",
    "OPENWAM_OPT_HOST_ASYNC",
    "OPENWAM_OPT_DEFERRED_GATHER",
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
    from openwam.optimizations.hw_queues import configure as configure_hw_queues

    configure_hw_queues()  # Before anything can initialize the HIP runtime.
    flags = {name: enabled(name) for name in SWITCHES}
    from openwam.optimizations.sac import checkpoint_skip_layers, validate

    validate()
    if flags["OPENWAM_OPT_VAE_LATENT_CACHE"]:
        from openwam.optimizations.latent_cache import directory

        directory()
    if flags["OPENWAM_OPT_LIGHTOP_NORM"]:
        import torch

        if getattr(torch.version, "hip", None):
            from openwam.optimizations.norms import lightop_ops

            lightop_ops()
    if os.environ.get("OPENWAM_VAE_CL", "0") == "1":
        import torch

        if getattr(torch.version, "hip", None):
            os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NHWC", "1")
            os.environ.setdefault("PYTORCH_MIOPEN_SUGGEST_NDHWC", "1")
    if os.environ.get("RANK", "0") == "0":
        settings = {name: int(value) for name, value in flags.items()}
        settings["skip_layers"] = checkpoint_skip_layers()
        if "GPU_MAX_HW_QUEUES" in os.environ:
            settings["GPU_MAX_HW_QUEUES"] = int(os.environ["GPU_MAX_HW_QUEUES"])
        logger.info("[optimizations] switches=%s", settings)


def configure_zero(config: dict) -> None:
    if (enabled("OPENWAM_OPT_ZERO_SCALED_PACK") or enabled("OPENWAM_OPT_ZERO_NORM_OVERLAP")):
        if not enabled("OPENWAM_OPT_ZERO_REDUCE_SCATTER"):
            raise ValueError("ZeRO scaled pack/norm overlap require OPENWAM_OPT_ZERO_REDUCE_SCATTER=1")
    if enabled("OPENWAM_OPT_HOST_ASYNC"):
        from openwam.optimizations.host_async import configure_deepspeed

        configure_deepspeed(config)
    if enabled("OPENWAM_OPT_ZERO_REDUCE_SCATTER") or enabled("OPENWAM_OPT_ONLINE_ENCODE"):
        zero = config["zero_optimization"]
        if zero.get("stage") != 2:
            raise ValueError("ReduceScatter and online encoding currently require ZeRO-2")
        if enabled("OPENWAM_OPT_ZERO_REDUCE_SCATTER"):
            zero.update(
                reduce_scatter=True,
                contiguous_gradients=True,
                use_multi_rank_bucket_allreduce=True,
                overlap_comm=True,
            )
    if enabled("OPENWAM_OPT_DEFERRED_GATHER") and config["zero_optimization"].get("stage") != 2:
        raise ValueError("OPENWAM_OPT_DEFERRED_GATHER currently requires ZeRO-2")
    if enabled("OPENWAM_OPT_ZERO_OVERLAP"):
        zero = config["zero_optimization"]
        zero.update(overlap_comm=True, contiguous_gradients=True)
        # Keep room for backward activations; preserve explicit caller settings.
        # Bucket size is in elements, not bytes.
        if zero.get("stage") == 2:
            zero.setdefault("reduce_bucket_size", 100_000_000)
            import torch

            # Limit transient GEMM workspace before the first BLASLt call (KiB).
            # PyTorch gives CUBLASLT_WORKSPACE_SIZE priority on HIP as well.
            if (
                getattr(torch.version, "hip", None)
                and os.environ.get("TORCH_BLAS_PREFER_HIPBLASLT") == "1"
                and "CUBLASLT_WORKSPACE_SIZE" not in os.environ
                and "HIPBLASLT_WORKSPACE_SIZE" not in os.environ
            ):
                os.environ["HIPBLASLT_WORKSPACE_SIZE"] = "32768"
                logger.info("[optimizations] HIP BLASLt workspace default=32 MiB")
        logger.info(
            "[optimizations] ZeRO overlap_comm=True, contiguous_gradients=True, reduce_bucket_size=%s",
            zero.get("reduce_bucket_size", "default"),
        )


def prepare_model(architecture) -> None:
    """Prepare selective activation checkpointing before DeepSpeed."""
    from openwam.optimizations.sac import prepare

    prepare(architecture)


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
