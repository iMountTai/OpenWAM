"""DCU-only runtime setup; AMD HIP and NVIDIA CUDA keep their native setup."""

import os
import re
from functools import lru_cache
from importlib import metadata
from pathlib import Path

from openwam.optimizations.config import settings


@lru_cache(maxsize=1)
def _installed_torch_build():
    """DTK wheels can leave torch.__version__ upstream while tagging metadata."""
    import torch

    try:
        distribution = metadata.distribution("torch")
    except metadata.PackageNotFoundError:
        return ""
    if Path(distribution.locate_file("torch/__init__.py")).resolve() != Path(torch.__file__).resolve():
        return ""
    return distribution.version.lower()


def is_dtk(torch_module=None):
    """Inspect build metadata without initializing CUDA/HIP or probing devices."""
    if torch_module is None:
        import torch as torch_module
        build = _installed_torch_build()
    else:
        build = ""
    version = str(torch_module.__version__).lower()
    return bool(getattr(torch_module.version, "hip", None)) and (
        bool(getattr(torch_module.version, "dtk", None))
        or re.search(r"(?:^|[.+_-])dtk(?:[0-9.+_-]|$)", version) is not None
        or re.search(r"(?:^|[.+_-])dtk(?:[0-9.+_-]|$)", build) is not None
    )


def enabled():
    return settings().dtk.enabled and is_dtk()


def configure():
    config = settings().dtk
    if not config.enabled:
        return
    if not is_dtk():
        raise RuntimeError("optimizations.dtk.enabled=true requires a DTK PyTorch build; AMD HIP/CUDA are unsupported")
    if config.hw_queues is not None:
        from openwam.optimizations.hw_queues import configure as configure_queues

        configure_queues()
    for active, name in (
        (settings().vae_channels_last, "PYTORCH_MIOPEN_SUGGEST_NHWC"),
        (settings().vae_channels_last, "PYTORCH_MIOPEN_SUGGEST_NDHWC"),
        (config.blas_prefer_hipblaslt, "TORCH_BLAS_PREFER_HIPBLASLT"),
        (config.allow_tf32, "HIPBLASLT_ALLOW_TF32"),
        (config.fine_grain_pcie, "HSA_FORCE_FINE_GRAIN_PCIE"),
    ):
        if active:
            os.environ[name] = "1"
    if config.lightop_norm:
        from openwam.optimizations.norms import lightop_ops

        lightop_ops()
