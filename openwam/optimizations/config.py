"""Typed Hydra settings. Optimization environment variables are not read."""

from dataclasses import dataclass, field

from omegaconf import OmegaConf


@dataclass(frozen=True)
class CompileSettings:
    mot: bool = False
    vae: bool = False
    pointwise: bool = False
    vae_pointwise: bool = False
    attention_mask: bool = False
    threads: int | None = None
    coordinate_descent_tuning: bool = False


@dataclass(frozen=True)
class SACSettings:
    attention: bool = False
    ffn_full: bool = False
    gemm: bool = False
    layers: int | None = None
    extra_ops: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DTKSettings:
    # Python equivalent of USE_DTK: explicitly opt in AND verify a DTK build.
    enabled: bool = False
    hw_queues: int | None = None
    lightop_norm: bool = False
    linear_bias2d: bool = False
    blas_prefer_hipblaslt: bool = False
    allow_tf32: bool = False
    fine_grain_pcie: bool = False


@dataclass(frozen=True)
class ProfilingSettings:
    ranks: str = "off"  # off / rank0 / all
    light: bool = False
    active_steps: int = 1


@dataclass(frozen=True)
class DiagnosticSettings:
    checkpoint_stats: bool = False
    trace_gc: bool = False


@dataclass(frozen=True)
class OptimizationSettings:
    vae_channels_last: bool = False
    constant_cache: bool = False
    text_cache: bool = False
    offline_text_cache: bool = False
    text_cache_size: int = 32
    text_cache_dir: str | None = None
    fa2_padding: bool = False
    padding_plan_cpu: bool = True
    mot_split_attn: bool = False
    rope_real: bool = False
    zero_overlap: bool = False
    checkpoint_skip_layers: int = 0
    uint8_preprocess: bool = True
    t5_prefix: bool = False
    text_trim: bool = False
    zero_reduce_scatter: bool = False
    online_encode: bool = False
    deferred_gather: bool = False
    fused_adamw: bool = False
    host_async: bool = False
    vae_latent_cache: bool = False
    vae_latent_cache_dir: str | None = None
    compile: CompileSettings = field(default_factory=CompileSettings)
    sac: SACSettings = field(default_factory=SACSettings)
    dtk: DTKSettings = field(default_factory=DTKSettings)
    profiling: ProfilingSettings = field(default_factory=ProfilingSettings)
    diagnostics: DiagnosticSettings = field(default_factory=DiagnosticSettings)


_settings = OptimizationSettings()


def settings() -> OptimizationSettings:
    return _settings


def option(path):
    value = _settings
    for name in path.split("."):
        value = getattr(value, name)
    return value


def configure(config=None) -> OptimizationSettings:
    """Validate the entire configuration before publishing it to model helpers."""
    global _settings
    schema = OmegaConf.structured(OptimizationSettings)

    def writable(node):
        OmegaConf.set_readonly(node, False)
        children = node.values() if OmegaConf.is_dict(node) else node
        for child in children:
            if OmegaConf.is_config(child):
                writable(child)

    writable(schema)
    candidate = OmegaConf.to_object(OmegaConf.merge(schema, {} if config is None else config))
    for path in ("checkpoint_skip_layers", "sac.layers", "compile.threads", "dtk.hw_queues", "text_cache_size",
                 "profiling.active_steps"):
        value = candidate
        for name in path.split("."):
            value = getattr(value, name)
        minimum = 2 if path == "dtk.hw_queues" else (
            1 if path in {"compile.threads", "text_cache_size", "profiling.active_steps"} else 0)
        if value is not None and value < minimum:
            raise ValueError(f"optimizations.{path} must be >= {minimum}, got {value}")
    if candidate.profiling.ranks not in {"off", "rank0", "all"}:
        raise ValueError("optimizations.profiling.ranks must be off, rank0 or all")
    for path in ("vae_latent_cache_dir", "text_cache_dir"):
        value = getattr(candidate, path)
        if value is not None and not value.strip():
            raise ValueError(f"optimizations.{path} must be a non-empty path or null")
    if candidate.vae_latent_cache and not candidate.vae_latent_cache_dir:
        raise ValueError("optimizations.vae_latent_cache requires optimizations.vae_latent_cache_dir")
    if candidate.offline_text_cache and not (candidate.text_cache_dir or candidate.vae_latent_cache_dir):
        raise ValueError("optimizations.offline_text_cache requires text_cache_dir or vae_latent_cache_dir")
    if candidate.compile.attention_mask and not candidate.fa2_padding:
        raise ValueError("optimizations.compile.attention_mask requires fa2_padding=true")
    dtk = candidate.dtk
    if not dtk.enabled and any((dtk.hw_queues is not None, dtk.lightop_norm, dtk.linear_bias2d, dtk.blas_prefer_hipblaslt,
                               dtk.allow_tf32, dtk.fine_grain_pcie)):
        raise ValueError("DCU-specific options require optimizations.dtk.enabled=true")
    _settings = candidate
    return candidate
