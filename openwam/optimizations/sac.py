__all__ = [
    "FLAGS",
    "active",
    "active_ffn_boundary",
    "attention_boundary",
    "begin_step",
    "context_fn",
    "dispatcher",
    "extras_enabled",
    "ffn_boundary",
    "layer_range",
    "layers",
    "prepare",
    "projection_boundary",
    "report_step",
    "run_attention",
    "run_ffn",
    "run_norm",
    "run_pointwise",
    "run_projection",
    "run_video_projection",
    "validate",
    "checkpoint_layers",
    "checkpoint_skip_layers",
    "set_architecture",
    "set_gradient_checkpointing_layers",
]

"""Standard PyTorch SAC policies for complete FFNs, attention and other ops.

Each checkpoint owns its standard forward/recompute modes. Only explicit eager
operator regions enter those modes; unselected compiled regions keep their path.
No activation cache, backward implementation or parameter replacement lives here.

Policies apply to the trailing OPENWAM_SAC_LAYERS of the checkpointed layers.
Layers without checkpointing keep their activations through autograd, so SAC
never caches them a second time.
"""

import json
import logging
import os
import re
import weakref
from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Optional

import torch
from torch.utils import checkpoint as torch_checkpoint
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts

from openwam.optimizations import enabled


FLAGS = ("OPENWAM_OPT_SAC_FA", "OPENWAM_OPT_SAC_FFN_FULL", "OPENWAM_OPT_SAC_GEMM")
logger = logging.getLogger(__name__)
_session = ContextVar("openwam_sac_session", default=None)
_regions = ContextVar("openwam_sac_regions", default=())
_inside_mode = ContextVar("openwam_inside_sac_mode", default=False)
_native_mot = ContextVar("openwam_sac_native_mot_expression", default=False)
_counts = Counter()

_architecture = None
_gradient_checkpointing_layers = None


def set_architecture(arch) -> None:
    global _architecture
    _architecture = weakref.ref(arch) if arch is not None else None


def set_gradient_checkpointing_layers(n: Optional[int]) -> None:
    global _gradient_checkpointing_layers
    _gradient_checkpointing_layers = n


def _integer(name, default, minimum=0):
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def checkpoint_layers(total: int) -> int:
    if _gradient_checkpointing_layers is not None:
        return min(total, _gradient_checkpointing_layers)
    if _architecture is not None and _architecture() is not None:
        val = getattr(_architecture(), "_gradient_checkpointing_layers", None)
        if val is not None:
            return min(total, val)
    env_val = os.environ.get("OPENWAM_GRADIENT_CHECKPOINTING_LAYERS")
    if env_val and env_val.strip():
        try:
            return min(total, int(env_val))
        except ValueError:
            pass
    return total


def checkpoint_skip_layers(total: int = 30) -> int:
    return max(0, total - checkpoint_layers(total))


def dispatcher(fn):
    """Let Dynamo inline a thin dispatcher, but never compile it as its own frame.

    A graph break inside an inlined dispatcher makes Dynamo compile the
    dispatcher separately, guarded on the callable argument. Every attention
    callable or per-layer module then recompiles it until recompile_limit.
    """
    try:
        from torch._dynamo.eval_frame import skip_code
    except ImportError:
        logger.warning("[optimizations] torch._dynamo skip_code is unavailable; SAC dispatchers may recompile")
        return fn
    skip_code(fn.__code__)
    return fn


_FA_OPS = frozenset((
    "flash_attn._flash_attn_forward.default", "flash_attn2_c_op.varlen_fwd.default",
    "aten._scaled_dot_product_flash_attention.default", "aten._scaled_dot_product_efficient_attention.default",
    "aten._scaled_dot_product_cudnn_attention.default", "aten._flash_attention_forward.default",
    "aten._efficient_attention_forward.default", "aten._scaled_dot_product_attention_math.default",
    "aten._scaled_dot_product_fused_attention_overrideable.default",
))
_MATH_ATTENTION_OPS = frozenset(("aten.bmm.default", "aten._safe_softmax.default", "aten._softmax.default"))
_ATTENTION_OPS = _FA_OPS | _MATH_ATTENTION_OPS
_GEMM_OPS = frozenset(("aten.mm.default", "aten.addmm.default", "aten.bmm.default"))


def extras_enabled():
    return bool(os.environ.get("OPENWAM_SAC_EXTRA_OPS", "").strip())


def active():
    return any(enabled(name) for name in FLAGS) or extras_enabled()


def ffn_boundary():
    return enabled("OPENWAM_OPT_SAC_FFN_FULL") or enabled("OPENWAM_OPT_SAC_GEMM") or extras_enabled()


def active_ffn_boundary() -> bool:
    return _session.get() is not None and ffn_boundary()


def attention_boundary():
    return enabled("OPENWAM_OPT_SAC_FA") or enabled("OPENWAM_OPT_SAC_GEMM") or extras_enabled()


def projection_boundary():
    return enabled("OPENWAM_OPT_SAC_GEMM") or extras_enabled()


@lru_cache(maxsize=16)
def _extra_ops(raw):
    result = set()
    for identifier in raw.split(","):
        identifier = identifier.strip()
        if not identifier:
            continue
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", identifier):
            raise ValueError(f"SAC extra op must be namespace.op.overload, got {identifier!r}")
        namespace, name, overload = identifier.split(".")
        try:
            op = getattr(getattr(getattr(torch.ops, namespace), name), overload)
        except AttributeError as exc:
            raise ValueError(f"SAC extra op is unavailable: {identifier}") from exc
        if op in getattr(torch_checkpoint, "SAC_IGNORED_OPS", ()) or op._schema.is_mutable:
            raise ValueError(f"SAC cannot cache a metadata or mutating op: {identifier}")
        if identifier in _ATTENTION_OPS:
            raise ValueError("Use OPENWAM_OPT_SAC_FA for attention operators, rather than SAC_EXTRA_OPS")
        result.add(identifier)
    if raw.strip() and not result:
        raise ValueError("OPENWAM_SAC_EXTRA_OPS contains no operator identifiers")
    return frozenset(result)


def layers(total):
    """SAC layer count; unset covers every checkpointed layer."""
    checkpointed = checkpoint_layers(total)
    raw = os.environ.get("OPENWAM_SAC_LAYERS", "all").strip().lower()
    if raw in ("", "all"):
        return checkpointed
    count = _integer("OPENWAM_SAC_LAYERS", checkpointed)
    if count > checkpointed:
        raise ValueError(
            f"OPENWAM_SAC_LAYERS={count} exceeds checkpointed layers={checkpointed} "
            f"(model layers={total}, no-checkpoint tail={total - checkpointed})"
        )
    return count


def layer_range(total):
    """Half-open layer range [start, stop) covered by SAC: the checkpointed tail."""
    stop = checkpoint_layers(total)
    return max(0, stop - layers(total)), stop


def validate(total=None):
    enabled("OPENWAM_CHECKPOINT_STATS")
    for name in ("OPENWAM_OPT_SAC_FFN", "OPENWAM_OPT_SAC_FFN_INPUT"):
        if enabled(name):
            raise ValueError(f"{name} was removed; use OPENWAM_OPT_SAC_FFN_FULL=1 with OPENWAM_SAC_LAYERS=N")
    if not active():
        return
    if total is not None:
        layers(total)
        _extra_ops(os.environ.get("OPENWAM_SAC_EXTRA_OPS", ""))
    elif os.environ.get("OPENWAM_SAC_LAYERS", "").strip().lower() not in ("", "all"):
        _integer("OPENWAM_SAC_LAYERS", 0)


def prepare(architecture):
    set_architecture(architecture)
    if hasattr(architecture, "_gradient_checkpointing_layers"):
        set_gradient_checkpointing_layers(architecture._gradient_checkpointing_layers)
    if not active():
        return
    from openwam.model.video_backbone.wan_backbone import WanBase

    backbone = architecture.video_backbone
    if (type(architecture).__name__ != "DualSystemSelfAttnArchitecture"
            or not isinstance(backbone, WanBase) or backbone.vace is not None):
        raise ValueError("Complete SAC currently requires native Wan dual_system/joint_self_attn without VACE")
    validate(backbone.num_layers)
    for block in (*backbone.dit.blocks, *architecture.action_backbone.blocks):
        ffn = block.ffn
        modules = block.modules() if projection_boundary() else ffn.modules()
        for module in modules:
            if getattr(module, "_compiled_call_impl", None) is not None or hasattr(module.forward, "_torchdynamo_orig_callable"):
                raise ValueError("Complete SAC requires operator regions without separate module/forward compilation")
    if enabled("OPENWAM_OPT_SAC_FA"):
        from openwam.model.video_backbone.wan.models.dit import fused_backend_name

        backend_name = fused_backend_name()
        if backend_name not in ("flash_attention_2", "torch_sdpa"):
            raise ValueError(f"FA SAC does not yet cover backend {backend_name}; do not benchmark partial coverage")
        if backend_name == "flash_attention_2":
            from flash_attn import flash_attn_interface as backend

            expected = ((backend._wrapped_flash_attn_forward, "flash_attn::_flash_attn_forward"),
                        (backend._wrapped_flash_attn_varlen_forward, "flash_attn2_c_op::varlen_fwd"))
            for entry, name in expected:
                op = getattr(entry, "default", entry)
                if getattr(getattr(op, "_schema", None), "name", None) != name:
                    raise ValueError(f"FA SAC requires registered forward op {name}, got {entry}")
    start, stop = layer_range(backbone.num_layers)
    logger.info(
        "[optimizations] standard SAC flags=%s, layers=[%d, %d) of %d, no-checkpoint tail=%d, extra ops=%s",
        {name: enabled(name) for name in FLAGS}, start, stop, backbone.num_layers, backbone.num_layers - stop,
        sorted(_extra_ops(os.environ.get("OPENWAM_SAC_EXTRA_OPS", ""))),
    )


class _Policy:
    def __init__(self):
        self.fa = enabled("OPENWAM_OPT_SAC_FA")
        self.ffn = enabled("OPENWAM_OPT_SAC_FFN_FULL")
        self.gemm = enabled("OPENWAM_OPT_SAC_GEMM")
        self.extra = _extra_ops(os.environ.get("OPENWAM_SAC_EXTRA_OPS", ""))
        self.diagnostics = enabled("OPENWAM_CHECKPOINT_STATS")
        self.seen_extra = set()
        self.seen_fa = False

    def __call__(self, ctx, op, *args, **kwargs):
        name = getattr(getattr(op, "_schema", None), "name", "")
        region = (_regions.get() or ("",))[-1]
        active = region != ""
        choice = None
        if name in _ATTENTION_OPS:
            choice = CheckpointPolicy.MUST_SAVE if (self.fa and active and region.endswith("attention")) else CheckpointPolicy.PREFER_RECOMPUTE
            if choice is CheckpointPolicy.MUST_SAVE:
                self.seen_fa = True
        elif active and region.endswith("ffn") and self.ffn:
            choice = CheckpointPolicy.MUST_SAVE
        elif name in _GEMM_OPS and self.gemm and active:
            choice = CheckpointPolicy.MUST_SAVE
        elif name in self.extra and active:
            choice = CheckpointPolicy.MUST_SAVE
            self.seen_extra.add(name)
        if self.diagnostics and choice is not None:
            tag = "saved" if choice is CheckpointPolicy.MUST_SAVE else "recomputed"
            _counts[(ctx, region, tag, name)] += 1
        return choice


@contextmanager
def _checkpoint_scope(mode, phase, policy):
    if policy is not None and phase == "forward":
        _counts[(phase, "checkpoint", "entered", "layer")] += 1
        if (policy.fa and policy.seen_fa) or (policy.extra and policy.seen_extra == policy.extra):
            _counts[(phase, "checkpoint", "selected", "layer")] += 1
    if mode is None:
        yield
        return
    token = _session.set(mode)
    try:
        with mode:
            yield
    finally:
        _session.reset(token)


def context_fn(layer_id, total):
    """Create fresh standard caches for the configured checkpointed layers."""
    start, stop = layer_range(total) if active() else (0, 0)
    selected = start <= layer_id < stop

    def contexts():
        if selected:
            policy = _Policy()
            forward, replay = create_selective_checkpoint_contexts(policy)
        else:
            forward = replay = None
            policy = None
        return _checkpoint_scope(forward, "forward", policy), _checkpoint_scope(replay, "replay", policy)

    return contexts


def native_mot_expression():
    return _native_mot.get()


def _invoke(region, fn, *args, **kwargs):
    mode = _session.get()
    if mode is None and region not in {"projection", "norm", "pointwise"} and not extras_enabled():
        return fn(*args, **kwargs)
    native = False
    native_token = _native_mot.set(native)
    previous = _regions.get()
    region_stack = previous if region == "attention" and previous and previous[-1].endswith("attention") else (*previous, region)
    region_token = _regions.set(region_stack)
    inside = _inside_mode.get()
    mode_token = _inside_mode.set(inside or mode is not None)
    try:
        if mode is None or inside:
            result = fn(*args, **kwargs)
        else:
            with mode:
                result = fn(*args, **kwargs)
        return result
    finally:
        _inside_mode.reset(mode_token)
        _regions.reset(region_token)
        _native_mot.reset(native_token)


_eager_invoke = torch.compiler.disable(_invoke)


@dispatcher
def _call(region, fn, *args, **kwargs):
    return _eager_invoke(region, fn, *args, **kwargs)


@dispatcher
def run_attention(fn, *args, label="attention", **kwargs):
    if not attention_boundary() or (
        _session.get() is None and not torch.compiler.is_compiling() and not extras_enabled()
    ):
        return fn(*args, **kwargs)
    return _call(label, fn, *args, **kwargs)


@dispatcher
def run_ffn(ffn, x, *, video=False):
    if video:
        from openwam.optimizations.linear_bias2d import video_ffn

        return _call("video_ffn", video_ffn, ffn, x)
    return _call("action_ffn", ffn, x)


@dispatcher
def run_projection(linear, x):
    if not projection_boundary():
        return linear(x)
    return _call("projection", linear, x)


@dispatcher
def run_video_projection(linear, x):
    from openwam.optimizations.linear_bias2d import _video_linear

    return _call("projection", _video_linear, linear, x)


@dispatcher
def run_norm(norm, x):
    if not extras_enabled():
        return norm(x)
    return _call("norm", norm, x)


@dispatcher
def run_pointwise(fn, *args):
    return _call("pointwise", fn, *args)


def begin_step(device):
    _counts.clear()
    torch.cuda.reset_peak_memory_stats(device)


def report_step(step, device, metrics, steps_per_sec, batch_size, opt_step):
    groups = {}
    for (phase, region, choice, op), count in sorted(_counts.items()):
        groups.setdefault(phase, {}).setdefault(region, {}).setdefault(choice, {})[op] = count
    print("[checkpoint_stats] " + json.dumps({
        "rank": int(os.environ.get("RANK", "0")), "step": step, "optimizer_step": opt_step,
        "sac_flags": {name: enabled(name) for name in FLAGS},
        "sac_layers": os.environ.get("OPENWAM_SAC_LAYERS", "all"),
        "skip_layers": checkpoint_skip_layers(), "operators": groups,
        "s_microstep": 1.0 / steps_per_sec,
        "global_samples_s": steps_per_sec * batch_size * int(os.environ.get("WORLD_SIZE", "1")),
        "samples_s_gpu": steps_per_sec * batch_size,
        "loss_total": float(metrics["loss_total"]), "grad_norm": float(metrics["grad_norm"]),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
    }, sort_keys=True), flush=True)
