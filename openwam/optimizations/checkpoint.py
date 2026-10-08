"""Selective caching for the video FFN output projection in MoT checkpoints."""

from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch.utils.checkpoint import CheckpointPolicy, create_selective_checkpoint_contexts


_ffn_mode = ContextVar("openwam_ffn_checkpoint_mode", default=None)


class _FFNOutputPolicy:
    def __init__(self, weight, input_weight=None):
        weights = (weight,) if input_weight is None else (weight, input_weight)
        self.targets = {(w.data_ptr(), (w.shape[1], w.shape[0])) for w in weights}

    def __call__(self, context, op, *args, **kwargs):
        if op == torch.ops.aten.addmm.default:
            matrix = args[2]
        elif op == torch.ops.aten.mm.default:
            matrix = args[1]
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE
        # Match this layer's parameter, rather than all projections with the
        # same shape. The transposed Linear weight shares its data pointer.
        if (matrix.data_ptr(), tuple(matrix.shape)) in self.targets:
            return CheckpointPolicy.MUST_SAVE
        return CheckpointPolicy.PREFER_RECOMPUTE


@contextmanager
def _ffn_scope(mode):
    token = _ffn_mode.set(mode)
    try:
        yield
    finally:
        _ffn_mode.reset(token)


def ffn_output_context_fn(weight, input_weight=None):
    """Create a per-checkpoint cache, dispatched only inside the video FFN."""
    def contexts():
        forward, recompute = create_selective_checkpoint_contexts(_FFNOutputPolicy(weight, input_weight))
        return _ffn_scope(forward), _ffn_scope(recompute)

    return contexts


def run_ffn(ffn, x):
    from openwam.optimizations import enabled

    if enabled("OPENWAM_OPT_GLOBAL_COMPILE") and torch.compiler.is_compiling():
        return _eager_run_ffn(ffn, x)
    mode = _ffn_mode.get()
    if mode is None:
        return ffn(x)
    # Keep attention, normalization and compiled pointwise dispatch unchanged.
    # Standard SAC still owns the cache and checks for mutated saved outputs.
    with mode:
        return ffn(x)


_eager_run_ffn = torch.compiler.disable(run_ffn)
