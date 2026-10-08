"""Exact mask-preserving fast paths for padding and structured MoT attention."""

import logging
import weakref
from collections import OrderedDict
from functools import lru_cache

import torch

from openwam.optimizations import enabled

logger = logging.getLogger(__name__)
_PADDING_PLANS = OrderedDict()


@lru_cache(maxsize=1)
def _varlen_kernel():
    try:
        from flash_attn import flash_attn_varlen_func
    except ImportError:
        logger.warning("[optimizations] FA2 padding skipped: flash_attn_varlen_func unavailable")
        return None
    from openwam.optimizations._fa2_varlen import packed_varlen_kernel

    return packed_varlen_kernel(flash_attn_varlen_func)


def _padding_keys(mask, batch, query_len, key_len):
    # A stride-zero query axis proves query-independent masking without a GPU reduction.
    # An arbitrary dense mask, even if its rows happen to match, stays on SDPA.
    if mask.ndim not in (3, 4) or mask.dtype != torch.bool or mask.shape[0] != batch or mask.shape[-1] != key_len:
        return None
    if mask.ndim == 4 and mask.shape[1] == 1:
        if mask.shape[2] in (1, query_len) and (mask.shape[2] == 1 or mask.stride(2) == 0):
            return mask[:, 0, 0, :]
    if mask.ndim == 3:
        if mask.shape[1] in (1, query_len) and (mask.shape[1] == 1 or mask.stride(1) == 0):
            return mask[:, 0, :]
    return None


def _padding_plan(mask, keep):
    root = mask
    while root._base is not None:
        root = root._base
    try:
        signature = (keep.data_ptr(), tuple(keep.shape), tuple(keep.stride()), keep.device, root._version)
    except RuntimeError:  # Inference tensors have no version counter.
        signature = None
    cached = _PADDING_PLANS.get(signature)
    if cached is not None and cached[0]() is root:
        _PADDING_PLANS.move_to_end(signature)
        return cached[1]
    lengths = keep.sum(dim=-1, dtype=torch.int32)
    nonempty = lengths > 0
    # FA2 builds differ in support for zero-length K. Give empty rows a dummy
    # first key, then zero their result (and gradients), matching SDPA semantics.
    positions = torch.arange(keep.shape[1], device=keep.device)
    packed_keep = keep | ((~nonempty).unsqueeze(1) & (positions == 0).unsqueeze(0))
    indices = packed_keep.reshape(-1).nonzero(as_tuple=False).flatten()
    offsets = torch.cat([lengths.new_zeros(1), lengths.clamp_min(1).cumsum(0, dtype=torch.int32)])
    plan = indices, offsets, nonempty
    if signature is not None:
        _PADDING_PLANS[signature] = weakref.ref(root), plan
    while len(_PADDING_PLANS) > 16:
        _PADDING_PLANS.popitem(last=False)
    return plan


def fa2_padding(q, k, v, mask, num_heads):
    if (
        enabled("OPENWAM_OPT_FA2_PADDING")
        and enabled("OPENWAM_OPT_GLOBAL_COMPILE")
        and torch.compiler.is_compiling()
    ):
        return _eager_fa2_padding(q, k, v, mask, num_heads)
    return _fa2_padding(q, k, v, mask, num_heads)


def _fa2_padding(q, k, v, mask, num_heads):
    """Return (B,Sq,H*D) or None when the mask/backend cannot use this fast path."""
    if not enabled("OPENWAM_OPT_FA2_PADDING") or mask is None:
        return None
    if not q.is_cuda or q.dtype not in (torch.float16, torch.bfloat16):
        return None
    if k.dtype != q.dtype or v.dtype != q.dtype or k.device != q.device or v.device != q.device:
        return None
    batch, sq, dim = q.shape
    sk = k.shape[1]
    if k.shape != v.shape or k.shape[0] != batch or k.shape[2] != dim:
        return None
    if dim % num_heads or dim // num_heads > 256 or dim // num_heads % 8 or not sq or not sk:
        return None
    keep = _padding_keys(mask, batch, sq, sk)
    if keep is None or mask.device != q.device:
        return None
    kernel = _varlen_kernel()
    if kernel is None:
        return None
    indices, cu_k, nonempty = _padding_plan(mask, keep)
    head_dim = dim // num_heads
    qp = q.reshape(batch * sq, num_heads, head_dim).contiguous()
    kp = k.reshape(batch * sk, num_heads, head_dim).index_select(0, indices).contiguous()
    vp = v.reshape(batch * sk, num_heads, head_dim).index_select(0, indices).contiguous()
    cu_q = torch.arange(batch + 1, device=q.device, dtype=torch.int32) * sq
    out = kernel(qp, kp, vp, cu_q, cu_k, sq, sk, dropout_p=0.0, causal=False)
    return out.reshape(batch, sq, dim).masked_fill((~nonempty).view(batch, 1, 1), 0)


_eager_fa2_padding = torch.compiler.disable(_fa2_padding)


def split_mask_plan(mask):
    """Group consecutive CPU mask rows with exactly the same visible key set."""
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.device.type != "cpu":
        return None
    plan = []
    start = 0
    while start < mask.shape[0]:
        stop = start + 1
        while stop < mask.shape[0] and torch.equal(mask[start], mask[stop]):
            stop += 1
        indices = mask[start].nonzero(as_tuple=False).flatten().tolist()
        ranges = []
        for index in indices:
            if ranges and ranges[-1][1] == index:
                ranges[-1] = (ranges[-1][0], index + 1)
            else:
                ranges.append((index, index + 1))
        plan.append((start, stop, tuple(ranges)))
        if len(plan) > 32 or len(ranges) > 8:
            return None  # Keep irregular masks on SDPA instead of multiplying launches.
        start = stop
    return tuple(plan)


def split_attention(q, k, v, mask, num_heads, attention_fn):
    plan = getattr(mask, "_openwam_split_plan", None)
    if not enabled("OPENWAM_OPT_MOT_SPLIT_ATTN") or plan is None:
        return None
    if q.shape[1] != mask.shape[-2] or k.shape[1] != mask.shape[-1]:
        return None
    # Reject plans after in-place edits to an otherwise identical-shaped mask.
    if mask._version != getattr(mask, "_openwam_split_version", None):
        return None
    outputs = []
    for start, stop, ranges in plan:
        query = q[:, start:stop]
        if not ranges:
            outputs.append(query * 0 + (k.sum() + v.sum()) * 0)
            continue
        keys = [k[:, left:right] for left, right in ranges]
        values = [v[:, left:right] for left, right in ranges]
        keys = keys[0] if len(keys) == 1 else torch.cat(keys, dim=1)
        values = values[0] if len(values) == 1 else torch.cat(values, dim=1)
        outputs.append(attention_fn(query, keys, values, num_heads=num_heads))
    return torch.cat(outputs, dim=1)
