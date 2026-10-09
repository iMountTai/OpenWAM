"""Compile training blocks, with explicit boundaries for optional eager kernels."""

import logging
import os

import torch

from openwam.optimizations import enabled

logger = logging.getLogger(__name__)


def compile_scope():
    if not enabled("OPENWAM_OPT_GLOBAL_COMPILE"):
        return set()
    scope = os.environ.get("OPENWAM_GLOBAL_COMPILE_SCOPE", "all").strip().lower()
    if scope not in ("all", "mot", "vae"):
        raise ValueError("OPENWAM_GLOBAL_COMPILE_SCOPE must be all, mot or vae")
    return {"mot", "vae"} if scope == "all" else {scope}


def _options():
    import torch._inductor

    # Retain BF16 rounding at the original expression boundaries.
    return {"emulate_precision_casts": True} if "emulate_precision_casts" in torch._inductor.list_options() else {}


def mark_mot_text_dynamic(vstate, astate):
    """Annotate prepared text inputs before entering the compiled MoT layer."""
    if not enabled("OPENWAM_OPT_TEXT_TRIM"):
        return
    from torch._dynamo import mark_dynamic

    payload = astate.payload
    tensors = (
        (vstate.context, 1),
        (vstate.context_mask, -1),
        (payload.context, 1),
        # Action masks can be [B, L], [B, S_action, L] or [B, H, S_action, L].
        (payload.context_mask, -1),
    )
    for tensor, dim in tensors:
        if tensor is None:
            continue
        axis = dim if dim >= 0 else tensor.ndim + dim
        # PyTorch specializes sizes 0/1; leave those as separate static cases.
        if tensor.shape[axis] <= 1 or axis in getattr(tensor, "_dynamo_dynamic_indices", ()):
            continue
        mark_dynamic(tensor, axis)


def compile_mot_layer(driver):
    from openwam.optimizations.sac import attention_boundary, active

    fragmented = any(enabled(name) for name in (
        "OPENWAM_OPT_MOT_SPLIT_ATTN", "OPENWAM_OPT_FA2_PADDING", "OPENWAM_OPT_SAC_FFN",
    )) or active()
    driver._compiled_attention = (
        torch.compiler.disable(driver._mixed_attention)
        if enabled("OPENWAM_OPT_MOT_SPLIT_ATTN") or attention_boundary() else driver._mixed_attention
    )

    def layer(video_block, action_block, vstate, astate, attn_mask):
        # Block lookup stays outside Dynamo so all equally shaped layers share
        # graphs instead of specializing on thirty ModuleList indices.
        return driver._step_impl(
            0, vstate, astate, attn_mask=attn_mask, suppress_inner_attn_ckpt=True,
            _blocks=(video_block, action_block),
        )

    logger.info(
        "[optimizations] MoT block compile enabled; fullgraph=%s, dynamic text length=%s",
        not fragmented, enabled("OPENWAM_OPT_TEXT_TRIM"),
    )
    return torch.compile(layer, dynamic=False, fullgraph=not fragmented, options=_options())


def compile_vae_encoder(vae):
    encoder = vae.model.encoder
    # Replace the callable, rather than the module, to retain checkpoint keys,
    # type checks and the causal feature-cache protocol.
    encoder.forward = torch.compile(encoder.forward, dynamic=False, fullgraph=True, options=_options())
    logger.info("[optimizations] native VAE encoder module compile enabled (fullgraph)")
