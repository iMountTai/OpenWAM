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


def compile_mot_layer(driver):
    fragmented = any(enabled(name) for name in (
        "OPENWAM_OPT_MOT_SPLIT_ATTN", "OPENWAM_OPT_FA2_PADDING", "OPENWAM_OPT_SAC_FFN",
    ))
    driver._compiled_attention = (
        torch.compiler.disable(driver._mixed_attention)
        if enabled("OPENWAM_OPT_MOT_SPLIT_ATTN") else driver._mixed_attention
    )

    def layer(video_block, action_block, vstate, astate, attn_mask):
        # Block lookup stays outside Dynamo so all equally shaped layers share
        # graphs instead of specializing on thirty ModuleList indices.
        return driver._step_impl(
            0, vstate, astate, attn_mask=attn_mask, suppress_inner_attn_ckpt=True,
            _blocks=(video_block, action_block),
        )

    logger.info("[optimizations] MoT block compile enabled; fullgraph=%s", not fragmented)
    return torch.compile(layer, dynamic=False, fullgraph=not fragmented, options=_options())


def compile_vae_encoder(vae):
    encoder = vae.model.encoder
    # Replace the callable, rather than the module, to retain checkpoint keys,
    # type checks and the causal feature-cache protocol.
    encoder.forward = torch.compile(encoder.forward, dynamic=False, fullgraph=True, options=_options())
    logger.info("[optimizations] native VAE encoder module compile enabled (fullgraph)")
