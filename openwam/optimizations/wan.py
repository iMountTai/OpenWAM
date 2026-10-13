"""Instance-local Wan adapters (projection/SAC, stored latents); keep the upstream backbone intact."""

from __future__ import annotations

import inspect
import types
from typing import TYPE_CHECKING, Tuple

from torch import Tensor

from openwam.model.video_backbone.wan import dit_forward as wan_dit_forward
from openwam.model.video_backbone.wan import encode as wan_encode
from openwam.model.video_backbone.wan.models.dit import modulate, rope_apply
from openwam.model.video_backbone.wan.preprocess import check_resize_height_width
from openwam.optimizations.linear_bias2d import cross_attention_kwargs, video_ffn, video_linear

if TYPE_CHECKING:
    from openwam.model.video_backbone.base import BlockLoopState

def pre_attn_at_layer_for_compile(
    self, layer_id: int, state: BlockLoopState, *, block=None
) -> Tuple[Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
    """Compile-friendly Wan pre-attention half using a tensor tuple post-state."""
    if block is None:
        block = state.extras["dit"].blocks[layer_id]

    time_modulation = state.time_mod
    has_seq = time_modulation.dim() == 4
    chunk_dim = 2 if has_seq else 1
    chunks = (
        block.modulation.to(dtype=time_modulation.dtype, device=time_modulation.device) + time_modulation
    ).chunk(6, dim=chunk_dim)
    if has_seq:
        chunks = tuple(c.squeeze(2) for c in chunks)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = chunks

    residual_x = state.hidden_states
    from openwam.optimizations.sac import run_norm

    attn_input = modulate(run_norm(block.norm1, state.hidden_states), shift_msa, scale_msa)

    self_attn = block.self_attn
    q = run_norm(self_attn.norm_q, video_linear(self_attn.q, attn_input))
    k = run_norm(self_attn.norm_k, video_linear(self_attn.k, attn_input))
    v = video_linear(self_attn.v, attn_input)
    q = rope_apply(q, state.rope_freqs, self_attn.num_heads)
    k = rope_apply(k, state.rope_freqs, self_attn.num_heads)

    post_state = (residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp)
    return q, k, v, post_state

def post_attn_at_layer_for_compile(
    self, layer_id: int, state: BlockLoopState, attn_out: Tensor, post_state: tuple[Tensor, ...], *, block=None
) -> BlockLoopState:
    """Compile-friendly Wan post-attention half consuming a tensor tuple."""
    if block is None:
        block = state.extras["dit"].blocks[layer_id]
    self_attn = block.self_attn
    residual_x, gate_msa, shift_mlp, scale_mlp, gate_mlp = post_state

    hidden_states = block.gate(residual_x, gate_msa, video_linear(self_attn.o, attn_out))
    context_mask = None
    if state.context_mask is not None:
        context_mask = state.context_mask.unsqueeze(1).expand(-1, hidden_states.shape[1], -1).unsqueeze(1)
    from openwam.optimizations.sac import ffn_boundary, run_norm
    from openwam.optimizations.sac import run_ffn as run_complete_ffn

    hidden_states = hidden_states + block.cross_attn(
        run_norm(block.norm3, hidden_states), state.context, ctx_mask=context_mask,
        **cross_attention_kwargs(block.cross_attn),
    )
    mlp_input = modulate(run_norm(block.norm2, hidden_states), shift_mlp, scale_mlp)
    if ffn_boundary():
        mlp_output = run_complete_ffn(block.ffn, mlp_input, video=True)
    else:
        mlp_output = video_ffn(block.ffn, mlp_input)
    hidden_states = block.gate(hidden_states, gate_mlp, mlp_output)
    state.hidden_states = hidden_states

    wan_dit_forward.apply_post_block_residuals(layer_id, state)
    return state


def prepare(backbone):
    if getattr(backbone, "_openwam_wan_adapter", False):
        return
    replacements = (pre_attn_at_layer_for_compile, post_attn_at_layer_for_compile)
    for function in replacements:
        original = getattr(backbone, function.__name__)
        expected = tuple(inspect.signature(function).parameters)[1:-1]  # self, optional block
        if tuple(inspect.signature(original).parameters) != expected:
            raise RuntimeError(f"Unsupported native Wan interface: {function.__name__}")
    for function in replacements:
        setattr(backbone, function.__name__, types.MethodType(function, backbone))
    backbone._openwam_wan_adapter = True


def preprocess_stored_latents(self, *, frames=None, text=None, video_latents=None, video_size=None, **kw) -> dict:
    """Native preprocessing with stored latents in place of pixels and the VAE.

    Mirrors WanBase.preprocess_input_for_train: same resize check, text
    encoding, TI2V conditioning and output keys. Batches without stored latents
    go through the native method unchanged.
    """
    if video_latents is None:
        return self._openwam_native_preprocess(frames=frames, text=text, **kw)
    device = self.device
    dtype = self.dtype
    num_frames, height, width = video_size
    height, width, num_frames = check_resize_height_width(
        height,
        width,
        num_frames,
        height_division_factor=self._height_division_factor,
        width_division_factor=self._width_division_factor,
        time_division_factor=self._time_division_factor,
        time_division_remainder=self._time_division_remainder,
    )
    batch_size = len(frames)
    context, seq_lens = wan_encode.encode_text(
        text, tokenizer=self._tokenizer, text_encoder=self.text_encoder, device=self.device
    )
    input_latents = video_latents.to(dtype=dtype, device=device)
    cond = self._variant.build_train_conditioning(
        self,
        input_latents=input_latents,
        frames=frames,
        ref_images=kw.get("ref_images"),
        vace_videos=kw.get("vace_videos"),
        stacked_inputs=None,
        B=batch_size,
        num_frames=num_frames,
        height=height,
        width=width,
        device=device,
        dtype=dtype,
        first_frame_image=kw.get("first_frame_image"),
    )
    return {
        "input_latents": input_latents,
        "context": context,
        "seq_lens": seq_lens,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "vace_context": cond.get("vace_context"),
        "fuse_vae_embedding_in_latents": cond.get("fuse_vae_embedding_in_latents", False),
        "num_clean_prefix_frames": cond.get("num_clean_prefix_frames", 0),
        "first_frame_latents": cond.get("first_frame_latents"),
        "clip_feature": cond.get("clip_feature"),
        "y": cond.get("y"),
    }


def prepare_stored_latents(backbone):
    """Route stored-latent batches around the VAE on this backbone instance only."""
    if getattr(backbone, "_openwam_native_preprocess", None) is not None:
        return
    original = backbone.preprocess_input_for_train
    if tuple(inspect.signature(original).parameters) != ("frames", "text", "kw"):
        raise RuntimeError("Unsupported native Wan interface: preprocess_input_for_train")
    backbone._openwam_native_preprocess = original
    backbone.preprocess_input_for_train = types.MethodType(preprocess_stored_latents, backbone)
