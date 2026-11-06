"""Optional training kernels (openwam.model.kernels) against their PyTorch references."""

import pytest
import torch
import torch.nn.functional as F

from openwam.model.kernels import attention, joint_cat, rope_triton
from openwam.model.video_backbone.wan.models.dit import precompute_freqs_cis_3d, rope_apply


def _grads(out, inputs, grad):
    return torch.autograd.grad(out, inputs, grad)


def test_cat_split_fallback_is_torch_cat_split():
    a, b = torch.randn(2, 5, 8), torch.randn(2, 3, 8)
    assert torch.equal(joint_cat.cat_seq(a, b), torch.cat([a, b], dim=1))
    x = torch.randn(2, 8, 8)
    head, tail = joint_cat.split_seq(x, 5)
    assert torch.equal(head, x[:, :5]) and torch.equal(tail, x[:, 5:])
    assert head.is_contiguous() and tail.is_contiguous()


def test_sdpa_without_cuda_is_the_default_call():
    q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
    mask = torch.rand(2, 1, 6, 6) > 0.3
    assert torch.equal(attention.sdpa(q, k, v, attn_mask=mask), F.scaled_dot_product_attention(q, k, v, attn_mask=mask))


needs_triton = pytest.mark.skipif(
    not (torch.cuda.is_available() and joint_cat._HAVE_TRITON and rope_triton._HAVE_TRITON),
    reason="needs CUDA and Triton",
)


@pytest.mark.gpu
@needs_triton
def test_triton_joint_cat_and_split_are_bitwise(monkeypatch):
    monkeypatch.setattr(joint_cat, "_ENABLED", True)
    kw = dict(device="cuda", dtype=torch.bfloat16, requires_grad=True)
    a, b = torch.randn(3, 330, 3072, **kw), torch.randn(3, 33, 3072, **kw)
    out, ref = joint_cat.cat_seq(a, b), torch.cat([a, b], dim=1)
    assert torch.equal(out, ref)
    g = torch.randn_like(out)
    for got, want in zip(_grads(out, (a, b), g), _grads(ref, (a, b), g)):
        assert torch.equal(got, want)

    x = torch.randn(3, 363, 3072, **kw)
    parts, ref_parts = joint_cat.split_seq(x, 330), x.split([330, 33], dim=1)
    for got, want in zip(parts, ref_parts):
        assert got.is_contiguous() and torch.equal(got, want)
    gs = [torch.randn_like(p) for p in parts]
    assert torch.equal(_grads(parts, x, gs)[0], _grads(ref_parts, x, gs)[0])


@pytest.mark.gpu
@needs_triton
def test_fused_rope_is_bitwise_rope_apply(monkeypatch):
    monkeypatch.setenv("OPENWAM_FUSED_ROPE", "1")
    num_heads, head_dim, (f, h, w) = 24, 128, (3, 6, 10)
    f_freqs, h_freqs, w_freqs = precompute_freqs_cis_3d(head_dim)
    freqs = (
        torch.cat(
            [
                f_freqs[:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                h_freqs[:h].view(1, h, 1, -1).expand(f, h, w, -1),
                w_freqs[:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        )
        .reshape(f * h * w, 1, -1)
        .cuda()
    )
    x = torch.randn(2, f * h * w, num_heads * head_dim, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    assert rope_triton.fused_rope_ok(x, freqs, num_heads)

    out, ref = rope_triton.fused_rope_apply(x, freqs, num_heads), rope_apply(x, freqs, num_heads)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
    g = torch.randn_like(out)
    got, want = _grads(out, x, g)[0], _grads(ref, x, g)[0]
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))


def test_fused_rope_ok_supports_2d_and_3d_freqs(monkeypatch):
    monkeypatch.setenv("OPENWAM_FUSED_ROPE", "1")
    monkeypatch.setattr(rope_triton, "_HAVE_TRITON", True)
    num_heads, head_dim = 16, 64
    s = 50
    # Dummy mock tensor with cuda properties
    class MockCudaTensor:
        def __init__(self, shape, dtype=torch.bfloat16):
            self.shape = shape
            self.dtype = dtype
            self.is_cuda = True
            self.device = torch.device("cuda:0")
            self.ndim = len(shape)
        def dim(self):
            return len(self.shape)
        def is_contiguous(self):
            return True

    x = MockCudaTensor((16, s, num_heads * head_dim), dtype=torch.bfloat16)
    # 3D freqs: (S, 1, head_dim // 2)
    freqs_3d = MockCudaTensor((s, 1, head_dim // 2), dtype=torch.complex128)
    assert rope_triton.fused_rope_ok(x, freqs_3d, num_heads) is True

    # 2D freqs: (S, head_dim // 2)
    freqs_2d = MockCudaTensor((s, head_dim // 2), dtype=torch.complex128)
    assert rope_triton.fused_rope_ok(x, freqs_2d, num_heads) is True

    # Bad shape
    freqs_bad = MockCudaTensor((s, head_dim // 4), dtype=torch.complex128)
    assert rope_triton.fused_rope_ok(x, freqs_bad, num_heads) is False


def test_rope_apply_auto_2d_freqs_reference_call():
    calls = []
    def mock_ref(x, freqs, num_heads):
        calls.append((x.shape, freqs.shape, num_heads))
        return x

    # CPU tensors - fused_rope_ok returns False, falls back to reference_fn
    x = torch.randn(2, 10, 32)
    freqs_2d = torch.zeros(10, 16, dtype=torch.complex128)
    out = rope_triton.rope_apply_auto(x, freqs_2d, 2, mock_ref)
    assert len(calls) == 1
    # reference_fn receives 3D unsqueezed freqs
    assert calls[0][1] == (10, 1, 16)
    assert torch.equal(out, x)
