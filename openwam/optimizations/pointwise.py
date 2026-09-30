"""Compile only stateless tensor expressions, outside ZeRO/checkpoint orchestration."""

from functools import lru_cache

import torch

from openwam.optimizations import enabled


@lru_cache(maxsize=None)
def _compiled(fn):
    return torch.compile(fn, dynamic=True, fullgraph=True)


def _run(fn, *args):
    if args[0].is_cuda and enabled("OPENWAM_OPT_POINTWISE_COMPILE"):
        return _compiled(fn)(*args)
    return fn(*args)


def _rms(x, weight, eps):
    normed = x.float() * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return normed.to(x.dtype) * weight


def rms_norm(x, weight, eps):
    return _run(_rms, x, weight, eps)


def _modulate(x, shift, scale):
    return x * (1 + scale) + shift


def modulate(x, shift, scale):
    return _run(_modulate, x, shift, scale)


def _gate(x, gate_value, residual):
    return x + gate_value * residual


def gate(x, gate_value, residual):
    return _run(_gate, x, gate_value, residual)


def _rotate(x, freqs, use_fp32):
    work_dtype = torch.float32 if use_fp32 else torch.float64
    pairs = x.to(work_dtype).reshape(*x.shape[:-1], -1, 2)
    real = freqs.real.to(device=x.device, dtype=work_dtype)
    imag = freqs.imag.to(device=x.device, dtype=work_dtype)
    left, right = pairs[..., 0], pairs[..., 1]
    rotated = torch.stack((left * real - right * imag, left * imag + right * real), dim=-1)
    return rotated.flatten(-2).to(x.dtype)


def rotate(x, freqs):
    return _run(_rotate, x, freqs, enabled("OPENWAM_OPT_ROPE_FP32"))
