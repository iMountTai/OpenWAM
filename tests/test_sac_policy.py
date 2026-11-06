import os
import pytest
import torch
from torch.utils.checkpoint import CheckpointPolicy

from openwam.optimizations.sac import _Policy, _checkpoint_scope, _regions, _session


class _MockOp:
    def __init__(self, name: str):
        self._name = name

    def __str__(self) -> str:
        return self._name


class _MockContext:
    def __init__(self, is_recompute: bool = False):
        self.is_recompute = is_recompute


def test_sac_policy_attention_and_ffn_matching(monkeypatch):
    monkeypatch.setenv("OPENWAM_OPT_SAC_FA", "1")
    monkeypatch.setenv("OPENWAM_OPT_SAC_FFN_FULL", "1")
    monkeypatch.setenv("OPENWAM_OPT_SAC_GEMM", "1")

    policy = _Policy()
    ctx = _MockContext(is_recompute=False)

    # In attention region with FA op -> MUST_SAVE
    _regions.set(("attention",))
    assert policy(ctx, _MockOp("flash_attn._flash_attn_forward.default")) == CheckpointPolicy.MUST_SAVE
    assert policy.seen_fa is True

    # In video_ffn region with GEMM op -> MUST_SAVE
    _regions.set(("video_ffn",))
    assert policy(ctx, _MockOp("aten.mm.default")) == CheckpointPolicy.MUST_SAVE

    # In action_ffn region -> MUST_SAVE
    _regions.set(("action_ffn",))
    assert policy(ctx, _MockOp("aten.addmm.default")) == CheckpointPolicy.MUST_SAVE

    # Outside instrumented regions -> PREFER_RECOMPUTE
    _regions.set(("norm",))
    assert policy(ctx, _MockOp("aten.add.Tensor")) == CheckpointPolicy.PREFER_RECOMPUTE


def test_sac_checkpoint_scope_context_manager(monkeypatch):
    monkeypatch.setenv("OPENWAM_OPT_SAC_FA", "1")
    policy = _Policy()
    policy.seen_fa = True

    dummy_mode = object()
    with _checkpoint_scope(dummy_mode, "forward", policy):
        assert _session.get() is dummy_mode
    assert _session.get() is None


def test_sac_checkpoint_scope_raises_on_unseen_fa(monkeypatch):
    monkeypatch.setenv("OPENWAM_OPT_SAC_FA", "1")
    policy = _Policy()
    policy.seen_fa = False

    dummy_mode = object()
    with pytest.raises(RuntimeError, match="FA SAC did not intercept an attention forward op"):
        with _checkpoint_scope(dummy_mode, "forward", policy):
            pass
