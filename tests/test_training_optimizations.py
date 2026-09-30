"""CPU numerical/dispatch checks; GPU kernels and multi-rank training need remote checks."""

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from openwam.optimizations import (
    attention,
    configure_zero,
    enabled,
    pointwise,
    prepare_model,
    prepare_runtime_constants,
    stage_training_indices,
)
from openwam.optimizations.text import cached_text
from openwam.optimizations.vae_fused._concat_layout import FrozenWeightCache, canonicalize
from openwam.optimizations.vae_fused.concat_conv_bias import fused_concat_conv_bias

ROOT = Path(__file__).resolve().parents[1]


def load_source(name, path):
    # Load self-contained source files without importing optional model registries
    # (Transformers, remote model assets and training-only packages).
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


vae = load_source("opt_test_vae", "openwam/model/video_backbone/wan/models/vae.py")
preprocess = load_source("opt_test_preprocess", "openwam/model/video_backbone/wan/preprocess.py")
masks = load_source("opt_test_masks", "openwam/model/architectures/utils/mask_modes.py")


def unmasked(q, k, v, num_heads):
    b, sq, dim = q.shape
    qh = q.reshape(b, sq, num_heads, -1).transpose(1, 2)
    kh = k.reshape(b, k.shape[1], num_heads, -1).transpose(1, 2)
    vh = v.reshape(b, v.shape[1], num_heads, -1).transpose(1, 2)
    return F.scaled_dot_product_attention(qh, kh, vh).transpose(1, 2).reshape(b, sq, dim)


class VideoMask:
    def __init__(self, mode):
        self.mode = mode

    def build_video_to_video_mask(self, video_seq_len, video_tokens_per_frame, device):
        positions = torch.arange(video_seq_len, device=device)
        if self.mode == "per_frame_causal":
            return (positions // video_tokens_per_frame).unsqueeze(1) >= (positions // video_tokens_per_frame)
        result = torch.ones(video_seq_len, video_seq_len, dtype=torch.bool, device=device)
        if self.mode == "first_frame_causal":
            result[:video_tokens_per_frame, video_tokens_per_frame:] = False
        return result


class TrainingOptimizationTests(unittest.TestCase):
    def setUp(self):
        flags = {name: "0" for name in os.environ if name.startswith("OPENWAM_OPT_")}
        self.env = patch.dict(os.environ, flags)
        self.env.start()
        self.addCleanup(self.env.stop)
        torch.manual_seed(31)

    def test_flag_values_and_zero_config(self):
        self.assertFalse(enabled("OPENWAM_OPT_ZERO_OVERLAP"))
        config = {"zero_optimization": {"stage": 2, "overlap_comm": False}}
        configure_zero(config)
        self.assertFalse(config["zero_optimization"]["overlap_comm"])
        for value in ("1", "true", " YES ", "on"):
            with patch.dict(os.environ, OPENWAM_OPT_ZERO_OVERLAP=value):
                configure_zero(config)
                self.assertTrue(config["zero_optimization"]["contiguous_gradients"])
        with patch.dict(os.environ, OPENWAM_OPT_ZERO_OVERLAP="typo"):
            with self.assertRaises(ValueError):
                enabled("OPENWAM_OPT_ZERO_OVERLAP")

    def test_causal_conv_channels_last_streaming_and_gradients(self):
        conv = vae.CausalConv3d(3, 4, 3, padding=1).double()
        for cache_t in (0, 1, 2):
            with self.subTest(cache_t=cache_t):
                x = torch.randn(2, 3, 4, 5, 6, dtype=torch.float64, requires_grad=True)
                cache = torch.randn(2, 3, cache_t, 5, 6, dtype=torch.float64) if cache_t else None
                ref = conv(x, cache)
                ref_grad = torch.autograd.grad(ref.square().mean(), (x, conv.weight))
                with patch.dict(os.environ, OPENWAM_OPT_VAE_CHANNELS_LAST="1"):
                    prepare_model(SimpleNamespace(video_backbone=SimpleNamespace(vae=conv)))
                    actual = conv(x, cache)
                    grad = torch.autograd.grad(actual.square().mean(), (x, conv.weight))
                torch.testing.assert_close(actual, ref)
                for a, b in zip(grad, ref_grad):
                    torch.testing.assert_close(a, b)
                self.assertTrue(actual.is_contiguous(memory_format=torch.channels_last_3d))

    def test_training_indices_preserve_cpu_rng_and_use_async_pinned_transfer(self):
        from unittest.mock import Mock

        indices = torch.randint(0, 1000, (16,))
        rng = torch.get_rng_state().clone()
        self.assertIs(stage_training_indices(indices, "cpu"), indices)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        fake = Mock(device=torch.device("cpu"))
        result = stage_training_indices(fake, "cuda:1")
        fake.pin_memory.assert_called_once_with()
        fake.pin_memory.return_value.to.assert_called_once_with(device=torch.device("cuda:1"), non_blocking=True)
        self.assertIs(result, fake.pin_memory.return_value.to.return_value)

    def test_split_attention_checkpoint_recomputation(self):
        from torch.utils.checkpoint import checkpoint

        mask = torch.ones(7, 7, dtype=torch.bool)
        mask[:2, 2:] = False
        mask._openwam_split_plan = attention.split_mask_plan(mask)
        mask._openwam_split_version = mask._version
        inputs = [torch.randn(2, 7, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]

        def run(q, k, v, saved_mask):
            result = attention.split_attention(q, k, v, saved_mask, 2, unmasked)
            self.assertIsNotNone(result)
            return result

        with patch.dict(os.environ, OPENWAM_OPT_MOT_SPLIT_ATTN="1"):
            ref = run(*inputs, mask)
            ref_grad = torch.autograd.grad(ref.square().sum(), inputs)
            actual = checkpoint(run, *inputs, mask, use_reentrant=False)
            grad = torch.autograd.grad(actual.square().sum(), inputs)
        torch.testing.assert_close(actual, ref)
        for a, b in zip(grad, ref_grad):
            torch.testing.assert_close(a, b)

    def test_streaming_resample_layout(self):
        resample = vae.Resample(4, "downsample3d").double()
        first = torch.randn(2, 4, 1, 8, 8, dtype=torch.float64)
        second = torch.randn(2, 4, 4, 8, 8, dtype=torch.float64)

        def run():
            cache = [None]
            a, _, _ = resample(first, cache, [0])
            b, _, _ = resample(second, cache, [0])
            return a, b, cache[0]

        ref = run()
        with patch.dict(os.environ, OPENWAM_OPT_VAE_CHANNELS_LAST="1"):
            prepare_model(SimpleNamespace(video_backbone=SimpleNamespace(vae=resample)))
            actual = run()
        for a, b in zip(actual, ref):
            torch.testing.assert_close(a, b)
            self.assertTrue(a.is_contiguous(memory_format=torch.channels_last_3d))

    def test_vae_constants_are_nonpersistent_and_follow_device_dtype(self):
        wrapper = vae.WanVideoVAE.__new__(vae.WanVideoVAE)
        torch.nn.Module.__init__(wrapper)
        wrapper.model = torch.nn.Conv3d(3, 4, 1).double()
        wrapper.scale = [torch.tensor([1.0, 2.0, 3.0, 4.0]), torch.tensor([0.5, 0.25, 0.2, 0.1])]
        scheduler = SimpleNamespace(
            timesteps=torch.arange(4.0), sigmas=torch.rand(4), linear_timesteps_weights=torch.rand(4)
        )
        arch = SimpleNamespace(
            device=torch.device("cpu"), video_backbone=SimpleNamespace(vae=wrapper, scheduler=scheduler)
        )
        with patch.dict(os.environ, OPENWAM_OPT_CONSTANT_CACHE="1"):
            prepare_runtime_constants(arch)
            values = wrapper._scale_for(torch.zeros(1, dtype=torch.float64))
            self.assertEqual(values[0].data_ptr(), wrapper._openwam_scale_0.data_ptr())
            wrapper.float()
            self.assertEqual(wrapper._scale_for(torch.zeros(1))[0].dtype, torch.float32)
        self.assertFalse(any("openwam_scale" in name for name in wrapper.state_dict()))
        torch.testing.assert_close(values[0], wrapper.scale[0].double())

    def test_small_complete_wan22_encoder_layout_and_constants(self):
        model = vae.VideoVAE38_(dim=8, z_dim=2, dec_dim=8, dim_mult=[1, 2, 2, 2], num_res_blocks=1).eval().double()
        x = torch.randn(2, 3, 9, 16, 16, dtype=torch.float64)
        scale = [torch.tensor([0.2, -0.3]), torch.tensor([0.5, 1.2])]
        with torch.no_grad():
            ref = model.encode(x, scale)
            with patch.dict(os.environ, OPENWAM_OPT_VAE_CHANNELS_LAST="1"):
                prepare_model(SimpleNamespace(video_backbone=SimpleNamespace(vae=model)))
                actual = model.encode(x, [s.double() for s in scale])
        torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-7)

    def test_real_checkpoint_method_round_trips_channels_last_weights(self):
        from safetensors.torch import load_file

        compile_options = load_source("opt_test_compile_options", "openwam/model/compile_options.py")
        model_namespace = ModuleType("openwam.model")
        model_namespace.__path__ = [str(ROOT / "openwam/model")]
        with patch.dict(
            sys.modules, {"openwam.model": model_namespace, "openwam.model.compile_options": compile_options}
        ):
            base = load_source("opt_test_base", "openwam/model/architectures/base.py")
        state = {
            "video_backbone.vae.weight": torch.randn(4, 3, 3, 3, 3).contiguous(memory_format=torch.channels_last_3d),
            "video_backbone.vae.conv2d": torch.randn(4, 3, 3, 3).contiguous(memory_format=torch.channels_last),
            "action.weight": torch.randn(4, 4),
            "vlm_backbone.weight": torch.randn(2, 2),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "weights.safetensors")
            base.BaseWAMArchitecture.save_checkpoint(SimpleNamespace(), path, state_dict=state)
            restored = load_file(path)
        self.assertNotIn("vlm_backbone.weight", restored)
        for key, value in restored.items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            self.assertTrue(value.is_contiguous())

    def test_video_preprocess_preserves_bf16_rounding(self):
        frames = [
            Image.fromarray(np.random.default_rng(i).integers(0, 256, (5, 7, 3), dtype=np.uint8)) for i in range(3)
        ]
        for dtype in (torch.float32, torch.bfloat16):
            ref = preprocess.preprocess_video(frames, dtype=dtype, device="cpu")
            with patch.dict(os.environ, OPENWAM_OPT_VIDEO_PREPROCESS="1"):
                actual = preprocess.preprocess_video(frames, dtype=dtype, device="cpu")
            torch.testing.assert_close(actual, ref, rtol=0, atol=0)

    def test_split_attention_all_mask_modes_outputs_and_gradients(self):
        for video_mode in ("bidirectional", "first_frame_causal", "per_frame_causal"):
            for mode in masks.VALID_ATTENTION_MASK_MODES:
                with self.subTest(video=video_mode, cross=mode):
                    mask = masks.build_cross_modal_attention_mask(
                        VideoMask(video_mode),
                        s_video=6,
                        s_action=3,
                        video_tokens_per_frame=2,
                        mode=mode,
                        device=torch.device("cpu"),
                    )
                    mask._openwam_split_plan = attention.split_mask_plan(mask)
                    mask._openwam_split_version = mask._version
                    inputs = [torch.randn(2, 9, 16, dtype=torch.float64, requires_grad=True) for _ in range(3)]
                    headed = [x.reshape(2, 9, 2, 8).transpose(1, 2) for x in inputs]
                    ref = F.scaled_dot_product_attention(*headed, attn_mask=mask).transpose(1, 2).reshape(2, 9, 16)
                    ref_grads = torch.autograd.grad(ref.square().sum(), inputs)
                    with patch.dict(os.environ, OPENWAM_OPT_MOT_SPLIT_ATTN="1"):
                        actual = attention.split_attention(*inputs, mask, 2, unmasked)
                    grads = torch.autograd.grad(actual.square().sum(), inputs)
                    torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-7)
                    for a, b in zip(grads, ref_grads):
                        torch.testing.assert_close(a, b, rtol=1e-6, atol=1e-7)

    def test_split_plan_rejects_mutated_and_prefix_masks(self):
        mask = torch.ones(3, 3, dtype=torch.bool)
        mask._openwam_split_plan = attention.split_mask_plan(mask)
        mask._openwam_split_version = mask._version
        q = torch.randn(1, 3, 8)
        with patch.dict(os.environ, OPENWAM_OPT_MOT_SPLIT_ATTN="1"):
            widened = masks.widen_mask_for_prefix_kv(mask, SimpleNamespace(prefix_kv_len=2, prefix_kv_mask=None))
            self.assertIsNone(
                attention.split_attention(q, torch.randn(1, 5, 8), torch.randn(1, 5, 8), widened, 1, unmasked)
            )
            mask[0, 0] = False
            self.assertIsNone(attention.split_attention(q, q, q, mask, 1, unmasked))

    def test_padding_plan_holes_empty_rows_and_mutation(self):
        keep = torch.tensor([[True, False, True, False], [False, False, False, False], [True, True, True, True]])
        mask = keep[:, None, None, :].expand(3, 1, 7, 4)
        keys = attention._padding_keys(mask, 3, 7, 4)
        self.assertIsNotNone(keys)
        plan = attention._padding_plan(mask, keys)
        self.assertEqual(plan[0].tolist(), [0, 2, 4, 8, 9, 10, 11])
        self.assertEqual(plan[1].tolist(), [0, 2, 3, 7])
        self.assertEqual(plan[2].tolist(), [True, False, True])
        self.assertIs(attention._padding_plan(mask, keys), plan)
        keep[0, 1] = True
        self.assertIsNot(attention._padding_plan(mask, keys), plan)
        self.assertIsNone(attention._padding_keys(mask.clone(), 3, 7, 4))
        self.assertIsNone(attention._padding_keys(mask.float(), 3, 7, 4))

    def test_cpu_padding_never_imports_flash_kernel(self):
        q = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        mask = torch.ones(2, 1, 1, 3, dtype=torch.bool)
        with (
            patch.dict(os.environ, OPENWAM_OPT_FA2_PADDING="1"),
            patch.object(attention, "_varlen_kernel", side_effect=AssertionError),
        ):
            self.assertIsNone(attention.fa2_padding(q, q, q, mask, 2))

    def test_fa2_packing_with_mock_kernel_outputs_and_gradients(self):
        class DispatchTensor(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        # Exercise the actual packing function while replacing only the GPU kernel.
        inputs = [
            torch.randn(3, length, 16, dtype=torch.bfloat16).as_subclass(DispatchTensor).requires_grad_()
            for length in (5, 4, 4)
        ]
        keep = torch.tensor([[True, False, True, False], [False, False, False, False], [True, True, True, True]])
        mask = keep[:, None, None, :].expand(3, 1, 5, 4)

        def kernel(q, k, v, cu_q, cu_k, max_q, max_k, dropout_p, causal):
            self.assertEqual((max_q, max_k, dropout_p, causal), (5, 4, 0.0, False))
            self.assertEqual(cu_q.dtype, torch.int32)
            self.assertEqual(cu_k.dtype, torch.int32)
            outputs = []
            for row in range(3):
                qi = q[cu_q[row] : cu_q[row + 1]].transpose(0, 1).unsqueeze(0)
                ki = k[cu_k[row] : cu_k[row + 1]].transpose(0, 1).unsqueeze(0)
                vi = v[cu_k[row] : cu_k[row + 1]].transpose(0, 1).unsqueeze(0)
                outputs.append(
                    F.scaled_dot_product_attention(qi.float(), ki.float(), vi.float()).to(q.dtype)[0].transpose(0, 1)
                )
            return torch.cat(outputs)

        headed = [x.reshape(3, x.shape[1], 2, 8).transpose(1, 2) for x in inputs]
        reference_keep = keep.clone()
        reference_keep[1, 0] = True
        ref = F.scaled_dot_product_attention(*[x.float() for x in headed], attn_mask=reference_keep[:, None, None])
        ref = ref.to(inputs[0].dtype).transpose(1, 2).reshape(3, 5, 16)
        ref = ref.masked_fill((~keep.any(-1)).view(3, 1, 1), 0)
        with (
            patch.dict(os.environ, OPENWAM_OPT_FA2_PADDING="1"),
            patch.object(attention, "_varlen_kernel", return_value=kernel),
        ):
            actual = attention.fa2_padding(*inputs, mask, 2)
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        for a, b in zip(
            torch.autograd.grad(actual.float().sum(), inputs), torch.autograd.grad(ref.float().sum(), inputs)
        ):
            torch.testing.assert_close(a, b, rtol=0.01, atol=0.01)

    def test_canonical_layout_cache_and_singleton_storage_offset(self):
        tensor = torch.randn(2, 3, 4, 5, 6).contiguous(memory_format=torch.channels_last_3d)
        sliced = tensor[1:2, :, 1:2]
        restrided = canonicalize(sliced, singleton=True)
        self.assertEqual(restrided.data_ptr(), sliced.data_ptr())
        self.assertEqual(restrided.storage_offset(), sliced.storage_offset())
        torch.testing.assert_close(restrided, sliced, rtol=0, atol=0)
        weight = torch.randn(4, 3, 3, 3, 3)
        cache = FrozenWeightCache(limit=2)
        first = cache.convert(weight, False)
        self.assertIs(cache.convert(weight, False), first)
        weight.add_(1)
        next_value = cache.convert(weight, False)
        self.assertIsNot(next_value, first)
        torch.testing.assert_close(next_value, weight, rtol=0, atol=0)

    def test_pointwise_rms_and_rope_outputs_and_gradients(self):
        for use_fp32 in (False, True):
            x = torch.randn(2, 5, 3, 8, dtype=torch.float64, requires_grad=True)
            freqs = torch.polar(torch.ones(5, 1, 4, dtype=torch.float64), torch.randn(5, 1, 4, dtype=torch.float64))
            ref = torch.view_as_real(torch.view_as_complex(x.reshape(2, 5, 3, 4, 2)) * freqs).flatten(-2)
            with patch.dict(os.environ, OPENWAM_OPT_ROPE_FP32=str(int(use_fp32))):
                actual = pointwise.rotate(x, freqs)
            tolerance = 1e-6 if use_fp32 else 1e-12
            torch.testing.assert_close(actual, ref, rtol=tolerance, atol=tolerance)
            a = torch.autograd.grad(actual.square().sum(), x, retain_graph=True)[0]
            b = torch.autograd.grad(ref.square().sum(), x)[0]
            torch.testing.assert_close(a, b, rtol=tolerance, atol=tolerance)
        x = torch.randn(2, 4, 16, dtype=torch.bfloat16, requires_grad=True)
        weight = torch.randn(16, dtype=torch.bfloat16, requires_grad=True)
        ref = (x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * weight
        actual = pointwise.rms_norm(x, weight, 1e-6)
        torch.testing.assert_close(actual, ref, rtol=0, atol=0)
        for a, b in zip(
            torch.autograd.grad(actual.float().sum(), (x, weight)), torch.autograd.grad(ref.float().sum(), (x, weight))
        ):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_text_cache_eviction_order_invalidation_and_training_bypass(self):
        model = torch.nn.Linear(2, 2).eval().requires_grad_(False)
        tokenizer = object()
        calls = []

        def encode(prompts):
            calls.append(list(prompts))
            context = torch.tensor([[[float(ord(p)), 2.0]] for p in prompts])
            return context, torch.ones(len(prompts), dtype=torch.long)

        def run(prompts):
            return cached_text(prompts, tokenizer=tokenizer, text_encoder=model, device="cpu", encode=encode)

        with patch.dict(os.environ, OPENWAM_OPT_TEXT_CACHE="1", OPENWAM_TEXT_CACHE_SIZE="2"):
            out, _ = run(["a", "a", "b", "c"])
            self.assertEqual(calls, [["a", "b", "c"]])
            self.assertEqual(out[:, 0, 0].tolist(), [97.0, 97.0, 98.0, 99.0])
            run(["b", "c"])
            self.assertEqual(len(calls), 1)
            with torch.no_grad():
                model.weight.add_(1)
            run(["b"])
            self.assertEqual(len(calls), 2)
            model.train()
            run(["b", "b"])
            self.assertEqual(calls[-1], ["b", "b"])
            self.assertEqual(len(model._openwam_text_cache), 1)

    def test_optional_hipdnn_reference_streaming_and_autograd(self):
        x = torch.randn(2, 3, 3, 4, 5, dtype=torch.float64, requires_grad=True)
        weight = torch.randn(4, 3, 3, 3, 3, dtype=torch.float64, requires_grad=True)
        bias = torch.randn(4, dtype=torch.float64, requires_grad=True)
        for cache_t in (0, 1, 2):
            cache = torch.randn(2, 3, cache_t, 4, 5, dtype=torch.float64) if cache_t else None
            joined = x if cache is None else torch.cat([cache, x], dim=2)
            ref = F.conv3d(F.pad(joined, (1, 1, 1, 1, 2 - cache_t, 0)), weight, bias)
            with patch.dict(os.environ, OPENWAM_OPT_VAE_HIPDNN="1"):
                actual = fused_concat_conv_bias(x, cache, weight, bias)
            torch.testing.assert_close(actual, ref)
            grads = torch.autograd.grad(actual.square().mean(), (x, weight, bias), retain_graph=True)
            ref_grads = torch.autograd.grad(ref.square().mean(), (x, weight, bias))
            for a, b in zip(grads, ref_grads):
                torch.testing.assert_close(a, b)


if __name__ == "__main__":
    unittest.main()
