"""Opt-in layout experiments, executed inside the opaque concat custom op."""

import os
import threading
import weakref
from collections import OrderedDict

import torch


def layout_flag(name: str) -> bool:
    return os.environ.get(name, "0").strip().lower() in {"1", "true", "yes", "on"}


def canonicalize(tensor: torch.Tensor, singleton: bool = False) -> torch.Tensor:
    n, c, t, h, w = tensor.shape
    expected = (c * t * h * w, 1, h * w * c, w * c, c)
    if tuple(tensor.stride()) == expected:
        return tensor
    if singleton and all(
        size == 1 or actual == wanted for size, actual, wanted in zip(tensor.shape, tensor.stride(), expected)
    ):
        # Preserve storage_offset: temporal slices need not start at storage zero.
        return tensor.as_strided(tensor.shape, expected)
    out = torch.empty(tensor.shape, dtype=tensor.dtype, device=tensor.device, memory_format=torch.channels_last_3d)
    out.copy_(tensor)
    return out


class FrozenWeightCache:
    """Bounded cache of frozen, versioned weights; one entry per source/stream.

    Inference tensors without a version counter bypass caching. Standard in-place
    updates invalidate entries; modifying weights via .data is unsupported.
    Stream-local entries avoid consuming an unfinished conversion on another stream.
    """

    def __init__(self, limit: int = 256):
        self.limit = limit
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def convert(self, weight: torch.Tensor, singleton: bool) -> torch.Tensor:
        if weight.requires_grad:
            return canonicalize(weight, singleton)
        try:
            version = weight._version
        except RuntimeError:
            return canonicalize(weight, singleton)
        stream = torch.cuda.current_stream(weight.device).cuda_stream if weight.is_cuda else None
        key = (id(weight), stream)
        signature = (
            version,
            weight.data_ptr(),
            tuple(weight.shape),
            tuple(weight.stride()),
            weight.dtype,
            weight.device,
            singleton,
        )
        with self.lock:
            entry = self.entries.get(key)
            if entry is not None and entry[0]() is weight and entry[1] == signature:
                self.entries.move_to_end(key)
                return entry[2]
            with torch.no_grad():
                converted = canonicalize(weight, singleton)
            # No copy means there is no conversion cost to cache (and no need
            # to retain a view of the original storage).
            if converted.data_ptr() == weight.data_ptr():
                self.entries.pop(key, None)
                return converted
            self.entries[key] = (weakref.ref(weight), signature, converted)
            self.entries.move_to_end(key)
            while len(self.entries) > self.limit:
                self.entries.popitem(last=False)
            return converted


WEIGHT_CACHE = FrozenWeightCache()
