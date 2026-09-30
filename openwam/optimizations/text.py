"""Bounded prompt caching restricted to frozen, deterministic text encoders."""

import logging
import os
from collections import OrderedDict

import torch

from openwam.optimizations import enabled

logger = logging.getLogger(__name__)


def cached_text(prompts, *, tokenizer, text_encoder, device, encode):
    if not enabled("OPENWAM_OPT_TEXT_CACHE"):
        return encode(prompts)
    params = tuple(text_encoder.parameters())
    if any(module.training for module in text_encoder.modules()) or any(p.requires_grad for p in params):
        if not getattr(text_encoder, "_openwam_cache_skip_warned", False):
            logger.warning("[optimizations] text cache skipped: encoder must be frozen and in eval mode")
            text_encoder._openwam_cache_skip_warned = True
        return encode(prompts)
    capacity = int(os.environ.get("OPENWAM_TEXT_CACHE_SIZE", "32"))
    if capacity <= 0:
        raise ValueError("OPENWAM_TEXT_CACHE_SIZE must be positive")
    signature = (id(tokenizer), str(device), tuple((p.data_ptr(), p._version, p.dtype, p.device) for p in params))
    if signature != getattr(text_encoder, "_openwam_text_cache_signature", None):
        text_encoder._openwam_text_cache_signature = signature
        text_encoder._openwam_text_cache = OrderedDict()
    cache = text_encoder._openwam_text_cache
    unique = list(dict.fromkeys(prompts))
    # Keep this batch's references separately so eviction cannot invalidate it.
    values = {prompt: cache[prompt] for prompt in unique if prompt in cache}
    missing = [prompt for prompt in unique if prompt not in values]
    if missing:
        with torch.no_grad():
            context, lengths = encode(missing)
        for index, prompt in enumerate(missing):
            values[prompt] = (context[index].detach().clone(), lengths[index].detach().clone())
    for prompt in unique:
        cache[prompt] = values[prompt]
        cache.move_to_end(prompt)
    while len(cache) > capacity:
        cache.popitem(last=False)
    return torch.stack([values[prompt][0] for prompt in prompts]), torch.stack(
        [values[prompt][1] for prompt in prompts]
    )
