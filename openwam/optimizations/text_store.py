"""Offline frozen T5 contexts, generated alongside the RoboTwin VAE store.

OPENWAM_OPT_TEXT_CACHE selects this store when OPENWAM_TEXT_CACHE_DIR is set,
or when OPENWAM_VAE_LATENT_CACHE_DIR is set (its ``text`` subdirectory). Without
either directory, the original bounded online text cache remains available.
Only valid token rows are saved, independently of TEXT_TRIM; readers restore
zero padding to the batch maximum or the full 512-token length. Version 1
stores must be removed and regenerated before using this format.
"""

import hashlib
import json
import logging
import os
import uuid
from collections import OrderedDict

import numpy as np
import torch

from openwam.optimizations import enabled
from openwam.optimizations.text import _native_t5, frozen_eval

logger = logging.getLogger(__name__)
FULL_LENGTH = 512
VERSION = 2
DTYPES = {"bfloat16": (torch.bfloat16, np.dtype("int16")),
          "float16": (torch.float16, np.dtype("float16")),
          "float32": (torch.float32, np.dtype("float32"))}


def directory():
    root = os.environ.get("OPENWAM_TEXT_CACHE_DIR", "").strip()
    if not root:
        latent_root = os.environ.get("OPENWAM_VAE_LATENT_CACHE_DIR", "").strip()
        root = os.path.join(latent_root, "text") if latent_root else None
    return root


def requested():
    return enabled("OPENWAM_OPT_TEXT_CACHE") and directory() is not None


def training_prompts(dataset):
    from openwam.dataloader.robotwin import _training_prompts
    from openwam.optimizations.latent_cache import episode_starts, readers

    prompts = set()
    for reader in readers(dataset):
        for index, path in enumerate(reader._episode_files):
            if episode_starts(reader, index):
                prompts.update(_training_prompts(reader._instructions, path, reader.task_name))
    return sorted(prompts)


def validate_backbone(backbone):
    encoder = getattr(backbone, "text_encoder", None)
    if encoder is None or not frozen_eval(encoder) or not _native_t5(encoder):
        raise ValueError("Offline text caching requires the native frozen Wan T5 in eval mode")


def signature(backbone):
    """Hash exact weight bytes and tokenizer behavior, without FP64 weight copies."""
    validate_backbone(backbone)
    encoder = backbone.text_encoder
    digest = hashlib.sha256()
    for name, value in sorted(encoder.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(f"{name}|{tuple(value.shape)}|{value.dtype}".encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    settings = ("dim", "dim_attn", "dim_ffn", "num_heads", "head_dim", "num_layers",
                "num_buckets", "shared_pos", "eps", "bidirectional", "max_dist")
    digest.update(json.dumps([
        (name, type(module).__name__, {key: getattr(module, key) for key in settings if hasattr(module, key)})
        for name, module in encoder.named_modules()
    ], sort_keys=True).encode())
    tokenizer = backbone._tokenizer
    native = tokenizer.tokenizer
    backend = getattr(native, "backend_tokenizer", None)
    sentencepiece = getattr(native, "sp_model", None)
    if backend is not None:
        # HF calls mutate these transient backend fields. Our encoder supplies
        # fixed padding/truncation arguments, so they are not tokenizer identity.
        state = json.loads(backend.to_str())
        state.pop("padding", None)
        state.pop("truncation", None)
        tokenizer_state = json.dumps(state, sort_keys=True, ensure_ascii=False).encode()
    elif sentencepiece is not None:
        tokenizer_state = sentencepiece.serialized_model_proto()
    else:
        raise ValueError("Offline text caching requires a serializable HF/SentencePiece tokenizer")
    token_digest = hashlib.sha256(tokenizer_state)
    token_digest.update(json.dumps({
        "vocab": native.get_vocab(), "special_tokens": native.special_tokens_map,
        "added_tokens": native.get_added_vocab(), "clean": tokenizer.clean,
        "seq_len": tokenizer.seq_len, "padding_side": native.padding_side,
        "truncation_side": native.truncation_side,
    }, sort_keys=True, ensure_ascii=False).encode())
    dtype = str(next(encoder.parameters()).dtype).removeprefix("torch.")
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported T5 cache dtype: {dtype}")
    return {"version": VERSION, "encoder": digest.hexdigest(), "tokenizer": token_digest.hexdigest(),
            "dtype": dtype, "length": FULL_LENGTH, "dim": int(encoder.dim)}


def _atomic(path, write):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        with open(temporary, "wb") as handle:
            write(handle)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _json(path, value):
    _atomic(path, lambda handle: handle.write(json.dumps(value, sort_keys=True).encode()))


def _key(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()


def _path(root, prompt):
    key = _key(prompt)
    return os.path.join(root, key[:2], key + ".npy")


def _validate_version(root, meta):
    if meta.get("version") != VERSION:
        raise ValueError(f"Unsupported T5 text cache version: remove the text cache and rerun step 1 ({root})")


def _entry(root, prompt, meta):
    _validate_version(root, meta)
    path = _path(root, prompt)
    with open(path + ".json", encoding="utf-8") as handle:
        record = json.load(handle)
    length = record["length"]
    if (record.get("prompt") != prompt or type(length) is not int or not 0 <= length <= FULL_LENGTH):
        raise ValueError(f"Invalid T5 text cache entry: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (length, meta["dim"]) or array.dtype != DTYPES[meta["dtype"]][1]:
        raise ValueError(f"T5 text cache shape/dtype mismatch: {path}")
    return length, array


def _finished(root, prompt, meta):
    try:
        _entry(root, prompt, meta)
        return True
    except (OSError, ValueError, KeyError, EOFError):
        return False


def generate(dataset, backbone, *, device, batch_size, rank=0, world=1):
    """Ranks encode disjoint unique prompts; complete files survive interruptions."""
    from openwam.model.video_backbone.wan.encode import _encode_text

    root = directory()
    if root is None:
        raise ValueError("Set OPENWAM_TEXT_CACHE_DIR or OPENWAM_VAE_LATENT_CACHE_DIR")
    meta = signature(backbone)
    meta_path = os.path.join(root, "meta.json")
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            existing = json.load(handle).get("signature", {})
        _validate_version(root, existing)
        if existing != meta:
            raise ValueError(f"T5 text cache settings/weights differ: use a new directory ({root})")
    else:
        _json(meta_path, {"signature": meta, "generation": {"t5_prefix": enabled("OPENWAM_OPT_T5_PREFIX")}})
    prompts = training_prompts(dataset)
    mine = [prompt for prompt in prompts[rank::world] if not _finished(root, prompt, meta)]
    logger.info("[text store] rank %d/%d: %d/%d unique prompts to encode (%d total)",
                rank, world, len(mine), len(prompts[rank::world]), len(prompts))
    if batch_size < 1:
        raise ValueError("T5 cache batch size must be positive")
    if not mine:
        return
    encoder = backbone.text_encoder
    encoder.to(device=device)
    with torch.no_grad():
        for start in range(0, len(mine), batch_size):
            batch = mine[start:start + batch_size]
            context, lengths = _encode_text(batch, tokenizer=backbone._tokenizer,
                                           text_encoder=encoder, device=device)
            context, lengths = context.cpu().contiguous(), lengths.cpu().tolist()
            if context.shape != (len(batch), FULL_LENGTH, meta["dim"]):
                raise ValueError(f"Unexpected T5 output shape: {context.shape}")
            for prompt, value, length in zip(batch, context, lengths):
                path = _path(root, prompt)
                value = value[:length].contiguous()
                array = value.view(torch.int16).numpy() if value.dtype == torch.bfloat16 else value.numpy()
                _atomic(path, lambda handle, array=array: np.save(handle, array, allow_pickle=False))
                # Written last: an interrupted tensor write is never a complete entry.
                _json(path + ".json", {"prompt": prompt, "length": length})
            logger.info("[text store] rank %d: %d/%d prompts", rank, min(start + batch_size, len(mine)), len(mine))


class PromptContexts:
    """CPU memory maps; GPU holds only this batch's padded context, never a prompt LRU."""

    def __init__(self, root, meta, prompts):
        self.root, self.meta, self.lengths, self.maps = root, meta, {}, OrderedDict()
        for prompt in prompts:
            try:
                self.lengths[prompt] = _entry(root, prompt, meta)[0]
            except (OSError, ValueError, KeyError, EOFError) as exc:
                raise ValueError(f"T5 text cache missing/incomplete for {prompt!r}; rerun step 1 ({root})") from exc

    def __call__(self, prompts, device):
        missing = [prompt for prompt in prompts if prompt not in self.lengths]
        if missing:
            raise ValueError(f"Prompt absent from offline T5 cache: {missing[0]!r}; rerun step 1")
        lengths = torch.tensor([self.lengths[prompt] for prompt in prompts], dtype=torch.long)
        stop = max(1, int(lengths.max())) if enabled("OPENWAM_OPT_TEXT_TRIM") else FULL_LENGTH
        dtype = DTYPES[self.meta["dtype"]][0]
        target = torch.device(device)
        context = torch.zeros((len(prompts), stop, self.meta["dim"]), dtype=dtype,
                              pin_memory=target.type == "cuda")
        for index, prompt in enumerate(prompts):
            array = self.maps.pop(prompt, None)
            if array is None:
                array = _entry(self.root, prompt, self.meta)[1]
            self.maps[prompt] = array
            while len(self.maps) > 64:
                self.maps.popitem(last=False)
            length = self.lengths[prompt]
            value = torch.from_numpy(np.array(array, copy=True))
            context[index, :length].copy_(value.view(torch.bfloat16) if dtype == torch.bfloat16 else value)
        # Lengths intentionally stay on CPU, as in the padding-plan path.
        return context.to(device=target, non_blocking=True), lengths


def offload(backbone):
    validate_backbone(backbone)
    backbone.text_encoder.to(device="cpu")
    # The older online LRU can otherwise retain GPU contexts after a mode switch.
    for name in ("_openwam_text_cache", "_openwam_text_cache_signature"):
        backbone.text_encoder.__dict__.pop(name, None)


def attach(dataset, backbone):
    """Validate every possible training prompt before relying on CPU-only T5 weights."""
    root = directory()
    path = os.path.join(root, "meta.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"T5 text cache has no meta.json; run step 1 with OPENWAM_OPT_TEXT_CACHE=True ({root})")
    with open(path, encoding="utf-8") as handle:
        meta = json.load(handle)["signature"]
    _validate_version(root, meta)
    if meta != signature(backbone):
        raise ValueError(f"T5 text cache weights/tokenizer/settings mismatch ({root})")
    prompts = training_prompts(dataset)
    contexts = PromptContexts(root, meta, prompts)
    offload(backbone)
    backbone.text_encoder._openwam_text_store = contexts
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("[optimizations] offline T5 cache %s: %d prompts; T5 weights on CPU", root, len(prompts))
