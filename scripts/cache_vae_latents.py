"""Step 1 of the VAE latent store: encode every RoboTwin training window once.

Use the same dataloader/model/optimizations overrides as the training run,
plus the store directory (node-local disk is preferred)::

    torchrun --nproc_per_node=8 scripts/cache_vae_latents.py optimizations=hcu dataloader=robotwin ... \\
        training.finetune_ckpt_path=... optimizations.vae_latent_cache_dir=/path/to/latents

Each rank encodes a disjoint set of episodes and writes one ``.npy`` per
episode (atomic rename), so an interrupted run resumes by skipping finished
files. Step 2 is the normal training command with
optimizations.vae_latent_cache=true and the same vae_latent_cache_dir.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


class _Windows(torch.utils.data.Dataset):
    """Decoded frames of (reader, episode, start) windows, in episode order."""

    def __init__(self, readers, items):
        self.readers, self.items = readers, items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        reader, episode, start = self.items[index]
        return self.items[index], self.readers[reader]._build_sample(episode, start)["video"]


def _write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.tmp{os.environ.get('RANK', '0')}"
    with open(temporary, "wb") as handle:
        np.save(handle, torch.stack(rows).view(torch.int16).numpy())
    os.replace(temporary, path)


def _finished(path, rows):
    if not os.path.isfile(path):
        return False
    try:
        return np.load(path, mmap_mode="r").shape[0] == rows
    except ValueError:
        return False


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "configs"), config_name="train")
def main(cfg: DictConfig) -> None:
    sys.path.insert(0, str(PROJECT_ROOT))
    from openwam.dataloader.registry import build_dataset
    from openwam.model.video_backbone.wan import encode as wan_encode
    from openwam.optimizations import configure_backends, create_trainer, prepare_runtime_constants
    from openwam.optimizations import latent_cache as store

    configure_backends(cfg.get("optimizations"))
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    root = store.directory()

    dataset = build_dataset(cfg.dataloader, split="train")
    readers = store.readers(dataset)
    for reader in readers:
        store.check_windows(reader)
    # The training constructor gives the VAE its training dtype, layout and compile settings.
    trainer = create_trainer(cfg, accelerator=None, dataset=dataset)
    architecture = trainer.architecture
    store.validate_backbone(architecture)
    backbone = architecture.video_backbone
    signature = store.store_signature(dataset, store.vae_fingerprint(backbone.vae))
    device = torch.device("cuda", local_rank)
    architecture.set_dtype_device(architecture.dtype, device)
    backbone.vae.to(device=device)
    prepare_runtime_constants(architecture)

    meta_path = os.path.join(root, store.META)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding="utf-8") as handle:
            if json.load(handle) != signature:
                raise ValueError(f"{meta_path} was written for different data/VAE settings; use a new directory")
    elif rank == 0:
        os.makedirs(root, exist_ok=True)
        with open(f"{meta_path}.tmp", "w", encoding="utf-8") as handle:
            json.dump(signature, handle)
        os.replace(f"{meta_path}.tmp", meta_path)

    episodes = [(r, e) for r, reader in enumerate(readers) for e in range(len(reader._episode_files))
                if store.episode_starts(reader, e)]
    mine = [(r, e) for r, e in episodes[rank::world]
            if not _finished(store.episode_path(root, readers[r], e), len(store.episode_starts(readers[r], e)))]
    items = [(r, e, start) for r, e in mine for start in store.episode_starts(readers[r], e)]
    logger.info("[latent store] rank %d/%d: %d of %d episodes to encode, %d windows",
                rank, world, len(mine), len(episodes[rank::world]), len(items))
    loader = torch.utils.data.DataLoader(
        _Windows(readers, items), batch_size=int(cfg.training.batch_size), shuffle=False,
        num_workers=int(cfg.training.dataset_num_workers), collate_fn=list,
    )

    current, rows, done, batches, started = None, [], 0, 0, time.monotonic()
    with torch.no_grad():
        for batch in loader:
            # Same calls as WanBase.preprocess_input_for_train's pixel path.
            pixels = torch.cat([wan_encode.preprocess_video(frames, encoder=None, dtype=backbone.dtype,
                                                            device=backbone.device) for _, frames in batch])
            latents = wan_encode.encode_video(pixels, vae=backbone.vae, encoder=None)
            latents = latents.to(dtype=backbone.dtype).cpu()
            for ((r, e, _), _), latent in zip(batch, latents):
                if (r, e) != current:
                    if current is not None:
                        _write(store.episode_path(root, readers[current[0]], current[1]), rows)
                    current, rows = (r, e), []
                rows.append(latent.contiguous())
            done += len(batch)
            batches += 1
            if rank == 0 and batches % 50 == 0:
                rate = done / (time.monotonic() - started)
                logger.info("[latent store] rank 0: %d/%d windows, %.1f windows/s", done, len(items), rate)
    if current is not None:
        _write(store.episode_path(root, readers[current[0]], current[1]), rows)
    logger.info("[latent store] rank %d finished %d windows in %.0f s", rank, done, time.monotonic() - started)


if __name__ == "__main__":
    main()
