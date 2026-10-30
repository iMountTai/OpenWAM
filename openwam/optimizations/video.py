"""Worker-side pixel packing for the native Wan training input path."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image


@dataclass
class Uint8Video:
    """CPU THWC pixels; keep them on the host until Wan submits the VAE input.

    Deliberately has no ``to`` method: Accelerate must not move these pixels
    ahead of the asynchronous gather. DataLoader pins them through this hook.
    Lazy PIL access preserves size/first-frame conditioning without retaining
    a second copy of every frame in the worker queue.
    """

    pixels: np.ndarray | torch.Tensor

    def __len__(self):
        return len(self.pixels)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return Image.fromarray(np.asarray(self.pixels[index]))

    def pin_memory(self):
        self.pixels = torch.as_tensor(self.pixels, device="cpu").pin_memory()
        return self


def collate_uint8_video(samples, *, collate=list):
    """Pack after dataset resize/augmentation; leave stored latents untouched."""
    result = []
    for sample in samples:
        frames = sample.get("video")
        if (sample.get("video_latents") is None and isinstance(frames, list) and frames
                and all(isinstance(frame, Image.Image) and frame.mode in ("RGB", "RGBA")
                        and frame.mode == frames[0].mode and frame.size == frames[0].size
                        for frame in frames)):
            sample = dict(sample)
            pixels = np.stack([np.asarray(frame, dtype=np.uint8) for frame in frames])
            # NumPy retains the original PIL path's byte-based IPC; avoid
            # placing full videos in multiprocessing's Tensor shared memory.
            sample["video"] = Uint8Video(pixels)
        result.append(sample)
    return collate(result)
