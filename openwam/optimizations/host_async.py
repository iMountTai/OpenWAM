"""Remove per-step host synchronizations from the training loop.

The default loop reads loss/grad-norm with .item() every step, DeepSpeed's
throughput timer calls cuda.synchronize() inside engine.step, and a few small
CPU tensors are uploaded with blocking copies. Each one drains the GPU queue,
so the host must relaunch the next work from an idle device. Here metrics are
reduced on the device, copied to pinned memory without blocking and reported
one step late; small uploads use pinned non-blocking copies.
"""

import torch

from openwam.optimizations import enabled


def active():
    return enabled("OPENWAM_OPT_HOST_ASYNC")


def configure_deepspeed(config):
    # The throughput timer is only used for DeepSpeed's own steps_per_print
    # report; when enabled it synchronizes the device at every engine.step.
    config["timers"] = {"throughput": {"enabled": False, "synchronized": False}}


def upload(tensor, device):
    """Pinned non-blocking H2D for small host tensors; others keep .to()."""
    target = torch.device(device)
    if tensor.device.type == "cpu" and target.type == "cuda":
        return tensor.pin_memory().to(device=target, non_blocking=True)
    return tensor.to(device=target)


def device_scalar(value, device):
    """Grad norm as a device tensor without reading a device value on the host."""
    if isinstance(value, torch.Tensor):
        return value.detach().to(device=device, dtype=torch.float32)
    return torch.tensor(float(value), device=device, dtype=torch.float32)


class LaggedMetrics:
    """Same cross-rank mean as reduce_step_metrics, delivered one step later."""

    def __init__(self, accelerator):
        self.accelerator = accelerator
        self.pending = None

    def push(self, losses, grad_norm, info):
        device = losses["total"].device
        values = torch.stack([
            torch.as_tensor(value, device=device).detach().float().reshape(())
            for value in (losses["total"], losses["video"], losses["action"], grad_norm)
        ])
        if self.accelerator is not None and self.accelerator.num_processes > 1:
            values = self.accelerator.gather(values.reshape(1, -1)).mean(dim=0)
        host = torch.empty(values.shape, dtype=values.dtype, pin_memory=True)
        host.copy_(values, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        ready = self.flush()
        self.pending = (event, host, info)
        return ready

    def flush(self):
        if self.pending is None:
            return []
        event, host, info = self.pending
        self.pending = None
        event.synchronize()
        loss, video, action, grad_norm = host.tolist()
        return [(info, {"loss_total": loss, "loss_video": video, "loss_action": action, "grad_norm": grad_norm})]


def rope_freqs(dit, grid, device, build):
    """Device-resident RoPE table per latent grid; the CPU table is immutable."""
    if not active() or torch.compiler.is_compiling():
        return build().to(device)
    cache = dit.__dict__.setdefault("_openwam_rope_cache", {})
    key = (tuple(int(size) for size in grid), str(device))
    value = cache.get(key)
    if value is None:
        value = cache[key] = upload(build(), device)
    return value
