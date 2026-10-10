"""One real batch of lookahead encoding during ZeRO-2 parameter AllGather."""

import functools
import inspect
import logging
import types
from dataclasses import dataclass

import torch
import torch.distributed as dist

from openwam.optimizations.zero import validate_zero

logger = logging.getLogger(__name__)


@dataclass
class PreparedInputs:
    values: dict


def _record_stream(value, stream):
    if isinstance(value, torch.Tensor):
        if value.is_cuda:
            value.record_stream(stream)
    elif isinstance(value, dict):
        for item in value.values():
            _record_stream(item, stream)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _record_stream(item, stream)


class OnlineEncoder:
    """Instance-local gather replacement; no threads, replay, or disk cache.

    The producer event precedes communication. Encoding waits for the producer,
    while the training stream waits for communication only after encoding has
    been enqueued. Waiting for encoded inputs happens at the next consumer.
    """

    def __init__(self, architecture, optimizer, *, num_workers, gradient_accumulation_steps):
        from openwam.optimizations.text import _native_t5, frozen_eval

        validate_zero(optimizer)
        bb = architecture.video_backbone
        if (type(bb).__name__ != "Wan22Ti2v" or bb.video_encoder is not None
                or type(bb.vae).__name__ != "WanVideoVAE38" or not _native_t5(bb.text_encoder)):
            raise ValueError("Online encoding currently requires native Wan22Ti2v VAE38 and T5")
        if not all(frozen_eval(module) for module in (bb.vae, bb.text_encoder)):
            raise ValueError("Online encoding requires frozen VAE/T5 in eval mode")
        frozen_ids = {id(p) for module in (bb.vae, bb.text_encoder) for p in module.parameters()}
        if any(id(p) in frozen_ids for group in optimizer.bit16_groups for p in group):
            raise ValueError("Frozen encoders must not belong to the ZeRO optimizer/AllGather groups")
        if gradient_accumulation_steps != 1 or num_workers < 1:
            raise ValueError("Online encoding requires gradient accumulation=1 and DataLoader workers>0")
        # Worker-side random transforms keep their own RNG/order. Native TI2V
        # preprocessing uses the VAE mean and no random draws; training noise
        # and timesteps remain in compute_loss on the original training stream.
        self.architecture = architecture
        self.optimizer = optimizer
        self.device = architecture.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.pending = None
        self.next_batch = None
        self.original_step = optimizer.step
        function = self.original_step.__func__
        if "all_gather_dp_groups" not in function.__code__.co_names:
            raise RuntimeError("Unsupported DeepSpeed step: no direct all_gather_dp_groups call")
        self.original_gather = function.__globals__["all_gather_dp_groups"]
        expected = ("groups_flat", "partitioned_param_groups", "dp_process_group",
                    "start_alignment_factor", "allgather_bucket_size")
        if tuple(inspect.signature(self.original_gather).parameters) != expected:
            raise RuntimeError("Unsupported DeepSpeed AllGather interface")
        namespace = dict(function.__globals__)
        namespace["all_gather_dp_groups"] = self._gather
        replacement = types.FunctionType(function.__code__, namespace, function.__name__,
                                         function.__defaults__, function.__closure__)
        replacement.__kwdefaults__ = function.__kwdefaults__
        functools.update_wrapper(replacement, function)
        self.step = types.MethodType(replacement, optimizer)

    def __enter__(self):
        self.optimizer.step = self.step
        logger.info("[optimizations] online next-batch encoding during ZeRO-2 AllGather")
        return self

    def __exit__(self, *exc):
        self.optimizer.step = self.original_step
        self.next_batch = None
        try:
            self.stream.synchronize()
        finally:
            self.pending = None

    def _gather(self, groups_flat, partitioned_param_groups, dp_process_group,
                start_alignment_factor, allgather_bucket_size):
        if self.next_batch is None:
            return self.original_gather(groups_flat, partitioned_param_groups, dp_process_group,
                                        start_alignment_factor, allgather_bucket_size)
        take, self.next_batch = self.next_batch, None
        # Fetch before launching collectives: an exhausted/broken iterator must
        # not leave this rank with outstanding communication work.
        batch = take()
        producer = torch.cuda.current_stream(self.device)
        ready = torch.cuda.Event()
        ready.record(producer)  # Includes any DataLoader device transfer.
        works = []
        try:
            for flat, parts, group in zip(groups_flat, partitioned_param_groups, dp_process_group):
                if dist.get_world_size(group) > 1:
                    works.append(dist.all_gather_into_tensor(
                        flat, parts[dist.get_rank(group)], group=group, async_op=True))
            with torch.cuda.stream(self.stream), torch.no_grad():
                self.stream.wait_event(ready)
                _record_stream(batch, self.stream)
                values = self.architecture.prepare_inputs(batch if isinstance(batch, list) else [batch])
                encoded = torch.cuda.Event()
                encoded.record(self.stream)
                self.pending = (PreparedInputs(values), encoded, batch)
        finally:
            # NCCL/RCCL Work.wait creates a consumer-stream dependency. This
            # wait must come AFTER the independent encoding has been enqueued.
            for work in works:
                work.wait()

    def batches(self, batches, *, global_step, max_steps, save_steps):
        iterator = iter(batches)
        try:
            for index in range(len(batches)):
                if self.pending is None:
                    batch = next(iterator)
                else:
                    batch, ready, raw = self.pending
                    producer = torch.cuda.current_stream(self.device)
                    producer.wait_event(ready)
                    _record_stream(batch.values, producer)
                    self.pending = None
                    del raw
                step = global_step + index + 1
                eligible = (index + 1 < len(batches) and (not max_steps or step < max_steps)
                            and (not save_steps or step % save_steps != 0))
                self.next_batch = (lambda: next(iterator)) if eligible else None
                yield batch
                # Overflow can skip optimizer.step/AllGather. In that case no
                # next batch was consumed, so the next iteration reads it here.
                self.next_batch = None
        finally:
            self.next_batch = None
