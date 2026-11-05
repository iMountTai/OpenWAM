"""Deferred ZeRO-2 parameter gather, waited per MoT layer by the next forward.

DeepSpeed ends every step with one AllGather of the updated bit16 partitions,
and the training stream waits for all of it before anything else. ONLINE_ENCODE
hides it behind next-batch VAE/T5 encoding; with offline latent and text caches
there is nothing left to encode, so the whole collective is exposed.

With OPENWAM_OPT_DEFERRED_GATHER each owner broadcasts its partition in pieces
instead, ordered by first use in the next forward: parameters outside the MoT
layers, then layer 0, 1, ... A piece lies inside one owner's partition, so it is
one broadcast from that owner. Pieces are split where the first-using layer
changes, so the order does not depend on the flat buffer layout (action
parameters precede video parameters there). Nothing waits at the step end. The
training stream waits for the prologue pieces before the next compute_loss and
for layer i's pieces right before layer i. Every piece is waited before the MoT
loop returns, hence before backward and the next optimizer update. The bytes
written are those of the AllGather; only the point of the wait moves.

The broadcasts use a dedicated process group with the same ranks, so small
collectives issued between steps (HOST_ASYNC metrics) neither queue behind them
nor make the training stream wait for them. Paths that may read parameters
outside the joint MoT loop (checkpoint saving, the video-only and compiled eval
loops, the end of training) call finish() first.
"""

import hashlib
import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist

from openwam.optimizations.zero import step_with_gather, validate_zero

logger = logging.getLogger(__name__)
# The gather with outstanding pieces; read by the MoT loop without a call.
PENDING = None
# A run shorter than this joins a neighbour that is needed no later.
_MERGE_ELEMENTS = 4 * 1024 * 1024


@dataclass(frozen=True)
class Piece:
    stage: int
    group: int
    start: int
    length: int
    owner: int


def layer_stages(architecture):
    """Stage per trainable parameter: MoT layer i is i + 1; everything else 0."""
    if type(architecture).__name__ != "DualSystemSelfAttnArchitecture":
        raise ValueError("OPENWAM_OPT_DEFERRED_GATHER currently requires dual_system/joint_self_attn")
    video = architecture.video_backbone.dit.blocks
    action = architecture.action_backbone.blocks
    if len(video) != len(action):
        raise ValueError("Deferred gather requires equal video and action layer counts")
    stages = {}
    for layer, blocks in enumerate(zip(video, action)):
        for block in blocks:
            for param in block.parameters():
                stages[id(param)] = layer + 1
    return stages, len(video) + 1


def plan(flats, groups, partitions, stages, tail):
    """Pieces in waiting order. Unmapped parameters are stage 0, alignment padding ``tail``."""
    pieces = []
    for index, (flat, params, parts) in enumerate(zip(flats, groups, partitions)):
        base, size = flat.data_ptr(), flat.element_size()
        storage = flat.untyped_storage().data_ptr()
        spans = []
        for param in params:
            data = param.data
            if data.untyped_storage().data_ptr() != storage or not data.is_contiguous():
                raise RuntimeError("A ZeRO parameter is not a view of its flat bit16 buffer")
            start = (data.data_ptr() - base) // size
            spans.append((start, start + data.numel(), stages.get(id(param), 0)))
        spans.sort()
        for owner, part in enumerate(parts):
            low = (part.data_ptr() - base) // size
            high = low + part.numel()
            runs, cursor = [], low
            for start, stop, stage in spans:
                left, right = max(start, cursor), min(stop, high)
                if right <= left:
                    continue
                if left > cursor:
                    runs.append((cursor, left, tail))
                runs.append((left, right, stage))
                cursor = right
            if cursor < high:
                runs.append((cursor, high, tail))
            merged = []
            for start, stop, stage in runs:
                if merged:
                    last = merged[-1]
                    small = stop - start < _MERGE_ELEMENTS
                    if last[2] == stage or (small and stage >= last[2]):
                        last[1] = stop
                        continue
                    if last[1] - last[0] < _MERGE_ELEMENTS and last[2] >= stage:
                        last[1], last[2] = stop, stage
                        continue
                merged.append([start, stop, stage])
            pieces.extend(Piece(stage, index, start, stop - start, owner) for start, stop, stage in merged)
    pieces.sort(key=lambda piece: (piece.stage, piece.group, piece.start))
    return pieces


class DeferredGather:
    """Replaces the step-end AllGather; ``patch_step=False`` when OnlineEncoder calls gather()."""

    def __init__(self, architecture, optimizer, *, patch_step=True):
        validate_zero(optimizer)
        stages, self.tail = layer_stages(architecture)
        dp_group = optimizer.dp_process_group
        self.flats = list(optimizer.bit16_groups_flat)
        self.pieces = []
        if dist.get_world_size(dp_group) > 1:
            self.pieces = plan(self.flats, optimizer.bit16_groups, optimizer.parallel_partitioned_bit16_groups,
                               stages, self.tail)
        self.piece_views = [self.flats[piece.group].narrow(0, piece.start, piece.length) for piece in self.pieces]
        self.sources = [dist.get_global_rank(dp_group, owner) for owner in range(dist.get_world_size(dp_group))]
        # Every rank must create the group, in the same order as other groups.
        self.group = dist.new_group(ranks=dist.get_process_group_ranks(dp_group))
        self._check_plan()
        self.optimizer = optimizer
        self.original_step = optimizer.step
        self.step = step_with_gather(optimizer, self.gather) if patch_step else None
        self.works = []
        self.next = 0
        layers = sum(piece.length for piece in self.pieces if 0 < piece.stage < self.tail)
        total = sum(piece.length for piece in self.pieces)
        logger.info(
            "[optimizations] deferred ZeRO gather: %d broadcasts, %.1f%% of elements waited per MoT layer",
            len(self.pieces), 100.0 * layers / max(total, 1),
        )

    def _check_plan(self):
        text = repr([(p.stage, p.group, p.start, p.length, p.owner) for p in self.pieces])
        digest = int.from_bytes(hashlib.sha256(text.encode()).digest()[:7], "little")
        device = self.flats[0].device if self.flats else torch.device("cpu")
        values = torch.tensor([digest, -digest], dtype=torch.int64, device=device)
        dist.all_reduce(values, op=dist.ReduceOp.MAX, group=self.group)
        if values.tolist() != [digest, -digest]:
            raise RuntimeError("Deferred gather plans differ across ranks")

    def __enter__(self):
        if self.step is not None:
            self.optimizer.step = self.step
        logger.info("[optimizations] ZeRO-2 parameter gather deferred into the next forward")
        return self

    def __exit__(self, *exc):
        self.finish()
        if self.step is not None:
            self.optimizer.step = self.original_step

    def gather(self, groups_flat, partitioned_param_groups, dp_process_group,
               start_alignment_factor=None, allgather_bucket_size=None):
        """Launch every piece and return; the training stream does not wait here."""
        global PENDING
        self.finish()
        if len(groups_flat) != len(self.flats) or any(a is not b for a, b in zip(groups_flat, self.flats)):
            raise RuntimeError("ZeRO flat buffers changed after the deferred gather plan was built")
        # Each collective first waits for the optimizer update already queued
        # on the current stream, then runs on the group's communication stream.
        if getattr(self, "piece_views", None) is None:
            self.piece_views = [self.flats[piece.group].narrow(0, piece.start, piece.length) for piece in self.pieces]
        self.works = [
            dist.broadcast(view, src=self.sources[piece.owner], group=self.group, async_op=True)
            for piece, view in zip(self.pieces, self.piece_views)
        ]
        self.next = 0
        if self.works:
            PENDING = self

    def wait(self, stage):
        """Make the current stream wait for every piece needed up to ``stage``."""
        global PENDING
        while self.next < len(self.works) and self.pieces[self.next].stage <= stage:
            self.works[self.next].wait()
            self.next += 1
        if self.next >= len(self.works):
            self.works = []
            if PENDING is self:
                PENDING = None

    def wait_layer(self, layer):
        self.wait(layer + 1)

    def finish(self):
        self.wait(self.tail)


def wait_prologue():
    if PENDING is not None:
        PENDING.wait(0)


def finish():
    if PENDING is not None:
        PENDING.finish()
