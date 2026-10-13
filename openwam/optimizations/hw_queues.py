"""HIP hardware-queue sizing so side-stream collectives can overlap compute.

DTK HIP maps streams round-robin onto ``GPU_MAX_HW_QUEUES`` hardware queues
(default 4) at first use. Two streams sharing a queue execute serially, so a
ZeRO reduction or RCCL stream landing on the compute stream's queue removes
all communication/compute overlap even though the streams are distinct.
The queue count does not change stream priorities.
"""

import logging
import os

from openwam.optimizations.config import settings
from openwam.optimizations.dtk import enabled

logger = logging.getLogger(__name__)

def requested():
    """Queue count from the switch; unset/false means leave HIP's default."""
    return (settings().dtk.hw_queues or 0) if enabled() else 0


def configure():
    """Must run before the HIP runtime initializes; an explicit GPU_MAX_HW_QUEUES wins."""
    count = requested()
    if not count:
        return
    import torch

    current = os.environ.get("GPU_MAX_HW_QUEUES")
    if torch.cuda.is_initialized() and current != str(count):
        raise RuntimeError("Set optimizations.dtk.hw_queues before the HIP runtime initializes")
    os.environ["GPU_MAX_HW_QUEUES"] = str(count)
