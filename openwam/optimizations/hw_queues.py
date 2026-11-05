"""HIP hardware-queue sizing so side-stream collectives can overlap compute.

DTK HIP maps streams round-robin onto ``GPU_MAX_HW_QUEUES`` hardware queues
(default 4) at first use. Two streams sharing a queue execute serially, so a
ZeRO reduction or RCCL stream landing on the compute stream's queue removes
all communication/compute overlap even though the streams are distinct.
The queue count does not change stream priorities.
"""

import logging
import os

logger = logging.getLogger(__name__)


def requested() -> int:
    """Queue count from GPU_MAX_HW_QUEUES; unset means leave HIP's default."""
    raw = os.environ.get("GPU_MAX_HW_QUEUES", "").strip()
    if raw.isdigit():
        return int(raw)
    return 0


def configure() -> None:
    """Must run before the HIP runtime initializes; GPU_MAX_HW_QUEUES is read directly by DTK."""
    count = requested()
    if count:
        logger.info("[optimizations] GPU_MAX_HW_QUEUES=%d active", count)
