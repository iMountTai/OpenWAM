"""HIP hardware-queue sizing so side-stream collectives can overlap compute.

DTK HIP maps streams round-robin onto ``GPU_MAX_HW_QUEUES`` hardware queues
(default 4) at first use. Two streams sharing a queue execute serially, so a
ZeRO reduction or RCCL stream landing on the compute stream's queue removes
all communication/compute overlap even though the streams are distinct.
The queue count does not change stream priorities.
"""

import logging
import os

from openwam.optimizations.runtime import integer

logger = logging.getLogger(__name__)

SWITCH = "OPENWAM_OPT_HW_QUEUES"


def requested():
    """Queue count from the switch; unset/false means leave HIP's default."""
    if os.environ.get(SWITCH, "").strip().lower() in {"", "0", "false", "no", "off"}:
        return 0
    return integer(SWITCH, 0, minimum=2)


def configure():
    """Must run before the HIP runtime initializes; an explicit GPU_MAX_HW_QUEUES wins."""
    count = requested()
    if not count:
        return
    current = os.environ.setdefault("GPU_MAX_HW_QUEUES", str(count))
    if current != str(count):
        logger.warning("[optimizations] %s=%d ignored: GPU_MAX_HW_QUEUES=%s already set", SWITCH, count, current)
