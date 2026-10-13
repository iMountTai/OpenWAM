"""Optional training profiler, configured by Hydra; DCU trace repair is isolated."""

import contextlib
import gzip
import json
import logging
import os
import time

from openwam.optimizations.config import settings

logger = logging.getLogger(__name__)


def sanitize_dtk_trace(trace_path):
    """Separate CPU and GPU tracks for the DTK Kineto pid-0 metadata quirk."""
    from openwam.optimizations.dtk import enabled as dtk_enabled

    if not dtk_enabled():
        return
    try:
        open_fn = gzip.open if trace_path.endswith(".gz") else open
        with open_fn(trace_path, "rt", encoding="utf-8") as stream:
            data = json.load(stream)
        events = data.get("traceEvents", [])
        cpu_pid = next((event.get("pid") for event in events
                        if event.get("name") == "process_labels"
                        and event.get("args", {}).get("labels") == "CPU"), None)
        modified = False
        for event in events:
            if event.get("pid") == 0 and event.get("cat") == "python_function" and cpu_pid is not None:
                event["pid"] = cpu_pid
                modified = True
            elif (event.get("ph") == "M" and event.get("pid") == 0 and event.get("tid") == 0
                  and event.get("name") == "thread_name"):
                if "thread" in event.get("args", {}).get("name", "").lower():
                    event["args"]["name"] = "stream 0 "
                    modified = True
        if modified:
            with open_fn(trace_path, "wt", encoding="utf-8") as stream:
                json.dump(data, stream)
    except Exception as error:
        logger.warning("[PROFILER] Trace sanitize warning: %s", error)


def build_profiler(output_path, rank):
    config = settings().profiling
    if config.ranks == "off" or (config.ranks == "rank0" and rank != 0):
        return contextlib.nullcontext()
    import torch

    trace_dir = os.path.join(output_path, "profiler_traces")
    os.makedirs(trace_dir, exist_ok=True)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    def trace_handler(profiler):
        step_tag = f"_step{profiler.step_num}" if hasattr(profiler, "step_num") else ""
        trace_path = os.path.join(trace_dir, f"rank{rank}{step_tag}_trace.json.gz")
        started = time.monotonic()
        profiler.export_chrome_trace(trace_path)
        sanitize_dtk_trace(trace_path)
        logger.info("[PROFILER] Rank %d exported %s in %.2fs", rank, trace_path, time.monotonic() - started)

    wait, warmup = (14, 1) if config.light else (15, 0)
    logger.info("[PROFILER] rank=%d wait=%d warmup=%d active=1 with_stack=%s record_shapes=%s traces=%s",
                rank, wait, warmup, not config.light, not config.light, trace_dir)
    return torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=1, repeat=1),
        on_trace_ready=trace_handler,
        record_shapes=not config.light,
        profile_memory=False,
        with_stack=not config.light,
        with_modules=False,
    )
