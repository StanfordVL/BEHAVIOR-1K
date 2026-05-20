import os
from cProfile import Profile as CProfile
from time import perf_counter

import psutil

import omnigibson.utils.pynvml_utils as pynvml


class Profiler:
    """
    Lightweight profiler that tracks wall-clock timing across enable/disable cycles.

    Used by the Simulator's `with_profiler` decorator. Each enable/disable pair records
    one timing sample. Callers can read cumulative or per-call averages at any time.

    Optionally wraps a cProfile.Profile for deep (function-level) profiling when
    gm.ENABLE_DEEP_PROFILING is set.
    """

    def __init__(self, deep=False):
        self._total_time = 0.0
        self._call_count = 0
        self._start = None
        self._last_dt = 0.0
        self._cprofiler = CProfile() if deep else None

    def enable(self):
        if self._cprofiler is not None:
            self._cprofiler.enable()
        self._start = perf_counter()

    def disable(self):
        dt = perf_counter() - self._start
        if self._cprofiler is not None:
            self._cprofiler.disable()
        self._last_dt = dt
        self._total_time += dt
        self._call_count += 1

    @property
    def last_dt(self):
        """Wall-clock seconds for the most recent enable/disable pair."""
        return self._last_dt

    @property
    def total_time(self):
        """Cumulative wall-clock seconds across all recorded calls."""
        return self._total_time

    @property
    def call_count(self):
        """Number of completed enable/disable cycles."""
        return self._call_count

    @property
    def average_time(self):
        """Average wall-clock seconds per call, or 0 if no calls recorded."""
        return self._total_time / self._call_count if self._call_count > 0 else 0.0

    def reset(self):
        """Clear all accumulated timing data (does not affect the cProfile profiler)."""
        self._total_time = 0.0
        self._call_count = 0
        self._last_dt = 0.0

    def dump_stats(self, filename):
        """Dump cProfile stats to a file (no-op if deep profiling is disabled)."""
        if self._cprofiler is not None:
            self._cprofiler.dump_stats(filename)


# Method copied from: https://github.com/wandb/wandb/blob/main/wandb/sdk/internal/system/assets/gpu.py
def gpu_in_use_by_this_process(gpu_handle, pid: int) -> bool:
    if psutil is None:
        return False

    try:
        base_process = psutil.Process(pid=pid)
    except psutil.NoSuchProcess:
        return False

    our_processes = base_process.children(recursive=True)
    our_processes.append(base_process)

    our_pids = {process.pid for process in our_processes}

    compute_pids = {process.pid for process in pynvml.nvmlDeviceGetComputeRunningProcesses(gpu_handle)}
    graphics_pids = {process.pid for process in pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu_handle)}

    pids_using_device = compute_pids | graphics_pids

    return len(pids_using_device & our_pids) > 0


def get_vram_usage():
    """Get VRAM usage in GB for the GPU used by this process."""
    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    vram_usage = 0.0
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        if gpu_in_use_by_this_process(handle, os.getpid()):
            vram_usage = pynvml.nvmlDeviceGetMemoryInfo(handle).used / 1024**3
            break
    pynvml.nvmlShutdown()
    return vram_usage
