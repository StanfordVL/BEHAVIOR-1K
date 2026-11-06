"""TEMPORARY: catch the RTX "Out of resource descriptors!" fatal under gdb and dump state.

Sets ONE breakpoint inside rtx::resourcemanager::Context::allocateResourceDescriptors (stripped
libgpu.foundation.plugin.so, omni.gpu_foundation f9bf0dda / Isaac Sim 6.0.1) on the branch taken only when the
inner allocator returned -1, i.e. right before the fatal is logged. No per-allocation overhead. On a hit, gdb
stops; the -ex commands after `run` then call fatal_dump() and kill the process.

DESC_FATAL_TEST=1 moves the breakpoint to the compare that runs on every allocation (same registers), to test
the dump path locally without a real fatal.
"""

import os
import time

import gdb

LIB = "libgpu.foundation.plugin.so"
TEST = os.environ.get("DESC_FATAL_TEST") == "1"
OFF = 0x247FB4 if TEST else 0x247FBE
EXPECT = {0x247FB4: "4183fcff0f853a01", 0x247FBE: "803d817364000075"}
OUTDIR = os.environ.get("DESC_FATAL_DIR", "desc_fatal")
os.makedirs(OUTDIR, exist_ok=True)
LOG = open(os.path.join(OUTDIR, "summary.log"), "a", buffering=1)
state = {"set": False, "hit": False, "t0": time.time()}


def log(msg):
    LOG.write(f"{time.time():.1f} {msg}\n")


def u32(addr):
    return int(gdb.parse_and_eval(f"*(unsigned int*){addr}"))


def u64(addr):
    return int(gdb.parse_and_eval(f"*(unsigned long*){addr}"))


class FatalBP(gdb.Breakpoint):
    def stop(self):
        state["hit"] = True
        return True


def on_new_objfile(ev):
    if state["set"] or not ev.new_objfile.filename.endswith(LIB):
        return
    base = None
    for line in gdb.execute("info proc mappings", to_string=True).splitlines():
        if line.strip().endswith(LIB):
            parts = line.split()
            if int(parts[3], 16) == 0:
                base = int(parts[0], 16)
                break
    state["set"] = True
    if base is None:
        log("no base found")
        return
    inf = gdb.selected_inferior()
    for off, want in EXPECT.items():
        got = bytes(inf.read_memory(base + off, 8)).hex()
        if got != want:
            log(f"BYTES MISMATCH at {off:#x}: got {got} want {want} -- different plugin build, NOT armed")
            return
    FatalBP(f"*{base + OFF:#x}", internal=True)
    log(f"armed at {base + OFF:#x} (offset {OFF:#x}, test={TEST}) lib={ev.new_objfile.filename}")


def fatal_dump():
    """Called from gdb -ex after `run` returns. No-op unless the breakpoint stopped the process."""
    if not state["hit"]:
        log("process ended without hitting the breakpoint")
        return
    inf = gdb.selected_inferior()
    ctx = int(gdb.parse_and_eval("$rbx"))
    alloc = u64(ctx + 0x1C58)
    info = {
        "uptime_s_since_gdb_start": round(time.time() - state["t0"], 1),
        "ctx": hex(ctx),
        "alloc": hex(alloc),
        "capacity": u32(ctx + 0x19B8),
        "reservedSetting": u32(ctx + 0x19C0),
        "live": u32(alloc + 0x24A4) if alloc else None,
        "count_r13": int(gdb.parse_and_eval("$r13")) & 0xFFFFFFFF,
        "ret_r12": int(gdb.parse_and_eval("$r12")) & 0xFFFFFFFF,
        "thread": gdb.selected_thread().num,
    }
    log(f"HIT {info}")
    with open(os.path.join(OUTDIR, "regs.txt"), "w") as f:
        f.write(gdb.execute("info registers", to_string=True))
    # Raw state for offline analysis: the allocator object (11 size-class buckets of 0x268 bytes plus counters)
    # and the start of the resource-manager context.
    for name, addr, size in (("alloc.bin", alloc, 0x2600), ("ctx.bin", ctx, 0x1D00)):
        try:
            with open(os.path.join(OUTDIR, name), "wb") as f:
                f.write(bytes(inf.read_memory(addr, size)))
        except Exception as e:
            log(f"dump {name} failed: {e}")
    try:
        bt = gdb.execute("thread apply all bt 40", to_string=True)
    except Exception as e:
        bt = f"bt failed: {e}"
    with open(os.path.join(OUTDIR, "threads_bt.txt"), "w") as f:
        f.write(bt)
    log(f"dumped to {OUTDIR}; {bt.count('Thread ')} threads")


gdb.events.new_objfile.connect(on_new_objfile)
