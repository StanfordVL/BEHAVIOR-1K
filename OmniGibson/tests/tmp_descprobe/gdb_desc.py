"""TEMPORARY: gdb tracer for RTX resource-descriptor usage (investigating CI "Out of resource descriptors!").

Breaks (without stopping) in rtx::resourcemanager::Context::allocateResourceDescriptors inside the stripped
libgpu.foundation.plugin.so and logs capacity / in-use / returned index. Offsets are for omni.gpu_foundation
f9bf0dda (Isaac Sim 6.0.1); the expected instruction bytes are checked before the breakpoints are set.
"""

import os
import time

import gdb

LIB = "libgpu.foundation.plugin.so"
OFF_ENTRY = 0x247E20  # allocateResourceDescriptors(this, count, reserve)
OFF_RET = 0x247FA9  # right after the inner range-allocator call; eax = index or 0xffffffff
EXPECT = {OFF_ENTRY: "f30f1efa415741b8", OFF_RET: "4189c485c00f84ac"}
OUT = open(os.environ.get("DESC_GDB_LOG", "gdb_desc.log"), "a", buffering=1)
state = {"set": False, "calls": 0, "rcalls": 0, "max": 0, "maxidx": -1, "last_print": 0, "last": None}


def u32(addr):
    return int(gdb.parse_and_eval(f"*(unsigned int*){addr}"))


def u64(addr):
    return int(gdb.parse_and_eval(f"*(unsigned long*){addr}"))


class AllocBP(gdb.Breakpoint):
    def stop(self):
        try:
            ctx = int(gdb.parse_and_eval("$rdi"))
            count = int(gdb.parse_and_eval("$esi")) & 0xFFFFFFFF
            reserve = int(gdb.parse_and_eval("$edx")) & 0xFFFFFFFF
            alloc = u64(ctx + 0x1C58)
            used = u32(alloc + 0x24A4) if alloc else -1
            cap = u32(ctx + 0x19B8)
            state["calls"] += 1
            state["max"] = max(state["max"], used)
            state["last"] = (used, cap, count, reserve)
            if (
                state["calls"] <= 3
                or abs(used - state["last_print"]) >= 1000
                or count >= 256
                or state["calls"] % 5000 == 0
            ):
                state["last_print"] = used
                OUT.write(
                    f"{time.time():.1f} calls={state['calls']} used={used} max={state['max']} cap={cap} "
                    f"reservedSetting={u32(ctx + 0x19C0)} count={count} reserve={reserve}\n"
                )
        except Exception as e:
            OUT.write(f"err {e}\n")
        return False


class RetBP(gdb.Breakpoint):
    def stop(self):
        try:
            idx = int(gdb.parse_and_eval("$eax")) & 0xFFFFFFFF
            state["rcalls"] += 1
            if idx == 0xFFFFFFFF:
                OUT.write(
                    f"{time.time():.1f} RET FAIL rcalls={state['rcalls']} last(used,cap,count,reserve)={state['last']} maxidx={state['maxidx']}\n"
                )
            elif idx > state["maxidx"]:
                prev = state["maxidx"]
                state["maxidx"] = idx
                if idx // 5000 != prev // 5000:
                    OUT.write(f"{time.time():.1f} RET maxidx={idx} rcalls={state['rcalls']} last={state['last']}\n")
            if state["rcalls"] % 5000 == 0:
                OUT.write(
                    f"{time.time():.1f} RET rcalls={state['rcalls']} idx={idx} maxidx={state['maxidx']} last={state['last']}\n"
                )
        except Exception as e:
            OUT.write(f"reterr {e}\n")
        return False


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
    if base is None:
        OUT.write("no base found\n")
        return
    inf = gdb.selected_inferior()
    for off, want in EXPECT.items():
        got = bytes(inf.read_memory(base + off, 8)).hex()
        if got != want:
            OUT.write(f"BYTES MISMATCH at {off:#x}: got {got} want {want} -- different plugin build, NOT tracing\n")
            state["set"] = True
            return
    AllocBP(f"*{base + OFF_ENTRY:#x}", internal=True)
    RetBP(f"*{base + OFF_RET:#x}", internal=True)
    state["set"] = True
    OUT.write(f"bp set at base={base:#x} lib={ev.new_objfile.filename}\n")


gdb.events.new_objfile.connect(on_new_objfile)
