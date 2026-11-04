"""pytest plugin: after each test, log RTX descriptor settings + any omni.stats entries about descriptors/resources."""

import json
import os
import subprocess

import pytest

OUT = os.environ.get("DESCPROBE_OUT", "descprobe.jsonl")
KEYS = ("descriptor", "Descriptor", "TotalLocalUsed", "texture", "Texture")


def _snapshot(label):
    import time

    rec = {"label": label, "t": time.time()}
    try:
        import carb

        s = carb.settings.get_settings()
        for k in (
            "/rtx/reservedDescriptors",
            "/rtx/descriptorSets",
            "/rtx-transient/resourcemanager/texturestreaming/enabled",
        ):
            rec[k] = s.get(k)
    except Exception as e:
        rec["settings_err"] = repr(e)
    try:
        import omni.stats

        si = omni.stats.get_stats_interface()
        stats = {}
        for scope in si.get_scopes():
            sid = scope["scopeId"] if isinstance(scope, dict) else scope
            sname = scope.get("name", str(sid)) if isinstance(scope, dict) else str(sid)
            for st in si.get_stats(sid):
                name = st.get("name", "")
                if any(k in name for k in KEYS):
                    stats[f"{sname}/{name}"] = st.get("value")
        rec["stats"] = stats
    except Exception as e:
        rec["stats_err"] = repr(e)
    try:
        pid = os.getpid()
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        for line in out.splitlines():
            p, m = [x.strip() for x in line.split(",")]
            if int(p) == pid:
                rec["vram_mib"] = int(m)
    except Exception:
        pass
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    yield
    _snapshot(item.name)
