"""
Logs USD stage changes (Usd.Notice.ObjectsChanged) to a file.

Activated by setting the OG_USD_CHANGE_LOG environment variable to a file path before
starting OmniGibson. Each notice entry records the changed prim/property paths and a
stack trace filtered to OmniGibson/user frames so the call site is easy to find.

Typical CI usage (set in the workflow env block):
    OG_USD_CHANGE_LOG: usd_changes.log
"""

import threading
import traceback
from datetime import datetime

# Module-level state — only one logger is active at a time.
_log_file = None
_listener = None  # Tf.Notice.Key; must be held to keep the listener alive
_lock = threading.Lock()

# Site-packages prefixes to strip from stack traces so only OmniGibson / user frames remain.
_SKIP_FRAME_SUBSTRINGS = (
    "/site-packages/",
    "/dist-packages/",
    "<frozen ",
)


def _filtered_stack() -> str:
    frames = traceback.format_stack()
    # Drop the bottom frames that are always this module's own internals.
    # traceback.format_stack() has innermost frame last; the last 2 are
    # _filtered_stack() itself and _on_objects_changed().
    frames = frames[:-2]
    kept = [f for f in frames if not any(s in f for s in _SKIP_FRAME_SUBSTRINGS)]
    return "".join(kept) if kept else "  <no user frames>\n"


def _on_objects_changed(notice, stage):
    if _log_file is None:
        return

    resynced = [str(p) for p in notice.GetResyncedPaths()]
    info_only = [str(p) for p in notice.GetChangedInfoOnlyPaths()]

    if not resynced and not info_only:
        return

    ts = datetime.now().isoformat(timespec="milliseconds")
    stack = _filtered_stack()

    with _lock:
        if resynced:
            paths = ", ".join(resynced[:20])
            suffix = f" (+{len(resynced) - 20} more)" if len(resynced) > 20 else ""
            _log_file.write(f"[{ts}] RESYNCED: {paths}{suffix}\n")
        if info_only:
            paths = ", ".join(info_only[:20])
            suffix = f" (+{len(info_only) - 20} more)" if len(info_only) > 20 else ""
            _log_file.write(f"[{ts}] INFO_CHANGED: {paths}{suffix}\n")
        _log_file.write(f"  Stack:\n{stack}\n")
        _log_file.flush()


def start(log_path: str, stage) -> None:
    """Register the USD change listener and open the log file.

    Safe to call more than once; subsequent calls are no-ops if already started.
    """
    global _log_file, _listener

    if _listener is not None:
        return  # already running

    from pxr import Tf, Usd

    _log_file = open(log_path, "w", buffering=1)  # line-buffered so CI can tail it live
    _listener = Tf.Notice.Register(Usd.Notice.ObjectsChanged, _on_objects_changed, stage)

    _log_file.write(f"# USD change log — started {datetime.now().isoformat()}\n")
    _log_file.write(f"# Stage: {stage.GetRootLayer().identifier}\n\n")
    _log_file.flush()


def stop() -> None:
    """Unregister the listener and close the log file."""
    global _log_file, _listener

    if _listener is None:
        return

    try:
        _listener.Revoke()
    except Exception:
        pass
    _listener = None

    if _log_file is not None:
        with _lock:
            _log_file.write(f"\n# USD change log — stopped {datetime.now().isoformat()}\n")
            _log_file.close()
        _log_file = None
