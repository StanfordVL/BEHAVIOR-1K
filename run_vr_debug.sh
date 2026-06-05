#!/usr/bin/env bash
# Run a VR teleop script and capture ALL terminal output (stdout + stderr +
# the C++ crash-handler text) into a timestamped log, then print the real
# error. Output is unbuffered + captured via `script` so nothing is lost when
# the process segfaults on teardown.
#
# Usage:
#   ./run_vr_debug.sh                          # default: controller --view all
#   ./run_vr_debug.sh --mode controller --hand right --arat --view all
#   ./run_vr_debug.sh --script vr_sharpa_hand_teleop.py --hand right --arat --view all
#
# The log path is printed at the start and end.

set -uo pipefail
cd "$(dirname "$0")"

SCRIPT="vr_sharpa_teleop.py"
# allow overriding the python script with --script <file> (must be first arg pair)
if [[ "${1:-}" == "--script" ]]; then SCRIPT="$2"; shift 2; fi

# default args if none given
if [[ $# -eq 0 ]]; then
  set -- --mode controller --hand right --arat --view all
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOG="vr_teleop_run_${TS}.log"

echo "[debug] script : $SCRIPT"
echo "[debug] args   : $*"
echo "[debug] logging full output -> $LOG"
echo "[debug] (Ctrl-C to stop; the real error is extracted at the end)"
echo

# `script` records the whole pseudo-terminal, so the kit [py stderr] lines and
# the breakpad C++ crash dump are all captured even across the segfault.
# python -u disables Python's stdout/stderr buffering so the last lines before
# the crash are flushed. -e propagates the child's exit code.
script -q -e -c "python -u '$SCRIPT' $*" "$LOG"
RC=$?

echo
echo "=================================================================="
echo "[debug] exit code: $RC   full log: $LOG"
echo "[debug] ---- most likely real error (first signal before the crash dump) ----"
# Stop scanning at the breakpad dump; show the meaningful error lines before it.
sed '/A crash has occurred/q' "$LOG" \
  | grep -nE "Traceback|RuntimeError|Error:|Exception|assert|USD edit detected|Segmentation fault|xrCreateInstance|KeyError|AttributeError|ValueError" \
  | grep -viE "deprecated|already registered|Warning|non optional plugin" \
  | tail -40
echo "=================================================================="
echo "[debug] full log saved at: $(pwd)/$LOG"
