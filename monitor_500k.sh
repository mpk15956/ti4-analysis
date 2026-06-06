#!/bin/bash
# Read-only status probe for the 500k tail recovery (array job 28494 on teach).
# Prints a parseable status block. Does NOT merge, notify, or mutate anything —
# the cron-fired Claude agent interprets this output and decides what to do.
#
# Safe to run from elis. Honors the "never run compute on login nodes" rule:
# only squeue/sacct/ls over ssh, no container, no python.

ARRAY=28494
LAUNCH_EPOCH=1780274722   # 2026-05-31 ~20:45 ET, array 28494 submit time
SOCK="$HOME/.ssh/cm-mpk15956@teach.gacrc.uga.edu:22"

echo "PROBE_TIME=$(date -Iseconds)"

# VPN check
if ~/bin/gacrc-vpn status 2>/dev/null | grep -q "VPN: UP"; then
    echo "VPN=UP"
else
    echo "VPN=DOWN"
    echo "STATUS=UNKNOWN_VPN_DOWN"
    echo "NOTE=teach unreachable; job 28494 keeps running regardless. Re-up needed only to merge."
    exit 0
fi

# Connectivity guard: clear a stale ControlMaster socket if a quick ssh hangs.
if ! timeout 20 ssh -o BatchMode=yes teach true 2>/dev/null; then
    rm -f "$SOCK"
    if ! timeout 25 ssh -o BatchMode=yes -o ConnectTimeout=15 teach true 2>/dev/null; then
        echo "STATUS=SSH_FAILED"
        echo "NOTE=VPN up but ssh teach failed even after clearing stale socket."
        exit 0
    fi
fi

# Authoritative array task states via sacct (-X = one row per array element).
read -r COMPLETED FAILED RUNNING PENDING OTHER < <(
  timeout 40 ssh teach "sacct -j ${ARRAY} -X -n -P --format=State 2>/dev/null" | awk '
    /COMPLETED/{c++} /FAILED|TIMEOUT|CANCELLED|NODE_FAIL|OUT_OF_MEMORY/{f++}
    /RUNNING/{r++} /PENDING/{p++} END{print c+0, f+0, r+0, p+0}'
)
echo "COMPLETED=${COMPLETED:-0}"
echo "FAILED=${FAILED:-0}"
echo "RUNNING=${RUNNING:-0}"
echo "PENDING=${PENDING:-0}"

# Seed dirs that have STARTED (engine writes results.csv header at launch, so
# this counts started seeds, not finished ones — completion keys off COMPLETED).
STARTED=$(timeout 30 ssh teach 'ls ~/projects/ti4-analysis/output/recovery_500k_tail/seed_*/benchmark_*/results.csv 2>/dev/null | wc -l')
echo "STARTED_SEEDS=${STARTED:-0}"

ELAPSED_H=$(( ( $(date +%s) - LAUNCH_EPOCH ) / 3600 ))
echo "ELAPSED_HOURS=${ELAPSED_H}"

if [ "${COMPLETED:-0}" -ge 84 ] && [ "${RUNNING:-0}" -eq 0 ] && [ "${PENDING:-0}" -eq 0 ]; then
    echo "STATUS=COMPLETE"
elif [ "${FAILED:-0}" -gt 0 ] && [ "${RUNNING:-0}" -eq 0 ] && [ "${PENDING:-0}" -eq 0 ]; then
    echo "STATUS=DONE_WITH_FAILURES"
elif [ "${FAILED:-0}" -gt 0 ]; then
    echo "STATUS=IN_PROGRESS_WITH_FAILURES"
else
    echo "STATUS=IN_PROGRESS"
fi
