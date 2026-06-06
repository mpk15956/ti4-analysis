#!/bin/bash
# Merge the resumed budget=500k seeds (16–99) into the banked stage-2 run.
# Pure file I/O — no container, no compute — so it is safe to run on the teach
# login node. Run ONLY after all 84 array tasks have finished successfully.
#
# It (1) backs up the golden results.csv, (2) appends the 84 new per-seed CSVs
# (headers stripped), (3) copies the new archive/trajectory files in, and
# (4) verifies budget 500k now has 100 distinct seeds before you trust it.
#
# After this passes, re-run Phase 6 + Phase 7 on $MASTER via a SEPARATE sbatch
# (they need apptainer) — do NOT run them here.

set -euo pipefail
cd "$(dirname "$0")"

MASTER="output/paper1_multialgo_canonical_stage2_20260526_105143/benchmark_20260526_105155"
RECOVERY="output/recovery_500k_tail"
MASTER_CSV="$MASTER/results.csv"
DONE_MARKER="$RECOVERY/.merged_done"
LOCK="$RECOVERY/.merge.lock"

# Idempotent + concurrency-safe: this script may be triggered by both the SLURM
# finalize job and the Claude monitor. flock serializes them; the DONE marker
# makes a second run a no-op so rows are never double-appended.
mkdir -p "$RECOVERY"
exec 9>"$LOCK"
flock -n 9 || { echo "merge already running elsewhere; skipping."; exit 0; }
if [ -f "$DONE_MARKER" ]; then
    echo "Already merged ($(cat "$DONE_MARKER")); nothing to do."
    exit 0
fi

[ -f "$MASTER_CSV" ] || { echo "FATAL: master CSV missing: $MASTER_CSV" >&2; exit 1; }
NEW_CSVS=( "$RECOVERY"/seed_*/benchmark_*/results.csv )
[ -e "${NEW_CSVS[0]}" ] || { echo "FATAL: no recovery CSVs under $RECOVERY" >&2; exit 1; }
echo "Found ${#NEW_CSVS[@]} recovery seed CSVs (expect 84)."

# 1) Backup the golden data.
BACKUP="$MASTER_CSV.pre_merge_$(date +%Y%m%d_%H%M%S).bak"
cp "$MASTER_CSV" "$BACKUP"
echo "Backed up master CSV -> $BACKUP"

# 2) Append rows (strip each file's header line).
before=$(wc -l < "$MASTER_CSV")
for f in "${NEW_CSVS[@]}"; do
    tail -n +2 "$f" >> "$MASTER_CSV"
done
after=$(wc -l < "$MASTER_CSV")
echo "Appended rows: $before -> $after (added $((after - before)); expect 84*18 = 1512)"

# 3) Copy new archive + trajectory files into the master.
for sub in pareto_archives unified_archives trajectories; do
    mkdir -p "$MASTER/$sub"
    n=0
    for d in "$RECOVERY"/seed_*/benchmark_*/"$sub"; do
        [ -d "$d" ] || continue
        cp -n "$d"/* "$MASTER/$sub/" 2>/dev/null && n=$((n+1)) || true
    done
    echo "Copied $sub files from $n recovery dirs."
done

# 4) Verify budget 500k now has 100 distinct seeds (awk only — text, no compute).
echo "--- verification ---"
N500=$(awk -F, 'NR==1{for(i=1;i<=NF;i++){if($i=="seed")s=i; if($i=="budget")b=i}; next}
         $b==500000{seen[$s]=1} END{n=0; for(k in seen)n++; print n}' "$MASTER_CSV")
echo "budget 500000 distinct seeds: $N500 $([ "$N500" -eq 100 ] && echo OK || echo '*** EXPECTED 100 ***')"
if [ "$N500" -eq 100 ]; then
    echo "merged $(date -Iseconds): budget 500k complete, 100/100 seeds" > "$DONE_MARKER"
    echo "Done. Wrote $DONE_MARKER. Next: re-run Phase 6 + Phase 7 on $MASTER via sbatch."
else
    echo "VERIFY FAILED: 500k has $N500 distinct seeds, not 100. NOT writing done marker." >&2
    echo "Master CSV restored-from backup available: $BACKUP" >&2
    exit 1
fi
