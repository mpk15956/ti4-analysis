#!/usr/bin/env bash
# Pre-submission gate. Four checks, all must pass:
#
#   1. No placeholder tokens (PENDING_MAIN_EXPERIMENT, ⚠ INSERT, [INSERT],
#      "TBD", "5:5:3", "replace with your", "must be re-run") remain in docs/.
#      The first three were the original gate; the latter four were added per
#      the May 2026 audit's catch that the gate was missing several real
#      placeholder forms (held-out CV stub text, legacy 5:5:3 weighting cited
#      under the corrected nominal, Goodhart fallback wording, etc.).
#   2. Methodology cross-references resolve (no orphaned §X.Y refs).
#      Implemented by tests/test_methodology_cross_refs.py.
#   3. Manuscript values match their source artifacts and appear in the
#      manuscript prose. Implemented by tests/test_manuscript_values.py
#      against tests/manuscript_values.yaml.
#   4. The claim-evidence audit's now-closed §3.10-3.11 verdicts (RQ2/RQ3/RQ4,
#      Cohen's d_z) carry their SUPERSEDED closure markers, so the dated audit
#      cannot ship asserting a claim the manuscript contradicts (the live
#      cross-file contradiction the June 2026 draft review caught).
#
# Exit 0 if all four pass; exit 1 otherwise. The combination is the
# "no silent desync at submission" gate the audit (feedback_storage_passive_
# verification_active.md) prescribed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

OVERALL_OK=1

# ── Check 1: placeholder regex ─────────────────────────────────────────────
# Scope: manuscript-facing text — README.md (the repo front page / methods
# narrative) plus docs/methodology/ and docs/limitations/. README IS scanned:
# it must not contradict the 1:1:1 canonical, and the gate previously skipped it,
# which let stale 5:5:3 prose drift there uncaught (June 2026). docs/audit/ and
# docs/lit_review/ stay excluded — those are research/historical notes that
# legitimately reference legacy artifacts (e.g., the audit doc records that 5:5:3
# was the legacy weighting; the lit_review summarizes external sources that use
# those weights). Firing the gate on those would be a false positive.
echo "── Pre-submission check 1: placeholder tokens ──"
PLACEHOLDER_PATTERN='PENDING_MAIN_EXPERIMENT|PENDING_CANONICAL_[A-Z0-9_]+|⚠ INSERT|\[INSERT\]|\bTBD\b|\b5:5:3\b|replace with your|must be re-run'
MANUSCRIPT_DIRS="README.md docs/methodology/ docs/limitations/"
MATCHES=$(grep -rE "$PLACEHOLDER_PATTERN" $MANUSCRIPT_DIRS 2>/dev/null || true)
if [ -z "$MATCHES" ]; then
  echo "  OK — 0 placeholder matches in manuscript-facing docs ($MANUSCRIPT_DIRS)"
else
  echo "  FAIL — placeholder matches found in manuscript-facing docs:"
  echo "$MATCHES" | sed 's/^/    /'
  OVERALL_OK=0
fi
echo

# ── Check 2: methodology cross-references resolve ──────────────────────────
echo "── Pre-submission check 2: methodology cross-references ──"
if pixi run pytest tests/test_methodology_cross_refs.py -q --no-header 2>&1 | tail -3; then
  echo "  OK — every §X.Y resolves to a defined section"
else
  echo "  FAIL — orphaned §-references; see test output above"
  OVERALL_OK=0
fi
echo

# ── Check 3: manuscript values match artifacts ──────────────────────────────
echo "── Pre-submission check 3: manuscript values vs source artifacts ──"
if pixi run pytest tests/test_manuscript_values.py -q --no-header 2>&1 | tail -3; then
  echo "  OK — manifest entries match artifacts and appear in manuscript"
else
  echo "  FAIL — manifest validation failed; see test output above"
  OVERALL_OK=0
fi
echo

# ── Check 4: no stale claim-evidence verdict contradicts §3.10 ─────────────
# The May-2026 claim-evidence audit recorded RQ2/RQ3/RQ4 and Cohen's d_z as
# untested; all are now closed in §3.9-3.10. The dated rows are kept as the
# registration trace, but each now carries a SUPERSEDED closure marker. If a
# stale "not formally tested" / "not computed (only VDA)" verdict ships WITHOUT
# that marker, the audit contradicts the manuscript it ships beside, a live
# cross-file contradiction, so fail the gate. Targeted at the one file with
# standing (it is the registration trace cited for the RQ4 wall-clock origin).
echo "── Pre-submission check 4: no stale audit verdict contradicts §3.10 ──"
AUDIT="docs/audit/claim_evidence_audit.md"
if [ ! -f "$AUDIT" ]; then
  echo "  FAIL: $AUDIT missing"
  OVERALL_OK=0
else
  STALE=$(grep -inE 'not formally tested|not computed \(only vda\)' "$AUDIT" | grep -vi 'SUPERSEDED' || true)
  if [ -z "$STALE" ]; then
    echo "  OK: every closed RQ / d_z verdict carries its SUPERSEDED closure marker"
  else
    echo "  FAIL: stale audit verdict(s) without a SUPERSEDED marker (contradicts §3.10):"
    echo "$STALE" | sed 's/^/    /'
    OVERALL_OK=0
  fi
fi
echo

# ── Overall verdict ─────────────────────────────────────────────────────────
if [ "$OVERALL_OK" -eq 1 ]; then
  echo "Pre-submission check: ALL CHECKS PASSED"
  exit 0
else
  echo "Pre-submission check: ONE OR MORE CHECKS FAILED — fix before submission"
  exit 1
fi
