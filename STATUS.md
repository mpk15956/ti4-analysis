# Project status

> **Snapshot of long-term state.** This file mirrors observable repo state at
> the moment of last update. It is not a hand-maintained TODO list — it should
> be regenerable from the commands in the *Verification* section below. The
> fine-grained in-session work tracking lives in Claude Code's `TodoWrite`
> (ephemeral); this file is the thing you read to pick up cold. The current
> close-out arc plan is `~/.claude/plans/where-we-are-composed-snail.md`.

**Last updated:** 2026-06-18 (close-out arc: §3.10 RQ2/RQ4 drafted + depth-vs-breadth mechanism block; breadth-tax figure single-sourced; §3.10 prose<->json manifest bridge added; Phase 6 pins 11/11)

---

## Where the analysis is

The analysis phase is **closed**. Two canonical experiments back the paper, each
serving different research questions:

| Experiment | Serves | Canonical run dir | State |
|-----------|--------|-------------------|-------|
| Five-condition ablation (SA instrument, C0-C4) | RQ1, RQ3 (within-C4) | `output/paper1_canonical_*` (submit_paper1.sh) | Complete |
| Multi-algorithm comparison (RS/HC/SA/SGA/NSGA-II/TS) | RQ2, RQ4 | `output/paper1_multialgo_canonical_stage2_20260526_105143/` | Complete; 500k tail recovered (`output/recovery_500k_tail/.merged_done`, 100/100 at 500k) |
| NSGA-II `evals_to_best` targeted fill (≤200k) | RQ4 | `output/rq4_nsga2_fill_20260615_001023/` (job 28684) | Complete; cross-container seam closed bit-identically (`scripts/rq4_verify_cross_container.py`) |
| Phase 6/7 finalize (Track B + RQ2/RQ3) + six-way RQ4 splice | RQ2/RQ3/RQ4 numbers | `output/paper1_500k_finalize_phase67_*_rq4splice/` | Complete; `manuscript_values_phase6.json` regenerated with RQ4 |

Six commits sit on `main` staged for the user's push. The headline RQ2 finding
is a **budget-dependent crossover**, not a flat tie: NSGA-II HV significantly
exceeds SA from b5000 (Holm-corrected) but with a negligible effect (VDA stays in
0.44-0.61 across budgets); n.s. at b1000. SA stays within a negligible effect of
the dedicated MOEA at every budget, unlike the other four scalars.

---

## Close-out arc (in flight)

Endpoint: gate-green, ready to format and submit. Plan file has the full arc.

| Step | What | Where | State |
|------|------|-------|-------|
| §3.10 | RQ2 mechanism prose (depth-vs-breadth) + RQ4 result + construct/portfolio caveat | manuscript | **Drafted this session** from the user/advisor-provided prose (RQ2/RQ4 Result lines + mechanism block); every cited number single-sourced + pinned. For user review/refinement |
| A | RQ4 instrumentation: NSGA-II `evals_to_best` (canonical 1:1:1, pure sink) + Friedman sentinel guard + RQ4 CSV + generator hook + pins | elis | **Done this session** (verified, see Tests) |
| B | RQ2/RQ4 prose<->json manifest bridge entries | elis | **Done this session** — 7 §3.10 Tier-2 entries + verifiers (RQ2 VDA crossover, RQ4 chi2, breadth-tax median/fraction); green against the local regenerated json (prose<->json edge) |
| C | Doc-drift sweep (5:5:3 comment, manifest header, this file, "advantage" language) | elis | **Done** |
| D | Review + push (close-out commits) | user | pending; does NOT block §3.10 |
| E | `.sif` rebuild + provenance record + in-SIF canary | GACRC (sbatch) | **Done** (job 28683) |
| F | NSGA-II `evals_to_best` fill (Option 1, ≤200k) + bit-identical seam check + six-way splice + finalize + regenerate json | GACRC + elis | **Done this session** (job 28684; see audit doc) |
| G | Fold RQ4/RQ2 numbers into §3.10; add RQ4 manifest entries (step B); full gate | GACRC | §3.10 prose + manifest **done**; **full gate pending** (`submit_rq4_finalize.sh` then `test_manuscript_values` where Phase 0/1 artifacts live) |

RQ4 needed a re-run because NSGA-II's `evals_to_best` was never instrumented (a
`-1` sentinel) and is not backfillable; the A fix records it as the canonical
1:1:1 composite-best eval index via a return-ignored callback (proven pure). The
fill was a **targeted NSGA-II-only re-run to ≤200k** (Option 1), spliced onto the
banked dataset after the freshly rebuilt `.sif` was shown to reproduce the banked
NSGA-II maps **bit-identically** (final_tile_layout 100% exact, every metric
Δ = 0). Six-way Friedman is significant at every budget ≤200k (canonical 200k:
chi2 = 456.80, p = 1.7e-96, df = 5, n = 100). Full diagnostic narrative:
`docs/audit/rq4_evals_to_best_close_2026-06.md`.

---

## Manuscript fill-ins outstanding

| Section | Outstanding | Source (exists when) |
|---------|-------------|----------------------|
| §3.10 RQ2 | mechanism prose + crossover numbers | **Drafted** (Result line + mechanism block); numbers single-sourced from `manuscript_values_phase6.json` (`rq2_by_budget`) and pinned |
| §3.10 RQ4 | result line + omnibus numbers + breadth tax | **Drafted**; chi2 456.80 / df 5 / n 100 / p 1.7e-96 + NSGA-II median `evals_to_best` 134,100 (67% of budget), all single-sourced + pinned |
| Manifest (step B) | §3.10 RQ2/RQ4 Tier-2 entries + verifiers | **Done** — 7 entries, green against local json |

§3.7 (B) Phase 0 cv/held-out values are concrete in `Methodology_Section.md` (no
PENDING markers remain there). RQ1/RQ3 prose and numbers are in place.

---

## Tests

Run on elis with **`pixi run --frozen`** (the bare `pixi run` re-solves over the
NFS home and hangs; `--frozen` skips the solve and works in seconds; pixi also
redirects its repodata cache off NFS to /tmp automatically). Real/full runs go to
GACRC inside the `.sif`.

Verified this session (`pixi run --frozen -e default pytest ...`):

| Suite | State |
|-------|-------|
| Full suite | **182 passed, 2 skipped, 1 failed** — the lone failure is `test_manuscript_values`; all 32 sub-failures are **absent Phase 0/1/4/5 artifacts** (GACRC-only, not a regression). The 7 new §3.10 prose<->json entries all pass (anchor/collision/existence/provenance). Green on GACRC where artifacts live |
| `tests/test_phase6_canonical_values.py` | **11/11 pass** against the rq4splice finalize dir: 3 RQ4 omnibus pins + the new breadth-tax pin (median 134,100 / 67% of budget, json<-csv) + RQ2/RQ3/Track B reproduce banked values (single-source proof) |
| `tests/test_nsga2_optimizer.py` | 22 passed, incl. `test_trajectory_callback_is_pure_sink` (RQ4 hook is a byte-identical pure sink) |
| Cross-container seam (`scripts/rq4_verify_cross_container.py`) | fresh `.sif` reproduces banked NSGA-II maps **bit-identically** (2100/2100 layouts exact, every metric Δ = 0) |

The full `test_manuscript_values` gate and `pre_submission_check.sh` run green on
GACRC where the Phase 0/1 artifacts live; on elis the former fails only on artifact
absence.

---

## Verification (how to regenerate this snapshot)

```bash
# Canonical artifacts on disk (GACRC home)
ls -d output/paper1_multialgo_canonical_stage2_*/benchmark_*/results.csv 2>/dev/null
ls -d output/paper1_500k_finalize_phase67_*/manuscript_values_phase6.json 2>/dev/null
cat output/recovery_500k_tail/.merged_done 2>/dev/null

# Tests (elis): MUST use --frozen or pixi hangs on the NFS solve
pixi run --frozen -e default pytest tests/ -q

# Gate state (run where the finalize dir exists — GACRC, or with output/ bound into the .sif)
bash scripts/pre_submission_check.sh

# §3.10 placeholders still open
grep -nE 'results integrated into this section once|PENDING_CANONICAL_[A-Z0-9_]+|\bTBD\b' docs/methodology/Methodology_Section.md

# Git state (6 commits staged for push; close-out A/C add more)
git log --oneline -8
```

If any command disagrees with this file, the file is stale; update is a manual
edit (refresh-on-meaningful-change, not on a schedule).

---

## What this file is *not*

- Not a TODO list (that's `TodoWrite`) and not the close-out plan (that's the
  plan file under `~/.claude/plans/`).
- Not a roadmap. Long-term direction (Paper 2 / Problem A variable-topology swap)
  lives in `README.md` *Future Work*.
- Not a reproducibility manifest. That's `tests/manuscript_values.yaml`
  (artifact-derived numbers) + `tests/test_phase6_canonical_values.py` (RQ2/RQ4
  claim-pins) + `*/run_config.json` sidecars.
