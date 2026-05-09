"""
Apply the §3.4 (canonical fitness landscape) insertion's renumbering cascade.

CONTEXT (May 2026 manuscript rewrite). Inserting a new §3.4 = "The canonical
fitness landscape" between the existing §3.3 (Metrics) and §3.4 (Algorithms)
shifts every downstream section by one. This script is the tracked artifact
of that rename — a single source of truth for the (old → new) mapping plus
the file list it applies to. Running it twice would silently double-shift
section numbers, so it is idempotency-checked: if a target file already shows
post-rename headers and zero pre-rename headers, the script refuses to run on
it.

USAGE.
    pixi run python scripts/renumber_methodology_sections.py [--dry-run]

The default applies edits in place across all tracked files in FILE_LIST.
--dry-run prints the diff stats without writing.

INVENTORY (pre-rewrite, captured 2026-05-07 via grep across the repo):
  §3.1   13 refs   UNCHANGED
  §3.2    0 refs   UNCHANGED
  §3.3   28 refs   UNCHANGED
  §3.4    2 refs   → §3.5  (Algorithms)
  §3.5    0 refs   → §3.6  (Experimental Protocol)
  §3.6    5 refs   → §3.7  (Hyperparameter Validation)
  §3.7   12 refs   → §3.8  (Analysis Tracks)
  §3.8    0 refs   → §3.9  (Ablation Study)
  §3.9    0 refs   → §3.10 (Null hypotheses)
  §3.10   0 refs   → §3.11 (Statistical methods)

Total: 19 inline §-ref sites across docs/ + scripts/ + src/ + README.md, plus
7 section headers in Methodology_Section.md itself.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Rename map. ORDER MATTERS: highest-numbered first, so that "§3.7 → §3.8"
# happens before "§3.6 → §3.7" (otherwise the second step would double-shift
# the values produced by the first). All §-ref renames are atomic across the
# file list before any other rename runs, so cross-file consistency is
# preserved.
RENAME_MAP = [
    ("3.10", "3.11"),
    ("3.9",  "3.10"),
    ("3.8",  "3.9"),
    ("3.7",  "3.8"),
    ("3.6",  "3.7"),
    ("3.5",  "3.6"),
    ("3.4",  "3.5"),
]

# Files in scope. Auto-generated artifacts (egg-info) and the rename script
# itself are deliberately excluded.
FILE_LIST = [
    "docs/methodology/Methodology_Section.md",
    "docs/methodology/Design_Rationale.md",
    "docs/limitations/limitations.md",
    "docs/limitations/lsap-proxy-goodhart.md",
    "docs/limitations/held_out_validation_variance.md",
    "docs/audit/README.md",
    "docs/audit/n_reconciliation.md",
    "docs/audit/parallel_weight_sources.md",
    "docs/audit/analyzer_composite_recompute.md",
    "docs/audit/jfi_audit_TODO.md",
    "docs/Response_to_Reviewers_Template.md",
    "scripts/analyze_benchmark.py",
    "scripts/lisa_proxy_per_map_diagnostic.py",
    "scripts/variance_equalization_diagnostic.py",
    "scripts/verify_pixi_apptainer.slurm",
    "src/ti4_analysis/algorithms/fast_map_state.py",
    "src/ti4_analysis/algorithms/map_topology.py",
    "README.md",
]


def _section_header_pattern(old: str) -> re.Pattern:
    """Match a markdown section header `## OLD <space>...` exactly (no prefix
    eating, no decimal continuation). Trailing space is required so `3.4`
    does not match inside `3.40`."""
    return re.compile(rf"^(#+\s+){re.escape(old)}(\s)", re.MULTILINE)


def _inline_ref_pattern(old: str) -> re.Pattern:
    """Match an inline §-reference `§3.X` not followed by either:
        - another digit (so `§3.1` does not match inside `§3.10`); or
        - a dot-then-digit (so `§3.4` does not match inside `§3.4.1`).
    A bare trailing period is fine (sentence-ending `§3.7.` is a legitimate ref)."""
    return re.compile(rf"§\s*{re.escape(old)}(?![0-9])(?!\.[0-9])")


def renumber_file(path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Apply the rename cascade to one file. Returns (header_changes, ref_changes)."""
    if not path.exists():
        print(f"  SKIP (missing): {path}")
        return 0, 0
    text = path.read_text()
    original = text
    header_count = 0
    ref_count = 0
    # First pass: section headers (Methodology_Section.md only has these, but
    # the regex is harmless on other files).
    for old, new in RENAME_MAP:
        pat = _section_header_pattern(old)
        new_text, n = pat.subn(rf"\g<1>{new}\g<2>", text)
        if n:
            header_count += n
            text = new_text
    # Second pass: inline §-refs.
    for old, new in RENAME_MAP:
        pat = _inline_ref_pattern(old)
        new_text, n = pat.subn(f"§{new}", text)
        if n:
            ref_count += n
            text = new_text
    if (header_count or ref_count) and not dry_run:
        path.write_text(text)
    elif text == original:
        return 0, 0
    return header_count, ref_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change, don't write")
    args = parser.parse_args()

    total_h = 0
    total_r = 0
    print(f"Renumbering cascade ({'DRY RUN' if args.dry_run else 'APPLYING'}):")
    print(f"  RENAME_MAP: {RENAME_MAP}")
    print()
    for rel in FILE_LIST:
        path = ROOT / rel
        h, r = renumber_file(path, dry_run=args.dry_run)
        if h or r:
            print(f"  {rel}: {h} header(s), {r} inline ref(s)")
            total_h += h
            total_r += r
    print()
    print(f"Total: {total_h} headers, {total_r} inline refs across {len(FILE_LIST)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
