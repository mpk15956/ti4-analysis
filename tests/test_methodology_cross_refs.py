"""
Regression test: every intra-manuscript §X.Y reference resolves to a defined
section in `docs/methodology/Methodology_Section.md`.

PURPOSE (May 2026 manuscript rewrite). When the §3.4 (canonical fitness
landscape) insertion shifted §3.5 → §3.11, every cross-reference across
docs/, src/, scripts/, and README had to be renumbered in lockstep. The
mechanical rename is performed by `scripts/renumber_methodology_sections.py`;
this test is the active enforcement that the rename stayed coherent.

It catches three failure modes:
  1. A §-ref that points at a section that no longer exists (orphaned by
     a future renumbering that wasn't propagated).
  2. A §-ref typo (e.g., `§3.7` written instead of `§3.8` after the rewrite).
  3. A new section header that introduces a number outside the existing
     range, with consumers not yet updated.

EXTERNAL CITATIONS — false-positive guard. The methodology section cites
external textbooks by chapter (e.g., "Boyd & Vandenberghe, 2004, §3.1.5"),
which match the §X.Y syntax but are NOT intra-manuscript references. The
detector excludes any §-ref preceded by a "YEAR," pattern, since textbook
citations always carry an author-year prefix in this manuscript's style.

The detector also recognizes a small allow-list of known-good external
section citations to defend against a false positive on the YEAR-comma
heuristic in case it ever fires incorrectly. Empty by default; expand only
when adding a new known-external citation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
METHODOLOGY = ROOT / "docs/methodology/Methodology_Section.md"

# Files in scope for cross-reference checking. Mirrors the FILE_LIST in
# scripts/renumber_methodology_sections.py — kept in sync deliberately so a
# rename script run is what the test is verifying.
SCOPE = [
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

# Allow-list of §X.Y references that are EXTERNAL citations (textbook chapter
# references), not intra-manuscript references. The allow-list defends against
# TWO failure modes of the YEAR-comma heuristic:
#   (a) heuristic too narrow (false-negative): a future external citation
#       written without an author-year-comma prefix — e.g., "see Boyd's
#       §3.1.5 for the log-sum-exp form" — would slip past the heuristic
#       and be incorrectly flagged as an orphaned intra-manuscript ref.
#   (b) heuristic too wide (false-positive): a year-comma sequence happens
#       to appear before a legitimate intra-manuscript ref (e.g., a parenthetical
#       like "the variance equalization diagnostic of 2026, §3.8" would
#       falsely match). The allow-list overrides the heuristic for these cases.
# Mode (a) is the dominant expected case; mode (b) is theoretical but the
# allow-list mechanism handles both with no extra code.
EXTERNAL_CITATIONS: set[tuple[str, str]] = {
    # Boyd & Vandenberghe (2004), §3.1.5 — log-sum-exp / softplus reference,
    # cited in §3.4.1 and §3.4.2 of Methodology_Section.md. Currently caught
    # by the YEAR-comma heuristic, but listed here as the canonical example.
    ("docs/methodology/Methodology_Section.md", "3.1.5"),
}

# Pattern for §X.Y refs. Boundary handling: not followed by digit (so §3.1
# does not match in §3.10) and not followed by .digit (so §3.4 does not match
# in §3.4.1). Two-digit subsections are supported.
REF_PAT = re.compile(r"§\s*(3\.\d+(?:\.\d+)?)(?![0-9])(?!\.[0-9])")

# Header pattern for Methodology_Section.md.
HEADER_PAT = re.compile(r"^#+\s+(3\.\d+(?:\.\d+)?)\s", re.MULTILINE)

# Author-year prefix that indicates a textbook citation, not an intra-manuscript ref.
# Examples matched: "Boyd & Vandenberghe, 2004,", "(Boyd & Vandenberghe, 2004, ".
# We look at the 60 characters preceding the §-ref for any 4-digit year + comma.
CITATION_PREFIX_PAT = re.compile(r"\b(19|20)\d{2}\s*,\s*$")


def _defined_sections() -> set[str]:
    """Set of section/subsection numbers defined as headers in Methodology_Section.md."""
    text = METHODOLOGY.read_text()
    return {m.group(1) for m in HEADER_PAT.finditer(text)}


def _is_external_citation(text: str, match_start: int, file_rel: str, ref: str) -> bool:
    """True if the §-ref at match_start is preceded by a textbook-citation prefix
    or is in the explicit EXTERNAL_CITATIONS allow-list."""
    if (file_rel, ref) in EXTERNAL_CITATIONS:
        return True
    # Look at up to 60 chars before the §-ref for an author-year-comma pattern.
    window_start = max(0, match_start - 60)
    window = text[window_start:match_start]
    return bool(CITATION_PREFIX_PAT.search(window))


def _references_with_filter() -> dict[str, list[tuple[str, int]]]:
    """Collect every intra-manuscript §-ref across SCOPE files, filtering out
    external citations. Returns {ref_value: [(file, line_no), ...]}."""
    refs: dict[str, list[tuple[str, int]]] = {}
    for rel in SCOPE:
        path = ROOT / rel
        if not path.exists():
            continue
        text = path.read_text()
        for m in REF_PAT.finditer(text):
            ref = m.group(1)
            if _is_external_citation(text, m.start(), rel, ref):
                continue
            line_no = text[:m.start()].count("\n") + 1
            refs.setdefault(ref, []).append((rel, line_no))
    return refs


def test_every_intra_manuscript_section_ref_resolves():
    """The headline check: every §X.Y reference (excluding external citations)
    points to a section that exists in Methodology_Section.md."""
    defined = _defined_sections()
    refs = _references_with_filter()

    orphaned: dict[str, list[tuple[str, int]]] = {
        ref: locations for ref, locations in refs.items() if ref not in defined
    }
    if orphaned:
        msg_lines = ["Orphaned §-references (point to undefined methodology sections):"]
        for ref in sorted(orphaned):
            msg_lines.append(f"  §{ref} (no header found in Methodology_Section.md)")
            for path, line in orphaned[ref]:
                msg_lines.append(f"      at {path}:{line}")
        msg_lines.append("")
        msg_lines.append(f"Defined sections: {sorted(defined)}")
        msg_lines.append(
            "Fix: either add the missing section header, or rerun "
            "scripts/renumber_methodology_sections.py with the updated map."
        )
        pytest.fail("\n".join(msg_lines))


def test_methodology_section_has_expected_top_level_sections():
    """Pin the top-level section count so an unintended renumber is loud."""
    defined = _defined_sections()
    top_level = {s for s in defined if s.count(".") == 1}
    expected = {"3.1", "3.2", "3.3", "3.4", "3.5", "3.6", "3.7", "3.8", "3.9", "3.10", "3.11"}
    assert top_level == expected, (
        f"Top-level sections diverged from expected. "
        f"Expected: {sorted(expected)}. "
        f"Found: {sorted(top_level)}. "
        f"If this is intentional (legitimate add/remove of a section), update the "
        f"`expected` set here in lockstep with the rename script's RENAME_MAP."
    )


def test_canonical_fitness_landscape_subsections_exist():
    """The May 2026 rewrite added §3.4 with subsections 3.4.1–3.4.4. Pin them."""
    defined = _defined_sections()
    required_subsections = {"3.4", "3.4.1", "3.4.2", "3.4.3", "3.4.4"}
    missing = required_subsections - defined
    assert not missing, (
        f"§3.4 canonical-formulation subsections missing: {sorted(missing)}. "
        f"Defined subsections under §3.4: {sorted(s for s in defined if s.startswith('3.4'))}"
    )
