"""
Manuscript-values validator. Asserts that every numerical value cited in the
manuscript matches the artifact it came from, AND appears in the manuscript
text where the manifest says it should.

CONTEXT (May 2026 manuscript rewrite). The manifest at
`tests/manuscript_values.yaml` enumerates the contract between
artifact-derived values and the manuscript prose. Without an active check at
the boundary, a CSV update silently drifts from the manuscript (per
`feedback_storage_passive_verification_active.md` — storage is passive,
verification is active).

FOUR PASSES:
    1. Schema + anchor resolution: validate manifest entry shape; for Tier 2
       entries, confirm section_anchor resolves to exactly one section header
       in the target file. Catches author error before any I/O on manuscript
       text or artifacts.
    2. Collision detection: build a (file, anchor, rendered_value) disambig
       index and flag any key that resolves to >1 manifest entry. Forces
       escalation Tier 1 → Tier 2 (or Tier 2 → Tier 3) at manifest-load time.
    3. Existence check: for each entry, count occurrences of rendered_value
       in the appropriate scope (whole file / section window / regex match).
       Fail if count < min_count. Token matching uses a custom-boundary regex
       that rejects [\\d.] on either side, so "0.05" does not match "0.057".
    4. Artifact provenance: each entry names a verifier function that reads
       the source artifact and confirms the raw_value the manifest claims
       matches what's actually in the artifact. Catches "the CSV updated but
       the manifest didn't."

HIERARCHY-AWARE SECTION INDEX. `build_section_index` parses markdown headers
and computes each section's end_line as the first subsequent header at level
≤ the current header's level. So:
    Anchor `3.4`   (level 2) → window includes all of 3.4.1–3.4.4 (until 3.5)
    Anchor `3.4.2` (level 3) → window is just §3.4.2's body (until next ###/##/#)

FAILURE ACCUMULATION. All four passes accumulate failures into one list and
report all-at-once. Fixing one entry and re-running surfaces the next failure
without bailing — needed for batch authoring.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).parent / "manuscript_values.yaml"


# ── Verifier registry ─────────────────────────────────────────────────────
#
# Each verifier reads one source artifact and asserts the manifest's raw_value
# matches what's there, within tolerance. New manifest entries register a new
# verifier here. The signature is (raw_value) -> (ok: bool, detail: str).

VERIFIERS: dict[str, Callable[[Any], tuple[bool, str]]] = {}


def verifier(name: str):
    """Decorator: register a verifier function under `name`."""
    def deco(fn: Callable[[Any], tuple[bool, str]]):
        VERIFIERS[name] = fn
        return fn
    return deco


@verifier("g_size_from_canonical_topology")
def _verify_g_size(raw_value: int) -> tuple[bool, str]:
    """|G| = 31 on the canonical 6p layout, recovered from MapTopology."""
    import sys
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT))
    from tests.test_metric_parity import _canonical_6p_state  # type: ignore[import-not-found]
    topo, _ = _canonical_6p_state()
    actual = int(topo.n_spatial)
    return (actual == raw_value,
            f"MapTopology.n_spatial = {actual}; manifest raw_value = {raw_value}")


@verifier("gen0_moran_share_under_1_1_1")
def _verify_gen0_share(raw_value: float | int) -> tuple[bool, str]:
    """Gen-0 weighted-variance share of the Moran's I hinge term under 1:1:1
    weights, computed from the post-fix variance_equalization CSV. Matches
    the canonical (May 6 post-audit-fix) run, not the legacy 5:5:3 92%."""
    csv_path = ROOT / "output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv"
    if not csv_path.exists():
        return False, f"Artifact missing: {csv_path}"
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    gen0 = df[df["phase"] == "gen0"]
    sigmas = gen0[["hinge_morans_i", "jfi_gap", "lisa_norm"]].std(ddof=1).values
    weights = np.array([1/3, 1/3, 1/3])
    weighted = sigmas * weights
    hinge_share_pct = (weighted[0] / weighted.sum()) * 100
    # raw_value is 90 (an integer percent rounded from the actual share).
    # Tolerance ±1 percentage point absorbs any small re-computation drift
    # while still catching a meaningful regime shift.
    rounded = round(hinge_share_pct)
    return (abs(rounded - raw_value) <= 1,
            f"variance_equalization_*.csv Gen-0 hinge share = {hinge_share_pct:.1f}% "
            f"(rounds to {rounded}); manifest raw_value = {raw_value}")


def _convergence_share_under_1_1_1(component_idx: int, raw_value: int | float) -> tuple[bool, str]:
    """Shared backend for convergence-phase variance shares. component_idx
    selects the component: 0=hinge_morans_i, 1=jfi_gap, 2=lisa_norm. Returns
    the share of total weighted variance under 1:1:1 weights at the
    convergence sample, matching the §3.8 reported numbers."""
    csv_path = ROOT / "output/verify_pixi_apptainer_20260506_205045/variance_equalization_20260506_224438.csv"
    if not csv_path.exists():
        return False, f"Artifact missing: {csv_path}"
    import pandas as pd
    import numpy as np
    df = pd.read_csv(csv_path)
    conv = df[df["phase"] == "convergence"]
    sigmas = conv[["hinge_morans_i", "jfi_gap", "lisa_norm"]].std(ddof=1).values
    weights = np.array([1/3, 1/3, 1/3])
    weighted = sigmas * weights
    share_pct = (weighted[component_idx] / weighted.sum()) * 100
    rounded = round(share_pct)
    return (abs(rounded - raw_value) <= 1,
            f"variance_equalization_*.csv convergence component[{component_idx}] share = "
            f"{share_pct:.1f}% (rounds to {rounded}); manifest raw_value = {raw_value}")


@verifier("convergence_lsap_share_under_1_1_1")
def _verify_convergence_lsap(raw_value: int | float) -> tuple[bool, str]:
    """LSAP share of weighted variance at convergence under 1:1:1 (component_idx=2)."""
    return _convergence_share_under_1_1_1(2, raw_value)


@verifier("convergence_jfi_share_under_1_1_1")
def _verify_convergence_jfi(raw_value: int | float) -> tuple[bool, str]:
    """JFI gap share of weighted variance at convergence under 1:1:1 (component_idx=1)."""
    return _convergence_share_under_1_1_1(1, raw_value)


@verifier("distance_weight_tau_from_sensitivity_report")
def _verify_distance_tau(raw_value: float) -> tuple[bool, str]:
    """Mean Kendall's τ across distance-weight perturbations from the
    legacy Phase 5 sensitivity report. Tolerance 1e-3 (the report rounds
    to 3 decimal places)."""
    path = ROOT / "output/saturation_20260314_205919/dist_sensitivity_20260316_100535/sensitivity_report.txt"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    text = path.read_text()
    m = re.search(r"Mean Kendall's tau across config pairs:\s*([\d.]+)", text)
    if not m:
        return False, f"Pattern 'Mean Kendall's tau across config pairs: <val>' not found in {path}"
    actual = float(m.group(1))
    return (abs(actual - raw_value) <= 1e-3,
            f"sensitivity_report.txt mean τ = {actual}; manifest raw_value = {raw_value}")


@verifier("goodhart_rho_alpha05_from_summary_json")
def _verify_goodhart_rho(raw_value: float) -> tuple[bool, str]:
    """Goodhart Spearman ρ vs α=0.05 cluster count, from the legacy LISA
    proxy validation summary. Tolerance 1e-3 (manifest rounds to 3 decimals
    via {:+.3f}, artifact has full precision)."""
    path = ROOT / "output/saturation_20260314_205919/lisa_validation_20260316_100413/proxy_validation_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("spearman_rho")
    if actual is None:
        return False, f"Key 'spearman_rho' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"proxy_validation_summary.json spearman_rho = {actual}; manifest raw_value = {raw_value}")


@verifier("goodhart_rho_fdr_from_diagnostic")
def _verify_goodhart_rho_fdr(raw_value: float) -> tuple[bool, str]:
    """Goodhart Spearman ρ under FDR-corrected target, from the per-map
    diagnostic produced by scripts/lisa_proxy_per_map_diagnostic.py. Tolerance
    1e-3 (manifest renders to 3 decimal places via {:+.3f})."""
    path = ROOT / "output/saturation_20260314_205919/lisa_validation_20260316_100413/lisa_proxy_per_map_diagnostic.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    try:
        actual = data["tests"]["per_map_rank_total_significant_fdr"]["spearman_rho"]
    except KeyError as e:
        return False, f"Path tests.per_map_rank_total_significant_fdr.spearman_rho missing from {path}: {e}"
    return (abs(actual - raw_value) <= 1e-3,
            f"lisa_proxy_per_map_diagnostic.json FDR ρ = {actual}; manifest raw_value = {raw_value}")


@verifier("lsap_threshold_tau_from_summary_json")
def _verify_lsap_threshold_tau(raw_value: float) -> tuple[bool, str]:
    """Threshold-sensitivity Kendall τ between baseline LSAP and the τ=0.05
    thresholded variant. Source: scripts/lsap_threshold_sensitivity.py
    summary JSON. Tolerance 1e-3."""
    path = ROOT / "output/saturation_20260314_205919/lsap_threshold_20260316_194325/lsap_threshold_summary.json"
    if not path.exists():
        return False, f"Artifact missing: {path}"
    data = json.loads(path.read_text())
    actual = data.get("kendalls_tau")
    if actual is None:
        return False, f"Key 'kendalls_tau' missing from {path}"
    return (abs(actual - raw_value) <= 1e-3,
            f"lsap_threshold_summary.json kendalls_tau = {actual}; manifest raw_value = {raw_value}")


# ── Data structures ──────────────────────────────────────────────────────


@dataclass
class ManifestEntry:
    key: str
    raw_value: Any
    format: str
    files: list[str]
    match: str
    section_anchor: str | None = None
    pattern: str | None = None
    source_artifact: str = ""
    verifier: str | None = None
    min_count: int = 1

    @property
    def rendered_value(self) -> str:
        return self.format.format(self.raw_value)


@dataclass
class Section:
    level: int       # 2 for ##, 3 for ###, etc.
    number: str      # "3.4.2" — used as anchor lookup key
    title: str       # heading text after the number
    start_line: int  # line index of the header
    end_line: int    # exclusive; first subsequent header at level <= self.level


# ── Section index builder (hierarchy-aware) ──────────────────────────────


_HEADER_PAT = re.compile(r"^(#+)\s+(\d+(?:\.\d+)*)\s+(.*?)\s*$")


def build_section_index(text: str) -> dict[str, Section]:
    """Parse a markdown document into a {section_number: Section} index.

    Hierarchical scope rule: end_line for a header at level L is the line of
    the first subsequent header at level <= L. So a `## 3.4` anchor's window
    includes all `### 3.4.X` subsections; a `### 3.4.2` anchor's window is
    just that subsection's own body.
    """
    lines = text.splitlines()
    headers: list[tuple[int, str, str, int]] = []
    for i, line in enumerate(lines):
        m = _HEADER_PAT.match(line)
        if m:
            headers.append((len(m.group(1)), m.group(2), m.group(3), i))
    index: dict[str, Section] = {}
    for i, (level, number, title, start) in enumerate(headers):
        end = len(lines)
        for j in range(i + 1, len(headers)):
            if headers[j][0] <= level:
                end = headers[j][3]
                break
        index[number] = Section(level, number, title, start, end)
    return index


# ── Token matching with custom boundary ──────────────────────────────────


def count_token_matches(haystack: str, token: str) -> int:
    """Count occurrences of `token` in `haystack` with custom word boundaries.

    Rejects matches where the token is part of a longer numeric run:
        - Preceded by a digit or dot ([\\d.]):  `0.31` blocks matching `31`
          inside it; `1.05` blocks `0.05`.
        - Followed by a digit:                  `315` blocks matching `31`.
        - Followed by `.<digit>`:               `31.5` blocks matching `31`,
          but `31.` (sentence-ending) does NOT block — periods at end-of-token
          are commonly punctuation in prose, so the trailing boundary must
          distinguish "decimal continuation" from "sentence break."

    Same two-clause trailing boundary as the cross-ref test in
    tests/test_methodology_cross_refs.py (which originally had `(?![\\d.])` and
    falsely rejected `§3.7.` at sentence end). Same fix recipe applies here.
    """
    pat = re.compile(rf"(?<![\d.]){re.escape(token)}(?![0-9])(?!\.[0-9])")
    return len(pat.findall(haystack))


# ── Validator ────────────────────────────────────────────────────────────


class ManifestValidator:
    def __init__(self, manifest_path: Path, repo_root: Path):
        with open(manifest_path) as f:
            data = yaml.safe_load(f) or {}
        raw_entries = data.get("entries", [])
        self.entries: list[ManifestEntry] = [ManifestEntry(**e) for e in raw_entries]
        self.repo_root = repo_root
        self.failures: list[str] = []
        self._section_indices: dict[str, dict[str, Section]] = {}
        self._file_texts: dict[str, str] = {}

    def _file_text(self, rel: str) -> str:
        if rel not in self._file_texts:
            path = self.repo_root / rel
            if not path.exists():
                raise FileNotFoundError(f"Manifest file missing on disk: {rel}")
            self._file_texts[rel] = path.read_text()
        return self._file_texts[rel]

    def _section_index_for(self, rel: str) -> dict[str, Section]:
        if rel not in self._section_indices:
            self._section_indices[rel] = build_section_index(self._file_text(rel))
        return self._section_indices[rel]

    def _fail(self, msg: str) -> None:
        self.failures.append(msg)

    # Pass 1 — Schema + anchor resolution -------------------------------------
    def pass1_schema_and_anchor_resolution(self) -> None:
        seen_keys: set[str] = set()
        for entry in self.entries:
            if entry.key in seen_keys:
                self._fail(f"[{entry.key}] duplicate manifest key")
            seen_keys.add(entry.key)
            if entry.match not in ("token", "section", "regex"):
                self._fail(f"[{entry.key}] invalid match type: {entry.match!r}")
                continue
            if entry.match == "section" and not entry.section_anchor:
                self._fail(f"[{entry.key}] match=section requires section_anchor")
                continue
            if entry.match == "regex" and not entry.pattern:
                self._fail(f"[{entry.key}] match=regex requires pattern")
                continue
            for rel in entry.files:
                try:
                    text = self._file_text(rel)
                except FileNotFoundError as e:
                    self._fail(f"[{entry.key}] {e}")
                    continue
                if entry.match == "section":
                    idx = self._section_index_for(rel)
                    if entry.section_anchor not in idx:
                        available = sorted(idx.keys())[:12]
                        self._fail(
                            f"[{entry.key}] anchor {entry.section_anchor!r} resolves to 0 "
                            f"sections in {rel}; first available: {available}"
                        )

    # Pass 2 — Collision detection --------------------------------------------
    def pass2_collision_detection(self) -> None:
        disambig: dict[tuple[str, str | None, str], list[str]] = {}
        entries_by_key = {e.key: e for e in self.entries}
        for entry in self.entries:
            if entry.match == "regex":
                continue  # Tier 3 entries skip token-collision check
            try:
                rendered = entry.rendered_value
            except Exception as e:
                self._fail(f"[{entry.key}] format/raw_value error: {e}")
                continue
            for rel in entry.files:
                anchor = entry.section_anchor if entry.match == "section" else None
                key = (rel, anchor, rendered)
                disambig.setdefault(key, []).append(entry.key)
        for (rel, anchor, token), entry_keys in disambig.items():
            if len(entry_keys) <= 1:
                continue
            tiers = {entries_by_key[k].match for k in entry_keys}
            if "section" in tiers and len(tiers) == 1:
                advice = "All colliding entries are already at Tier 2; escalate one to Tier 3 with explicit regex."
            else:
                advice = "Tier 1 collision; escalate at least one entry to Tier 2 with section_anchor."
            self._fail(
                f"COLLISION at ({rel}, anchor={anchor!r}, token={token!r}): "
                f"entries {sorted(entry_keys)}. {advice}"
            )

    # Pass 3 — Existence check ------------------------------------------------
    def pass3_existence_check(self) -> None:
        for entry in self.entries:
            for rel in entry.files:
                try:
                    text = self._file_text(rel)
                except FileNotFoundError:
                    continue  # Already reported in Pass 1
                if entry.match == "token":
                    haystack = text
                    count = count_token_matches(haystack, entry.rendered_value)
                elif entry.match == "section":
                    idx = self._section_index_for(rel)
                    sec = idx.get(entry.section_anchor)
                    if sec is None:
                        continue  # Already reported in Pass 1
                    haystack = "\n".join(text.splitlines()[sec.start_line:sec.end_line])
                    count = count_token_matches(haystack, entry.rendered_value)
                else:  # regex
                    if entry.pattern is None:
                        continue
                    count = len(re.findall(entry.pattern, text))
                if count < entry.min_count:
                    rendered = entry.rendered_value if entry.match != "regex" else f"pattern={entry.pattern!r}"
                    src = entry.source_artifact or "(no source recorded)"
                    self._fail(
                        f"[{entry.key}] expected '{rendered}' to appear "
                        f">= {entry.min_count} time(s) in {rel} (match={entry.match}"
                        + (f", anchor={entry.section_anchor!r}" if entry.match == "section" else "")
                        + f"); found {count}. Source: {src}"
                    )

    # Pass 4 — Artifact provenance --------------------------------------------
    def pass4_artifact_provenance(self) -> None:
        for entry in self.entries:
            if not entry.verifier:
                continue
            verifier_fn = VERIFIERS.get(entry.verifier)
            if verifier_fn is None:
                self._fail(f"[{entry.key}] verifier {entry.verifier!r} not registered in VERIFIERS")
                continue
            try:
                ok, detail = verifier_fn(entry.raw_value)
            except Exception as e:
                self._fail(
                    f"[{entry.key}] verifier {entry.verifier!r} raised {type(e).__name__}: {e}"
                )
                continue
            if not ok:
                self._fail(f"[{entry.key}] artifact provenance failed: {detail}")

    def validate(self) -> list[str]:
        self.pass1_schema_and_anchor_resolution()
        # If schema is malformed, Pass 2/3/4 may explode on rendered_value.
        # Filter to schema-valid entries before proceeding.
        schema_failures = [f for f in self.failures
                           if "invalid match type" in f or "requires" in f]
        if not schema_failures:
            self.pass2_collision_detection()
            self.pass3_existence_check()
            self.pass4_artifact_provenance()
        return self.failures


# ── Pytest entry point ────────────────────────────────────────────────────


def test_manuscript_values():
    """All-passes-accumulate validator: report every failure, not just the first."""
    if not MANIFEST_PATH.exists():
        pytest.skip(f"Manifest not present: {MANIFEST_PATH}")
    validator = ManifestValidator(MANIFEST_PATH, ROOT)
    failures = validator.validate()
    if failures:
        msg = ["MANIFEST validation failed:", ""]
        for i, f in enumerate(failures, 1):
            msg.append(f"  {i}. {f}")
        msg.append("")
        msg.append(f"Total failures: {len(failures)}")
        msg.append("Manifest: " + str(MANIFEST_PATH))
        msg.append("Fix the manifest (or the manuscript / artifact it points at) and re-run.")
        pytest.fail("\n".join(msg))


def test_section_index_hierarchy():
    """Pin the hierarchical scope rule end_line = first subsequent header at level <= L.

    Constructs a synthetic markdown with nested sections and asserts:
      - Parent anchor `3.4` window includes all child subsections
      - Child anchor `3.4.2` window is just that subsection
      - No bleed across parent boundaries (3.4's window does not include 3.5)
    """
    md = "\n".join([
        "## 3.4 Parent",       # line 0
        "Parent preamble.",    # line 1
        "",                    # line 2
        "### 3.4.1 First",     # line 3
        "First body.",         # line 4
        "",                    # line 5
        "### 3.4.2 Second",    # line 6
        "Second body.",        # line 7
        "",                    # line 8
        "## 3.5 Sibling",      # line 9
        "Sibling body.",       # line 10
    ])
    idx = build_section_index(md)
    assert "3.4" in idx and "3.4.1" in idx and "3.4.2" in idx and "3.5" in idx

    # Parent window covers preamble + both subsections, ends at sibling.
    assert idx["3.4"].start_line == 0
    assert idx["3.4"].end_line == 9, f"parent should end at sibling header (line 9), got {idx['3.4'].end_line}"

    # Child window is bounded by the next H3.
    assert idx["3.4.1"].start_line == 3
    assert idx["3.4.1"].end_line == 6, f"3.4.1 should end at 3.4.2 (line 6), got {idx['3.4.1'].end_line}"

    assert idx["3.4.2"].start_line == 6
    assert idx["3.4.2"].end_line == 9, f"3.4.2 should end at sibling H2 (line 9), got {idx['3.4.2'].end_line}"

    # Sibling window is independent of parent.
    assert idx["3.5"].start_line == 9
    assert idx["3.5"].end_line == 11  # EOF


def test_count_token_matches_boundary():
    """Custom boundary regex must reject [\\d.] on either side."""
    assert count_token_matches("the value is 31 today", "31") == 1
    assert count_token_matches("0.31 percent", "31") == 0, "31 should not match inside 0.31"
    assert count_token_matches("ratio 31.5", "31") == 0, "31 should not match inside 31.5"
    assert count_token_matches("31, 31, and 31 again", "31") == 3
    # Whole numbers vs decimals
    assert count_token_matches("0.05 alpha and 0.057 elsewhere", "0.05") == 1
    assert count_token_matches("alpha 0.05.", "0.05") == 1, "trailing period should not block match"
