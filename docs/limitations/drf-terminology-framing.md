# DRF Terminology and Reviewer Triggers

## The flaw

We cite **Ghodsi et al. (2011)** and **Dominant Resource Fairness (DRF)** to justify the Multi-Jain Bottleneck approach. DRF is an **allocation mechanism**: it takes user demands and outputs resource bundles, and guarantees properties like strategy-proofness and envy-freeness in that setting. A board game map is a **fixed spatial topology**, not a dynamic allocation mechanism. Leading with "DRF" and "Ghodsi et al." can trigger economics or CS reviewers who treat this as a misapplication of mechanism design, even when we add caveats.

## The fix (framing, not a limitation of the study)

The math is fine; the issue is **presentation**. Reframing avoids the trigger without changing the study.

### 1. Rename the metric in the main text

Use a self-contained name first:

- **"Bottleneck Multi-Dimensional Fairness"** or **"Max-Min JFI"**

Define it as: JFI computed per dimension (Resources, Influence), then take the minimum. No need to mention DRF in the definition sentence.

### 2. Move Ghodsi et al. to inspiration, not definition

- **Avoid:** *"This is inspired by Dominant Resource Fairness (Ghodsi et al., 2011): map fairness is limited by the least fair resource dimension."*
- **Prefer:** *"Map fairness is limited by the least-fair resource dimension. The same bottleneck intuition appears in multi-resource allocation (Ghodsi et al., 2011), though that work concerns allocation mechanisms, not fixed topologies."*

Define the metric on its own; then one short "related intuition" sentence with the citation.

### 3. Soften or replace the "Relationship to DRF" callout

The sentence *"DRF (Ghodsi et al., 2011) guarantees envy-freeness and strategy-proofness…"* is what triggers mechanism-design reviewers.

- **Option A:** Remove the callout and keep a single in-text sentence (as in §2).
- **Option B:** Replace with: *"The bottleneck principle (fairness bounded by the most constrained dimension) is analogous to the intuition in multi-resource allocation (Ghodsi et al., 2011). We do not claim formal allocation-theoretic properties (e.g. strategy-proofness or envy-freeness), which apply to mechanisms with demands, not to fixed map topologies."*

Never state that DRF "guarantees" anything in our setting; only "analogous intuition" plus clear scope.

### 4. Lit review / related work

For the paper, shorten the DRF subsection to: *"We use a **bottleneck JFI** across resource dimensions (max-min JFI). The idea that fairness is limited by the worst-off dimension is familiar from multi-resource allocation (Ghodsi et al., 2011); we use it only as motivation for our metric, not as an allocation mechanism."*

## Is this a limitation of the study?

**No.** The limitation is **terminology and framing**, not the design:

- The object is well-defined: JFI on R and I separately, then min.
- The "fairness bounded by the worst dimension" idea is general; DRF is one formalization in allocation settings.
- Our caveats (fixed topology, no allocation mechanism) are correct; the risk is that reviewers stop at "DRF" and never read the caveats.

**Summary:** Rename to Bottleneck Multi-Dimensional Fairness or Max-Min JFI, define it without DRF, cite Ghodsi et al. only as inspiration/analogy, and avoid stating that DRF "guarantees" anything in our context. That addresses the flaw without changing the study.
