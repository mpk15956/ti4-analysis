# Tabu Search Exclusion Justification

**For insertion into the paper's Methodology / Algorithm Selection section.**

---

## Why Tabu Search is excluded from the benchmark

Grichshenko et al. (2020) demonstrated that Tabu Search (TS) outperforms steepest-ascent hill-climbing for map generation in *Terra Mystica*, a related tabletop game with a hexagonal spatial layout. TS escapes local optima via a deterministic tabu list that prevents the algorithm from revisiting recently explored configurations, enabling non-improving moves that a greedy hill-climber would reject.

Our SA implementation addresses the same local-optima escape mechanism through a complementary stochastic pathway: the Metropolis acceptance criterion `P(accept) = exp(-Δ/T)` probabilistically accepts worsening moves at a rate controlled by temperature. With `T₀` calibrated to an 80% initial acceptance rate (Section X.Y), SA permits approximately four out of five uphill moves during early exploration — functionally equivalent to TS's deterministic acceptance of non-improving moves, but without requiring tuning of tabu tenure or neighbourhood structure. As temperature decays geometrically toward `min_temp`, SA smoothly transitions from broad exploration to greedy refinement within a single control parameter (the iteration budget), whereas TS requires separate tuning of tabu list length, aspiration criteria, and neighbourhood size.

Critically, the comparison reported by Grichshenko et al. was TS versus steepest-ascent hill-climbing — not TS versus SA. Steepest-ascent HC cannot escape any local optimum and serves as a weaker baseline than the SA implemented here. No published evidence demonstrates that TS produces superior map configurations to SA on combinatorial tile-placement problems when both algorithms receive the same evaluation budget.

We therefore exclude TS from the primary benchmark and retain it as a candidate for future comparative work. Should reviewers require a direct comparison, TS is implementable as a variant of our existing HC module by augmenting the swap-acceptance loop with a bounded deque of recently visited (s₁, s₂) pairs.

### References

- Grichshenko, A., Lankveld, G. van, & Spronck, P. (2020). Using Tabu Search Algorithm for Map Generation in the Terra Mystica Tabletop Game. *Proceedings of the 4th International Conference on Intelligent Systems, Metaheuristics & Swarm Intelligence (ISMSI)*, 66–70. ACM.
- Glover, F. (1989). Tabu search — Part I. *ORSA Journal on Computing*, 1(3), 190–206.
- Glover, F. (1990). Tabu search — Part II. *ORSA Journal on Computing*, 2(1), 4–32.
