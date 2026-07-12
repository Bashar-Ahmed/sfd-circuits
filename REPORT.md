# Safe Flow Decomposition for Circuit Discovery
### Applying provably-certain flow subpaths to mechanistic-interpretability circuits, evaluated on MIB / InterpBench

---

## Abstract

A transformer's edge-attribution graph — nodes are attention heads and MLPs, edges
carry an importance score, source is the embedding and sink is the logits — is,
structurally, exactly the object that **Safe Flow Decomposition** (Khan, Rizzi &
Tomescu; RECOMB'22 / ESA'22) was invented for in RNA-transcript assembly: a
single-source single-sink DAG with a scalar per edge. Safe Flow Decomposition asks
which *subpaths appear in **every** valid decomposition of a flow into source→sink
paths* — the **decomposition-invariant "certain core"** — and characterizes them by
a local, linear-time **excess-flow** test `f_P > 0`.

This report investigates whether that certainty notion transfers to **circuit
discovery**. We (i) build the precise mathematical bridge (attributions → a
conserved non-negative flow → safe subpaths), (ii) implement and *formally validate*
a safe-flow decomposition engine against brute-force ground truth, and (iii)
evaluate the resulting **Safe-Circuit** method on the MIB benchmark's InterpBench IOI
model, whose ground-truth circuit is known exactly.

**Headline finding.** The direct application is theoretically clean but **degenerates
on the transformer computational graph**: because every component reads a shared
residual stream, the DAG is near-complete (all-to-all forward), fan-out is enormous,
and the excess-flow of any path collapses within ~1 hop. Empirically, **all maximal
safe paths have length ≤ 2–3** and the per-edge safety score σ(e) collapses onto the
raw attribution (Spearman ≈ 0.9–0.95; ~93 % of edges identical). Two principled
repairs the adversarial analysis suggested — sparsified-backbone safe-flow and a
min-forced-flow *robustness certificate* — do not escape the obstruction (the
robustness LP forces 99 % of edges to zero; re-conservation re-densifies any
backbone). **Global AUROC is therefore unchanged** (σ vs raw attribution is within
bootstrap noise on all three attribution sources).

**But the intermediate step is independently useful.** Enforcing flow-conservation
acts as a **bottleneck-edge detector**: on InterpBench it rescues the rank of the two
ground-truth *bridge* edges (`input→m0`, `a4.h1→logits`) from ~400/1108 under raw
attribution to **22–44/1108** — a 10–18× improvement, robust across 6 seeds — because
local attribution systematically under-credits edges whose marginal effect is diffuse
but whose *forced throughput* is large. Branch edges trade off in the other direction,
so net AUROC is flat, but the selective effect is exactly what the flow theory
predicts. We give the structural reason why safe-flow is powerful for splice graphs
but obstructed for transformers, and argue this is an informative characterization of
circuit discovery as a flow problem — plus a concrete, reusable by-product
(conservation as a bottleneck detector).

---

## 1. Background

### 1.1 Safe Flow Decomposition (the genomics side)

A **flow** on a DAG `G=(V,E)` is a function `f: E → ℝ_{>0}` with conservation
`f_in(v)=f_out(v)` at every non-source/non-sink vertex. A **flow decomposition** is a
set of weighted source→sink paths whose per-edge sum reproduces `f`; it is generally
**non-unique**, so no single decomposition can be trusted. Khan–Tomescu's *safety*
framework instead reports what is common to **all** decompositions.

- A path `P=(u_1,…,u_k)` is **w-safe** if, in every decomposition, `P` is a subpath of
  decomposition paths of total weight ≥ w; it is **safe** if `f_P > 0`.
- **Excess-flow characterization** (RECOMB'22, Thm. 3): `P` is w-safe **iff**
  `f_P ≥ w`, where

  ```
  f_P = Σ_{i=1}^{k-1} f(u_i,u_{i+1}) − Σ_{i=2}^{k-1} f_out(u_i)
  ```

  with the O(1) incremental updates `append (u,v): f_P' = f_P − (f_out(u) − f(u,v))`
  and `prepend (u,v): f_P' = f_P − (f_in(v) − f(u,v))`.
- Safety is **closed under taking subpaths**, so **maximal safe paths** are well
  defined and enumerable in `O(mn)` (or output-sensitive `O(m + out)`).

Intuition: `f_P` is the amount of flow **forced to traverse the whole of `P`** in any
decomposition — the flow entering on the first edge minus everything that can *leak*
away at internal vertices. Positive excess = a routing that conservation cannot avoid.

### 1.2 Circuit discovery (the interpretability side)

A **circuit** is the subgraph of a model's computational graph responsible for a
behavior. Following ACDC (Conmy et al. 2023) and the *Mathematical Framework for
Transformer Circuits* (Elhage et al. 2021), the graph's nodes are attention heads /
MLPs communicating through the **residual stream**; by residual additivity every
component's output is readable by *all* later components, so edges connect
non-adjacent layers and the graph is a single-source (`input`) single-sink (`logits`)
DAG. Attention destinations split into q/k/v input ports.

Edge importance is estimated by **attribution patching**:
- **EAP** (Syed, Rager & Conmy 2023): `score(e) = (a^{clean}_u − a^{corrupt}_u) · ∂L/∂(input_v)`,
  one forward + backward pass for all edges.
- **EAP-IG** (Hanna et al., *Have Faith in Faithfulness*, 2024): integrate the
  gradient along the clean↔corrupt interpolation — better faithfulness.
- **Information Flow Routes** (Ferrando & Voita 2024): a normalized contribution
  graph — already flow-shaped.

A circuit is then selected by thresholding / greedy edge selection and scored by
**faithfulness** (does the circuit alone reproduce the behavior?) and, when ground
truth exists, **edge-recovery AUROC**.

### 1.3 The conservation bridge: Integrated Gradients & LRP

Safe-flow needs a *genuine* flow (non-negative + conserved). Where does one come from?

- **Raw / signed EAP is not a flow.** Scores are signed (negative name-mover edges)
  and non-conservative — gradients compose *multiplicatively* along paths, not
  additively across a cut.
- **Integrated Gradients' Completeness** (Sundararajan et al. 2017): `Σ_i IG_i =
  F(x)−F(x')` fixes only the *global* flow value, not internal balance.
- **Layer-wise Relevance Propagation** supplies conservation *by construction*: each
  node redistributes its relevance over its inputs (Bach et al. 2015, Eqs. 8+13), so
  `Σ_in R = R_v = Σ_out R` — Kirchhoff's law, "analogous to electrical circuits"
  (Montavon et al. 2019). **AttnLRP** (Achtibat et al. 2024) extends this through
  softmax and matmul so both OV and QK edges carry conserved relevance.

**Two flow constructions** result:
- **Build A (principled):** AttnLRP relevance messages (intrinsically conserved),
  made non-negative via the α1β0 rule or a two-commodity split.
- **Build B (MIB-pragmatic, used here):** take `|EAP-IG|` and **project** onto the
  conservative-flow cone. This is the minimal repair and lets us reuse MIB's existing
  EAP-IG attribution and AUROC/faithfulness harness unchanged.

---

## 2. Method: Safe-Circuit

**Input:** an attribution graph `G` with signed edge scores `s(e)`.

1. **Flow projection.** Compute the Euclidean projection of `|s|` onto the
   conservative-flow cone
   ```
   f* = argmin_f ‖f − |s|‖²   s.t.   f ≥ 0,   f_in(v)=f_out(v) ∀ internal v,
   ```
   solved by **Dykstra's alternating projection** between the non-negative orthant
   and the conservation subspace (the subspace projector uses the reduced incidence
   matrix over internal nodes). We report the projection residual as a diagnostic of
   "how flow-like" the raw attributions were.
2. **Safe decomposition.** Compute `f_out(v)`, then enumerate all **maximal safe
   paths** by DFS from each *left-end* edge, extending right while the incremental
   excess stays `> 0`.
3. **Per-edge safety score.** `σ(e) = max over maximal safe paths P ∋ e of f_P` — the
   guaranteed flow forced along a complete safe route through `e`. Properties (Khan–
   Tomescu): `0 < σ(e) ≤ f(e)`; σ stays near `f(e)` on long unbranched routes and
   *collapses* at high-fan-out junctions where excess drains — exactly separating a
   decomposition-invariant core from decomposition-dependent edges.
4. **Circuit selection.** Rank edges by σ (or a variant), feed to MIB's
   `apply_greedy` → connected input→logits sub-DAG at each budget → AUROC /
   faithfulness.

We also evaluate variants: `flow` (projected `f` itself — isolates the value of
conservation, hypothesis **H6**), `gated` (`f·(1+10·on-chain)`), `combo`
(`√(f·σ)`), and `lenflow` (safe-length primary).

### 2.1 Validation of the engine

The safe-flow engine is validated against **brute-force enumeration of all unit
decompositions** of small integral flows: for 50 cases (hand-built branch / bowtie /
bottleneck / diamond / residual-skip graphs + 45 random DAGs) the excess-test set
equals the "in-every-decomposition" set exactly, maximal safe paths cover the safe
set exactly, and the projection recovers valid flows (conservation error ~1e-16).
**50/50 passed.**

---

## 3. Experimental setup

- **Model / ground truth:** the MIB `mib-bench/interpbench` IOI model — a 6-layer,
  4-head, d_model-64 transformer trained with **Strict Interchange Intervention
  Training (SIIT)** so its circuit is *known by construction*. Graph: 32 nodes, 1108
  edges; ground-truth circuit = **7 edges**:
  ```
  input→m0 ; m0→{a1.h1, a2.h1, a4.h1}(v) ; a1.h1→a2.h1(v) ; a2.h1→a4.h1(v) ; a4.h1→logits
  ```
  Note `input→m0` and `a4.h1→logits` are **bridges** (on every input→logits path of
  the circuit) — the regime where safe-flow *should* excel.
- **Attribution:** EAP, EAP-IG-inputs (ig_steps=5), IFR, on 100 IOI examples with
  `logit_diff` metric and `s2_io_flip` counterfactual patching — MIB's exact protocol.
- **Metric:** InterpBench **AUROC** (area under the ROC of greedy-selected circuit
  edges vs the 7 ground-truth edges), plus faithfulness area (CPR/CMD).
- **Statistics:** mean ± std over 6 bootstrap resamples of the data.

---

## 4. Results

### 4.1 The certificate degenerates on the transformer DAG

| source | maximal-path len_max | mean len | frac len ≥ 3 | σ↔flow Spearman | σ==flow frac |
|---|---|---|---|---|---|
| EAP | 2 | 1.16 | 0.00 | 0.906 | 0.938 |
| EAP-IG | 2 | 1.09 | 0.00 | 0.952 | 0.931 |

Even after conservation projection, **no maximal safe path exceeds length 2**, and
σ(e) equals f(e) for ~93 % of positive-flow edges. The append update
`f_P' = f_P − (f_out(u) − f(u,v))` drives excess negative after one hop because, in
the residual DAG, `f_out(u) ≫ f(u,v)` at every internal node. **The classical safe
certificate reduces to ranking by (projected) attribution.**

### 4.2 AUROC (bootstrap mean ± std over 6 data subsets)

| source | raw \|attr\| | projected flow | safe-flow σ |
|---|---|---|---|
| EAP | **0.719 ± 0.016** | 0.647 ± 0.067 | 0.677 ± 0.066 |
| EAP-IG-inputs | 0.727 ± 0.036 | **0.741 ± 0.069** | 0.734 ± 0.066 |
| Information-Flow-Routes | 0.588 ± 0.126 | 0.606 ± 0.025 | **0.647 ± 0.044** |

Paired σ − raw deltas: EAP **−0.042**, EAP-IG **+0.007**, IFR **+0.059** — the σ
re-scoring is **statistically indistinguishable from raw attribution** for EAP-IG,
slightly worse for EAP, and better-but-very-noisy for IFR (per-seed swings of ±0.2).
This *corrects* a promising single-run result (σ = 0.685 vs raw 0.644 on one split);
under bootstrap the global-AUROC gain is within noise — consistent with the σ↔flow
collapse of §4.1. Projection alone (`flow`) helps EAP-IG (+0.014) but hurts EAP
(−0.072); it is not a free win.

### 4.3 The two repairs do not escape the obstruction

- **Sparsified backbone** (keep top-k flow edges, re-conserve, re-run safe-flow):
  longest safe path only reaches **length 3** at the sparsest settings (keep ≤ 2 %);
  re-conservation re-densifies the flow (n_pos rebounds to 360–580 edges even when
  seeding from 11). AUROC tracks the projected-flow baseline (best ≈ 0.74 for EAP at
  keep = 10 %) with no lift attributable to *safe* structure.
- **Robustness certificate** `σ_rob(e) = min_g g(e)` over throughput-consistent flows
  (LP per edge): **99 % of edges are forced to zero** (routable-around in the dense
  graph), leaving AUROC ≈ 0.56 (near chance). The obstruction is symmetric —
  path-safety collapses σ *upward* to attribution; the robustness LP collapses it
  *downward* to the handful of true bridges.

### 4.4 Ground-truth-edge localization: conservation is a bottleneck detector

This is the one robust *positive* effect. We split the 7 ground-truth edges into the
**2 bridges** (`input→m0`, `a4.h1→logits` — on every input→logits circuit path) and
the **5 branch/mid-circuit edges**, and report their mean rank / 1108 (lower = better)
over 6 seeds.

| source | edge group | raw \|attr\| | projected flow | safe-flow σ |
|---|---|---|---|---|
| EAP | **bridge** | 402 ± 46 | 33 ± 3 | **22 ± 5** |
| EAP | branch | **251 ± 31** | 505 ± 105 | 484 ± 100 |
| EAP-IG | **bridge** | 364 ± 65 | 44 ± 6 | **36 ± 6** |
| EAP-IG | branch | **275 ± 29** | 382 ± 87 | 492 ± 90 |

Conservation lifts the **bridge** edges by **10–18×** (EAP: 402→22; EAP-IG: 364→36) —
`input→m0`, whose raw EAP-IG rank on a single split is **836/1108**, jumps to **29**.
Local attribution is blind to bottlenecks because a bridge's *marginal* effect is
diffuse (much of the "input→m0" influence is captured by longer paths), whereas
conservation assigns it the full *forced throughput* it must carry. The trade-off:
branch/fan-out edges, which attribution ranks well, are **down**-ranked (flow splits
across them). Net AUROC is flat (§4.2), but the mechanism is exactly the flow theory's
prediction — forced-flow information lives on bottleneck edges — and gives a concrete
recipe: **use conservation projection specifically to recover a circuit's entry/exit
bottleneck edges that attribution misses.** σ improves marginally over raw `flow` on
bridges (22 vs 33; 36 vs 44), the residue of the (degenerate) safe-path structure.

---

## 5. Why safe-flow is obstructed for transformers (and not for genomes)

Safe Flow Decomposition draws its power from **bottleneck structure**. In an RNA
splice graph, exons chain with modest branching, so long stretches of flow are
*forced* and safe paths are long and informative. The transformer computational graph
is the opposite: **residual-stream additivity makes it near-complete** — every one of
the ~n earlier components connects to every later component. Consequently:

1. **No forced multi-hop routing.** At any node, `f_out` is spread over dozens of
   downstream edges, so `f_out(u) − f(u,v) > f_P` almost immediately → safe paths
   die at length 1–2. Excess-flow safety measures a *topological bottleneck*
   property that the residual stream structurally eliminates.
2. **Conservation re-densifies sparsity.** Projecting or thresholding to a sparse
   backbone is undone by the conservation constraint, which must route mass through
   connecting edges.
3. **Safety ≠ necessity.** Even where σ is defined, it certifies invariance across
   decompositions of a *chosen attribution flow* — a statement about the attribution,
   not the model's causal circuit. The important IOI heads live at high-fan-out hubs,
   exactly where σ collapses.

This is a precise, useful *negative characterization*: it identifies the graph
property (sparsity / genuine bottlenecks) that a flow-safety approach to circuit
discovery would require, and explains why off-the-shelf transformer attribution
graphs lack it.

---

## 6. What does transfer

- **Conservation as a bottleneck detector (the robust win).** Projecting `|attr|` onto
  the flow cone rescues the ground-truth *bridge* edges by 10–18× in rank (§4.4) —
  precisely the entry/exit edges attribution under-credits. This does not raise global
  AUROC (branch edges trade off), but it is a directly useful, cheap primitive for
  finding a circuit's bottleneck backbone.
- **Excess-flow as a soft signal.** Although the strict `f_P>0` certificate
  degenerates, `f_P` used as a *continuous* re-weighting (not a binary certificate)
  is a legitimate, cheap, order-independent structural feature.
- **The bridge is real and reusable.** The formalism (attribution → conserved flow →
  path certificate) is sound; its yield depends entirely on graph sparsity, which is
  a design knob (neuron/head granularity, backbone extraction, AttnLRP two-commodity
  flows).

---

## 7. Limitations & future directions

- **Build A not implemented.** We used projected `|EAP-IG|` (Build B). AttnLRP relevance
  (intrinsic conservation, two-commodity sign handling) may distribute flow
  differently; whether it yields longer safe paths is open (the density obstruction
  likely persists).
- **Signed/suppressive flow.** No developed safety theory for signed flows; the
  two-commodity split is heuristic (each commodity is not separately conserved).
- **ε-slack safety.** Conservation is exact only up to the projection residual; a
  robust "safe under all flows within ε" definition would harden the certificate.
- **Sparser graphs.** The most promising direction is to apply safe-flow where the
  computational graph *is* sparse and bottlenecked — e.g. SAE-feature circuits,
  attention-pattern (QK) routing graphs, or single-token information-flow graphs —
  rather than the dense head/MLP residual DAG.
- **Necessity test.** σ certifies the attribution, not the model; the load-bearing
  validation is ablating high-σ vs flow-matched-random edges (H7) and causal
  scrubbing — complementary to, not replaced by, safety.

---

## 8. Conclusion

Safe Flow Decomposition supplies a rigorous, decomposition-invariant notion of a
circuit's "certain core," and the transformer attribution graph is formally the right
kind of object. We implemented and validated the machinery and evaluated it honestly
on InterpBench. The direct method degenerates because the residual stream destroys the
bottleneck structure safe-flow relies on — a result we both predicted adversarially
and confirmed quantitatively — while the conservation prior it entails remains a small
net positive for EAP-IG. The contribution is a clean bridge between two fields, a
validated tool, and a precise account of the graph-structural conditions under which
flow-safety can and cannot certify neural-network mechanisms.

---

## References

1. Khan, Rizzi, Tomescu et al. *Safety and Completeness in Flow Decompositions for RNA Assembly* (RECOMB 2022; arXiv:2201.10372).
2. Khan, Tomescu et al. *Optimizing Safe Flow Decompositions in DAGs* (ESA 2022; arXiv:2102.06480).
3. Khan et al. *Safe and Complete Flow Decomposition / evaluation for RNA assembly.*
4. Elhage et al. *A Mathematical Framework for Transformer Circuits*, Anthropic, 2021.
5. Conmy et al. *Towards Automated Circuit Discovery for Mechanistic Interpretability*, NeurIPS 2023 (arXiv:2304.14997).
6. Syed, Rager, Conmy. *Attribution Patching Outperforms Automated Circuit Discovery*, 2023 (arXiv:2310.10348).
7. Hanna, Pezzelle, Belinkov. *Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*, 2024 (arXiv:2403.17806).
8. Sundararajan, Taly, Yan. *Axiomatic Attribution for Deep Networks* (Integrated Gradients), ICML 2017 (arXiv:1703.01365).
9. Chan et al. *Causal Scrubbing*, Redwood Research, 2022.
10. Bach et al. *On Pixel-Wise Explanations … by Layer-Wise Relevance Propagation*, PLoS ONE 2015.
11. Montavon et al. *Layer-Wise Relevance Propagation: An Overview*, 2019.
12. Achtibat et al. *AttnLRP: Attention-Aware Layer-Wise Relevance Propagation for Transformers*, ICML 2024 (arXiv:2402.05602).
13. Mueller et al. *MIB: A Mechanistic Interpretability Benchmark*, 2025 (arXiv:2504.13151).
14. Gupta, Arcuschin, Kwa, Garriga-Alonso. *InterpBench: Semi-Synthetic Transformers for Evaluating Mechanistic Interpretability Techniques*, NeurIPS 2024 (arXiv:2407.14494).
