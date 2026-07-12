"""
Cross-Input Stability Analysis: Safe-Flow vs SOTA Circuit Discovery Methods.

Measures how consistently each method ranks edges across different data subsets.
A method that finds the same bottleneck edges regardless of which examples it sees
is more trustworthy than one whose rankings fluctuate with the sample.

Stability dimensions tested:
  1. Split-to-split: pairwise Spearman ρ + top-k Jaccard across 8 random splits
  2. Size convergence: how rankings change as data grows (20→50→100→200 examples)
  3. Bottleneck focus: coefficient of variation of bridge-edge ranks
  4. Per-edge score variance: which method's scores are most stable

Methods compared:
  - EAP raw |score|         (SOTA baseline, AUROC 0.78)
  - EAP-IG raw |score|      (SOTA baseline, AUROC 0.71)
  - EAP flow                (our conservation projection)
  - EAP σ                   (our safe-flow sigma)
  - EAP-IG flow
  - EAP-IG σ
"""
import os, sys, json, time
from functools import partial
import numpy as np
from collections import defaultdict

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))

from common import (
    load_interpbench_model, load_reference_graph, get_metric, HFEAPDataset,
    run_attribution
)
from safeflow_eap import safe_flow_pipeline

OUT = "/workspace/sfd-circuits/artifacts"
os.makedirs(OUT, exist_ok=True)

BRIDGES = ["input->m0", "a4.h1->logits"]
ALL_GT = ["input->m0", "m0->a1.h1<v>", "m0->a2.h1<v>", "m0->a4.h1<v>",
          "a1.h1->a2.h1<v>", "a2.h1->a4.h1<v>", "a4.h1->logits"]


def dataloader_split(model, seed, num_examples=100, batch_size=50):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="interpbench", num_examples=None)
    n = len(ds.dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_examples, n), replace=False).tolist()
    ds.dataset = ds.dataset.select(idx)
    return ds.to_dataloader(batch_size=batch_size)


def compute_stability_metrics(all_scores, edge_names):
    """
    Given a list of per-split score dicts, compute stability metrics.
    all_scores: list of dicts [{edge_name: score}] — one per split
    """
    n_splits = len(all_scores)
    E = len(edge_names)

    # Build score matrix: [n_splits × E]
    score_mat = np.zeros((n_splits, E))
    for i, scores in enumerate(all_scores):
        for j, name in enumerate(edge_names):
            score_mat[i, j] = scores.get(name, 0.0)

    # Build rank matrix
    rank_mat = np.zeros((n_splits, E), dtype=int)
    for i in range(n_splits):
        order = np.argsort(-score_mat[i])
        for r, j in enumerate(order):
            rank_mat[i, j] = r + 1

    # 1. Pairwise Spearman correlations
    spearman_pairs = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            # Spearman on ranks is equivalent to Pearson on ranks
            rho = np.corrcoef(rank_mat[i], rank_mat[j])[0, 1]
            spearman_pairs.append(rho)

    # 2. Top-k Jaccard overlap
    edge_arr = np.array(edge_names)
    topk_jaccard = {}
    for k in [10, 20, 50, 100, 200, 500]:
        overlaps = []
        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                set_i = set(edge_arr[rank_mat[i] <= k])
                set_j = set(edge_arr[rank_mat[j] <= k])
                jacc = len(set_i & set_j) / len(set_i | set_j)
                overlaps.append(jacc)
        topk_jaccard[str(k)] = (float(np.mean(overlaps)), float(np.std(overlaps)))

    # 3. Per-edge score coefficient of variation (CV = std/mean)
    cv_per_edge = np.zeros(E)
    for j in range(E):
        mean_s = np.mean(score_mat[:, j])
        std_s = np.std(score_mat[:, j])
        cv_per_edge[j] = std_s / (mean_s + 1e-12)

    # 4. Bridge-edge rank stability
    bridge_stability = {}
    for b in BRIDGES:
        if b in edge_names:
            idx = edge_names.index(b)
            ranks = rank_mat[:, idx]
            bridge_stability[b] = {
                "mean_rank": float(np.mean(ranks)),
                "std_rank": float(np.std(ranks)),
                "min_rank": int(np.min(ranks)),
                "max_rank": int(np.max(ranks)),
                "cv_rank": float(np.std(ranks) / (np.mean(ranks) + 1e-12)),
                "ranks": ranks.tolist(),
            }

    # 5. Top-20 edge stability: which edges appear in top-20 most consistently?
    top20_freq = defaultdict(int)
    for i in range(n_splits):
        top20 = set(edge_arr[rank_mat[i] <= 20])
        for e in top20:
            top20_freq[e] += 1
    most_stable_edges = sorted(top20_freq.items(), key=lambda x: -x[1])[:20]

    return {
        "n_splits": n_splits,
        "n_edges": E,
        "spearman_mean": float(np.mean(spearman_pairs)),
        "spearman_std": float(np.std(spearman_pairs)),
        "spearman_min": float(np.min(spearman_pairs)),
        "spearman_pairs": [float(x) for x in spearman_pairs],
        "topk_jaccard": topk_jaccard,
        "cv_per_edge_mean": float(np.mean(cv_per_edge)),
        "cv_per_edge_median": float(np.median(cv_per_edge)),
        "cv_per_edge_q90": float(np.percentile(cv_per_edge, 90)),
        "bridge_stability": bridge_stability,
        "most_stable_top20": [(e, int(f)) for e, f in most_stable_edges],
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("Cross-Input Stability Analysis")
    print("Safe-Flow vs SOTA Circuit Discovery Methods")
    print("=" * 72)

    model = load_interpbench_model()

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: Split-to-split stability (8 splits × 100 examples)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Test 1: Split-to-split stability (8 × 100 examples) ──")
    N_SPLITS = 8
    SOURCES = ["EAP", "EAP-IG-inputs"]
    VARIANTS = ["raw", "flow", "sigma"]
    METHODS = [(src, var) for src in SOURCES for var in VARIANTS]

    split_scores = {m: [] for m in METHODS}
    split_diags = {m: [] for m in METHODS}

    for split_idx in range(N_SPLITS):
        dl = dataloader_split(model, seed=split_idx * 10, num_examples=100)
        for src in SOURCES:
            g_attr = run_attribution(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g_attr, use_abs=True)
            for var in VARIANTS:
                split_scores[(src, var)].append(pipe["scorings"][var])
            if split_idx == 0:
                split_diags[(src, "raw")] = pipe["diag"]
        if split_idx % 2 == 0:
            print(f"  split {split_idx+1}/{N_SPLITS} done ({time.time()-t0:.0f}s)")

    edge_names = list(g_attr.edges.keys())
    edge_arr = np.array(edge_names)
    print(f"  All splits done ({time.time()-t0:.0f}s)")

    # Compute stability for each method
    stability_results = {}
    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        metrics = compute_stability_metrics(split_scores[method], edge_names)
        stability_results[label] = metrics

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Size convergence (20, 50, 100, 200 examples)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Test 2: Size convergence (20 → 50 → 100 → 200) ──")
    SIZES = [20, 50, 100, 200]
    size_scores = {m: {} for m in METHODS}

    for size in SIZES:
        dl = dataloader_split(model, seed=42, num_examples=size)
        for src in SOURCES:
            g = run_attribution(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g, use_abs=True)
            for var in VARIANTS:
                size_scores[(src, var)][size] = pipe["scorings"][var]
        print(f"  size={size} done ({time.time()-t0:.0f}s)")

    # Cross-size rank correlation: compare each size to the 200-example "reference"
    size_convergence = {}
    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        ref_scores = size_scores[method][200]
        ref_arr = np.array([ref_scores.get(n, 0.0) for n in edge_names])
        ref_order = np.argsort(-ref_arr)
        ref_rank = np.zeros(len(edge_names), dtype=int)
        for r, j in enumerate(ref_order):
            ref_rank[j] = r + 1

        conv = {}
        for size in SIZES[:-1]:  # compare each smaller size to 200
            scores = size_scores[method][size]
            arr = np.array([scores.get(n, 0.0) for n in edge_names])
            order = np.argsort(-arr)
            rank = np.zeros(len(edge_names), dtype=int)
            for r, j in enumerate(order):
                rank[j] = r + 1
            rho = np.corrcoef(ref_rank, rank)[0, 1]

            # Top-k overlap with reference
            top_overlaps = {}
            for k in [10, 20, 50, 100]:
                ref_set = set(edge_arr[ref_rank <= k])
                size_set = set(edge_arr[rank <= k])
                top_overlaps[str(k)] = len(ref_set & size_set) / k

            # Bridge rank delta from reference
            bridge_deltas = {}
            for b in BRIDGES:
                if b in edge_names:
                    idx = edge_names.index(b)
                    bridge_deltas[b] = int(ref_rank[idx] - rank[idx])

            conv[str(size)] = {
                "spearman_with_ref": float(rho),
                "topk_overlap": top_overlaps,
                "bridge_rank_delta": bridge_deltas,
            }
        size_convergence[label] = conv

    # ═══════════════════════════════════════════════════════════════════════
    # Print Results
    # ═══════════════════════════════════════════════════════════════════════

    # --- Split-to-split stability ---
    print(f"\n{'='*72}")
    print("TEST 1: Split-to-Split Stability (8 random 100-example splits)")
    print(f"{'='*72}")

    print(f"\n{'Method':<22s} {'Spearman ρ':>12s} {'Top-20 Jacc':>12s} "
          f"{'Top-50 Jacc':>12s} {'Top-100 Jacc':>12s} {'CV(score)':>10s} "
          f"{'Bridge CV(rank)':>15s}")
    print("-" * 90)

    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        st = stability_results[label]

        sp_mean = st["spearman_mean"]
        sp_std = st["spearman_std"]
        j20 = st["topk_jaccard"]["20"][0]
        j50 = st["topk_jaccard"]["50"][0]
        j100 = st["topk_jaccard"]["100"][0]
        cv_score = st["cv_per_edge_mean"]

        # Average bridge rank CV across the 2 bridges
        bridge_cv = np.mean([st["bridge_stability"][b]["cv_rank"]
                             for b in BRIDGES if b in st["bridge_stability"]])

        print(f"  {label:<20s} {sp_mean:>9.4f}±{sp_std:.3f} {j20:>11.3f} "
              f"{j50:>11.3f} {j100:>11.3f} {cv_score:>9.3f} {bridge_cv:>14.3f}")

    # Highlight best
    best_sp = max(stability_results.items(), key=lambda x: x[1]["spearman_mean"])
    best_j20 = max(stability_results.items(),
                   key=lambda x: x[1]["topk_jaccard"]["20"][0])
    print(f"\n  Most rank-stable:       {best_sp[0]} (ρ={best_sp[1]['spearman_mean']:.4f})")
    print(f"  Most top-20 stable:     {best_j20[0]} (Jacc={best_j20[1]['topk_jaccard']['20'][0]:.3f})")

    # --- Bridge-edge rank stability specifically ---
    print(f"\n{'='*72}")
    print("Bridge-Edge Rank Stability Across Splits")
    print(f"{'='*72}")
    for b in BRIDGES:
        print(f"\n  Edge: {b}")
        print(f"  {'Method':<22s} {'Mean rank':>10s} {'Std rank':>10s} "
              f"{'Min':>6s} {'Max':>6s} {'CV':>8s}")
        print(f"  {'-'*60}")
        for method in METHODS:
            src, var = method
            label = f"{src}+{var}"
            if b in stability_results[label]["bridge_stability"]:
                bs = stability_results[label]["bridge_stability"][b]
                print(f"  {label:<20s} {bs['mean_rank']:>9.0f} {bs['std_rank']:>9.0f} "
                      f"{bs['min_rank']:>5d} {bs['max_rank']:>5d} {bs['cv_rank']:>7.3f}")

    # --- Size convergence ---
    print(f"\n{'='*72}")
    print("TEST 2: Size Convergence (Spearman ρ vs 200-example reference)")
    print(f"{'='*72}")

    print(f"\n  {'Method':<22s} {'20→200':>10s} {'50→200':>10s} {'100→200':>10s}")
    print(f"  {'-'*55}")
    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        sc = size_convergence[label]
        r20 = sc["20"]["spearman_with_ref"]
        r50 = sc["50"]["spearman_with_ref"]
        r100 = sc["100"]["spearman_with_ref"]
        print(f"  {label:<20s} {r20:>9.4f} {r50:>9.4f} {r100:>9.4f}")

    # Which method converges fastest?
    print(f"\n  Convergence speed (ρ_100 - ρ_20, higher = faster convergence):")
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        sc = size_convergence[label]
        delta = sc["100"]["spearman_with_ref"] - sc["20"]["spearman_with_ref"]
        print(f"    {label:<20s} {delta:+.4f}")

    # --- Most stable edges (top-20 consistently) ---
    print(f"\n{'='*72}")
    print("Most Consistently Top-20 Edges Across Splits")
    print(f"{'='*72}")
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        stable = stability_results[label]["most_stable_top20"][:5]
        edges_str = ", ".join(f"{e}({f}x)" for e, f in stable)
        gt_marks = []
        for e, f in stable:
            marks = []
            if e in BRIDGES:
                marks.append("BRIDGE")
            if e in ALL_GT:
                marks.append("GT")
            gt_marks.append("+".join(marks) if marks else "")
        marks_str = "  ".join(f"{m:>12s}" for m in gt_marks)
        print(f"  {label}:")
        print(f"    {edges_str}")
        print(f"    {marks_str}")
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"{'='*72}")
    print("VERDICT: Cross-Input Stability")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability_results[label]
        checks = []

        # Rank stability
        if st["spearman_mean"] > 0.85:
            checks.append(f"high rank stability (ρ={st['spearman_mean']:.3f})")
        elif st["spearman_mean"] > 0.7:
            checks.append(f"moderate rank stability (ρ={st['spearman_mean']:.3f})")

        # Bridge stability
        bridge_cvs = [st["bridge_stability"][b]["cv_rank"]
                      for b in BRIDGES if b in st["bridge_stability"]]
        if bridge_cvs:
            avg_bcv = np.mean(bridge_cvs)
            if avg_bcv < 0.3:
                checks.append(f"very stable bridge ranks (CV={avg_bcv:.2f})")
            elif avg_bcv < 0.6:
                checks.append(f"stable bridge ranks (CV={avg_bcv:.2f})")

        # Top-k stability
        j50 = st["topk_jaccard"]["50"][0]
        if j50 > 0.5:
            checks.append(f"high top-50 overlap ({j50:.2f})")

        print(f"  {label:<20s}: {' | '.join(checks) if checks else 'no strong stability signal'}")

    # ── Save ──
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output = {
        "test": "cross_input_stability",
        "model": "interpbench_ioi",
        "n_edges_total": len(edge_names),
        "bridge_edges": BRIDGES,
        "all_gt_edges": ALL_GT,
        "split_stability": stability_results,
        "size_convergence": size_convergence,
    }
    with open(f"{OUT}/stability_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {OUT}/stability_analysis.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
