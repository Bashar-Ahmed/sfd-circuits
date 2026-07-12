"""
BENCHMARK: Bottleneck Edge Detection on InterpBench IOI.

Compares safe-flow against published SOTA circuit discovery methods
(EAP, EAP-IG, IFR) on the specific task where safe-flow theory predicts
it should excel: identifying bridge/bottleneck edges.

Ground truth: InterpBench IOI has 2 known bridge edges:
  - input→m0        (the MLP bottleneck: all circuit paths go through it)
  - a4.h1→logits    (the final head→output bottleneck)

Published SOTA baselines (from MIB/BlackboxNLP 2025):
  - EAP:              AUROC 0.78 (MIB paper)
  - EAP-IG-inputs:    AUROC 0.71 (MIB paper)
  - EAP-IG-activations: AUROC 0.81 (best single method)
  - IFR:              AUROC 0.71 (MIB paper)
  - Random:           AUROC 0.44 (lower bound)

Our methods:
  - Conservation projection (Dykstra): projects |attr| onto flow cone
  - Safe-flow σ:                 safe-path excess on projected flow
  - Combo:                       sqrt(flow × σ)

Metrics:
  - Mean Reciprocal Rank (MRR) of the 2 bridge edges
  - Recall@k (k = 5, 10, 20, 50, 100, 200)
  - Precision@k
  - Bootstrap statistics (6 seeds × 100 examples)
  - Ablation impact: Δ(metric) when removing top-ranked edges vs random
"""
import os, sys, json, time
from functools import partial
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))

from common import (
    load_interpbench_model, load_reference_graph, run_attribution,
    get_metric, HFEAPDataset, clone_graph, auroc_mib_raw
)
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.evaluation import evaluate_area_under_roc, compare_graphs
from eap.graph import Graph

OUT = "/workspace/sfd-circuits/artifacts"
os.makedirs(OUT, exist_ok=True)

# Ground truth bridge edges (on every circuit input→logits path)
BRIDGES = ["input->m0", "a4.h1->logits"]

# All ground truth circuit edges (for comparison)
ALL_GT = ["input->m0", "m0->a1.h1<v>", "m0->a2.h1<v>", "m0->a4.h1<v>",
          "a1.h1->a2.h1<v>", "a2.h1->a4.h1<v>", "a4.h1->logits"]

# Published SOTA AUROC scores (from MIB paper, BlackboxNLP 2025)
PUBLISHED_AUROC = {
    "EAP-IG-activations (best)": 0.81,
    "EAP": 0.78,
    "EAP-IG-inputs": 0.71,
    "IFR": 0.71,
    "UGS": 0.74,
    "NAP-IG": 0.62,
    "Random": 0.44,
}


def dataloader_seed(model, seed, num_examples=100, batch_size=50):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="interpbench", num_examples=None)
    n = len(ds.dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_examples, n), replace=False).tolist()
    ds.dataset = ds.dataset.select(idx)
    return ds.to_dataloader(batch_size=batch_size)


def compute_metrics(edge_names, scores, bridge_set, all_gt_set):
    """Compute bottleneck detection metrics for a scoring method."""
    E = len(edge_names)
    arr = np.array([scores.get(n, 0.0) for n in edge_names])
    order = np.argsort(-arr)
    ranked = [edge_names[order[i]] for i in range(E)]

    # MRR of bridge edges
    mrr = 0.0
    for b in bridge_set:
        rank = ranked.index(b) + 1 if b in ranked else E
        mrr += 1.0 / rank
    mrr /= len(bridge_set)

    # Recall@k
    ks = [5, 10, 20, 50, 100, 200]
    recall = {}
    precision = {}
    for k in ks:
        top_k = set(ranked[:k])
        recall[k] = len(top_k & bridge_set) / len(bridge_set)
        precision[k] = len(top_k & all_gt_set) / k if k > 0 else 0.0

    # AUROC for all GT edges (standard MIB metric)
    # Build a graph with these scores and compute AUROC
    return {
        "mrr": float(mrr),
        "recall": {str(k): float(v) for k, v in recall.items()},
        "precision": {str(k): float(v) for k, v in precision.items()},
        "bridge_ranks": {b: int(ranked.index(b) + 1) if b in ranked else E
                         for b in bridge_set},
    }


def main():
    t0 = time.time()
    np.random.seed(0)

    print("=" * 72)
    print("BENCHMARK: Bottleneck Edge Detection on InterpBench IOI")
    print("=" * 72)
    print(f"\nPublished SOTA AUROC scores (MIB/BlackboxNLP 2025):")
    for method, auc in sorted(PUBLISHED_AUROC.items(), key=lambda x: -x[1]):
        marker = " ← BEST" if auc == max(PUBLISHED_AUROC.values()) else ""
        print(f"  {method:35s} {auc:.2f}{marker}")
    print(f"\nBridge edges (ground truth): {BRIDGES}")
    print(f"All GT circuit edges: {len(ALL_GT)}")

    # Load model once
    model = load_interpbench_model()
    ref = load_reference_graph()

    N_SEEDS = 6
    SOURCES = ["EAP", "EAP-IG-inputs"]
    # Our methods: raw, projected flow, safe-flow sigma, combo
    METHOD_VARIANTS = ["raw", "flow", "sigma", "combo"]

    all_results = {src: {var: [] for var in METHOD_VARIANTS}
                   for src in SOURCES}
    ablation_results = {src: {var: [] for var in METHOD_VARIANTS}
                        for src in SOURCES}

    for seed in range(N_SEEDS):
        print(f"\n── Seed {seed+1}/{N_SEEDS} ──")
        dl = dataloader_seed(model, seed)

        for src in SOURCES:
            g_attr = run_attribution(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g_attr, use_abs=True)
            edge_names = list(g_attr.edges.keys())

            for var in METHOD_VARIANTS:
                scores = pipe["scorings"][var]
                metrics = compute_metrics(edge_names, scores,
                                          set(BRIDGES), set(ALL_GT))
                all_results[src][var].append(metrics)

            # Standard AUROC for reference
            for var in METHOD_VARIANTS:
                g_scored = build_scored_graph(g_attr.cfg, pipe["scorings"][var])
                auc, _ = auroc_mib_raw(ref, g_scored)
                all_results[src][var][-1]["auroc"] = float(auc)

            # Store diagnostics from first seed
            if seed == 0:
                diag = pipe["diag"]
                print(f"  {src}: σ_collapse={diag.get('sigma_collapse_frac',0):.3f} "
                      f"σ-vs-flow ρ={diag.get('spearman_sigma_flow',0):.3f} "
                      f"safe_len_max={diag.get('safe_len_max',0)}")

    # ── Aggregate results ──
    print(f"\n\n{'='*72}")
    print("RESULTS: Bottleneck Edge Detection Benchmark")
    print(f"{'='*72}")

    print(f"\n{'Method':<30s} {'MRR':>8s} {'Recall@10':>10s} "
          f"{'Recall@20':>10s} {'Recall@50':>10s} {'Recall@100':>10s} "
          f"{'AUROC':>8s}")
    print("-" * 80)

    summary = {}
    for src in SOURCES:
        for var in METHOD_VARIANTS:
            results = all_results[src][var]
            mrr_vals = [r["mrr"] for r in results]

            rec10_vals = [r["recall"]["10"] for r in results]
            rec20_vals = [r["recall"]["20"] for r in results]
            rec50_vals = [r["recall"]["50"] for r in results]
            rec100_vals = [r["recall"]["100"] for r in results]
            auroc_vals = [r.get("auroc", 0) for r in results]

            label = f"{src}+{var}"
            summary[label] = {
                "mrr": (np.mean(mrr_vals), np.std(mrr_vals)),
                "recall10": (np.mean(rec10_vals), np.std(rec10_vals)),
                "recall20": (np.mean(rec20_vals), np.std(rec20_vals)),
                "recall50": (np.mean(rec50_vals), np.std(rec50_vals)),
                "recall100": (np.mean(rec100_vals), np.std(rec100_vals)),
                "auroc": (np.mean(auroc_vals), np.std(auroc_vals)),
            }

            m = np.mean(mrr_vals)
            s = np.std(mrr_vals)
            r10 = np.mean(rec10_vals)
            r20 = np.mean(rec20_vals)
            r50 = np.mean(rec50_vals)
            r100 = np.mean(rec100_vals)
            a_m, a_s = np.mean(auroc_vals), np.std(auroc_vals)
            print(f"  {label:<28s} {m:>7.4f}±{s:.3f} {r10:>9.2f} "
                  f"{r20:>9.2f} {r50:>9.2f} {r100:>9.2f} "
                  f"{a_m:>7.3f}±{a_s:.3f}")

    # ── Highlight the best methods ──
    print(f"\n── Key comparisons ──")

    # Best MRR overall
    best_mrr = max(summary.items(), key=lambda x: x[1]["mrr"][0])
    print(f"  Best MRR:       {best_mrr[0]} ({best_mrr[1]['mrr'][0]:.4f})")

    # Best bottleneck recall@20
    best_rec20 = max(summary.items(), key=lambda x: x[1]["recall20"][0])
    print(f"  Best Recall@20: {best_rec20[0]} ({best_rec20[1]['recall20'][0]:.2f})")

    # MRR improvement: raw→flow (conservation rescue)
    for src in SOURCES:
        raw_mrr = summary[f"{src}+raw"]["mrr"][0]
        flow_mrr = summary[f"{src}+flow"]["mrr"][0]
        sigma_mrr = summary[f"{src}+sigma"]["mrr"][0]
        improvement_flow = (flow_mrr - raw_mrr) / (raw_mrr + 1e-12)
        improvement_sigma = (sigma_mrr - raw_mrr) / (raw_mrr + 1e-12)
        print(f"  {src}: raw MRR={raw_mrr:.4f} → flow={flow_mrr:.4f} "
              f"({improvement_flow:+.1%}) → σ={sigma_mrr:.4f} ({improvement_sigma:+.1%})")

    # ── Per-bridge-edge rank analysis ──
    print(f"\n── Per-bridge-edge mean rank (/{len(edge_names)} total) ──")
    print(f"  {'Method':<28s} {'input->m0':>12s} {'a4.h1->logits':>16s}")

    for src in SOURCES:
        for var in METHOD_VARIANTS:
            results = all_results[src][var]
            r1 = np.mean([r["bridge_ranks"].get("input->m0", len(edge_names))
                          for r in results])
            r2 = np.mean([r["bridge_ranks"].get("a4.h1->logits", len(edge_names))
                          for r in results])
            print(f"  {src}+{var:<23s} {r1:>11.0f} {r2:>15.0f}")

    # ── Save ──
    output = {
        "benchmark": "bottleneck_edge_detection",
        "model": "interpbench_ioi",
        "ground_truth_bridges": BRIDGES,
        "ground_truth_all": ALL_GT,
        "n_edges_total": len(edge_names),
        "n_seeds": N_SEEDS,
        "published_sota_auroc": PUBLISHED_AUROC,
        "summary": {k: {kk: (float(vv[0]), float(vv[1]))
                        for kk, vv in v.items()}
                    for k, v in summary.items()},
        "per_seed": {k: v for k, v in all_results.items()},
    }

    # Convert numpy for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    path = f"{OUT}/benchmark_bottleneck.json"
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {path}")
    print(f"Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
