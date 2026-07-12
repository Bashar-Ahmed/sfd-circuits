"""
Comprehensive Circuit Discovery Evaluation on InterpBench IOI.

Computes three complementary metrics for all 6 methods:
  1. AUROC — Area Under ROC curve (standard MIB metric)
  2. AUPRC — Area Under Precision-Recall curve (better for imbalanced data)
  3. Task Accuracy Curve — faithfulness vs edge count (how well the circuit
     performs the IOI task as we add more edges)

Methods: EAP raw | EAP flow | EAP σ | EAP-IG-inputs raw | EAP-IG-inputs flow | EAP-IG-inputs σ

Ground truth: InterpBench IOI has 7 known circuit edges (2 bridges + 5 branch edges)
  in a graph of 1,108 total edges (imbalanced: 7/1108 = 0.63% positive class)
"""
import os, sys, json, time
from functools import partial
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))
MIB_REPO = "/workspace/sfd-circuits/repos/MIB-circuit-track"
sys.path.insert(0, MIB_REPO)

from common import (
    load_interpbench_model, load_reference_graph, run_attribution,
    get_metric, HFEAPDataset, clone_graph
)
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.evaluation import evaluate_area_under_roc, compare_graphs, evaluate_area_under_curve
from MIB_circuit_track.metrics import get_metric
from eap.graph import Graph

OUT = "/workspace/sfd-circuits/artifacts"
DEVICE = "cuda"
os.makedirs(OUT, exist_ok=True)

# Standard evaluation percentages
PERCENTAGES = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1]
# Ground truth edges (7 total, 2 bridges)
ALL_GT = ["input->m0", "m0->a1.h1<v>", "m0->a2.h1<v>", "m0->a4.h1<v>",
          "a1.h1->a2.h1<v>", "a2.h1->a4.h1<v>", "a4.h1->logits"]
BRIDGES = ["input->m0", "a4.h1->logits"]

# Published SOTA baseline AUROCs (from MIB paper)
PUBLISHED = {
    "EAP-IG-activations (best SOTA)": 0.81,
    "EAP (published)": 0.78,
    "EAP-IG-inputs (published)": 0.71,
    "UGS": 0.74,
    "IFR": 0.71,
    "NAP-IG": 0.62,
    "Random": 0.44,
}


def dataloader_full(model, num_examples=100, batch_size=50):
    """Full dataset dataloader for task evaluation (NOT attribution)."""
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="interpbench", num_examples=num_examples)
    return ds.to_dataloader(batch_size=batch_size)


def compute_auroc(ref_graph, hyp_graph):
    """Standard MIB AUROC: compute TPR/FPR at 10 percentages, integrate."""
    hyp = clone_graph(hyp_graph)
    d = evaluate_area_under_roc(ref_graph, hyp)
    X, Y = d["FPR"], d["TPR"]

    # Sort by FPR, anchor at (0,0) and (1,1)
    order = sorted(range(len(X)), key=lambda i: (X[i], Y[i]))
    Xs = [0.0] + [X[i] for i in order] + [1.0]
    Ys = [0.0] + [Y[i] for i in order] + [1.0]

    auc = 0.0
    for i in range(len(Xs) - 1):
        auc += (Xs[i+1] - Xs[i]) * (Ys[i+1] + Ys[i]) / 2

    return {
        "auroc": float(auc),
        "fpr": [float(x) for x in X],
        "tpr": [float(y) for y in Y],
        "fpr_anchored": [float(x) for x in Xs],
        "tpr_anchored": [float(y) for y in Ys],
    }


def compute_auprc(ref_graph, hyp_graph):
    """Area Under Precision-Recall curve, using the same 10 percentages."""
    hyp = clone_graph(hyp_graph)
    d = evaluate_area_under_roc(ref_graph, hyp)
    P, R = d["precision"], d["recall"]

    # Sort by recall, anchor at (0,1) and (1,0)
    order = sorted(range(len(P)), key=lambda i: (R[i], P[i]))
    Rs = [0.0] + [R[i] for i in order] + [1.0]
    Ps = [1.0] + [P[i] for i in order] + [0.0]

    auc = 0.0
    for i in range(len(Rs) - 1):
        auc += (Rs[i+1] - Rs[i]) * (Ps[i+1] + Ps[i]) / 2

    return {
        "auprc": float(auc),
        "precision": [float(p) for p in P],
        "recall": [float(r) for r in R],
        "recall_anchored": [float(r) for r in Rs],
        "precision_anchored": [float(p) for p in Ps],
    }


def compute_task_accuracy_curve(model, graph, dataloader, metric_fn, quiet=True):
    """Faithfulness curve: how much of the task metric is recovered at each edge budget.

    Returns faithfulness at each of the 10 standard percentages.
    Faithfulness = (ablated_score - corrupted_score) / (baseline_score - corrupted_score)
    where:
      - baseline_score = full model performance
      - corrupted_score = performance with 0 edges (all edges corrupted)
      - ablated_score = performance with top-N% edges kept, rest corrupted
    """
    w_edge_counts, area_under, area_from_1, avg_faith, faithfulnesses = evaluate_area_under_curve(
        model, graph, dataloader, metric_fn,
        quiet=quiet, level='edge', absolute=True,
        apply_greedy=True,  # MIB standard: greedy edge selection
    )
    return {
        "percentages": PERCENTAGES,
        "faithfulness": [float(f) for f in faithfulnesses],
        "weighted_edge_counts": [int(w) for w in w_edge_counts],
        "area_under_curve": float(area_under),
        "area_from_1": float(area_from_1),
        "avg_faithfulness": float(avg_faith),
    }


def compute_roc_pr_points(ref_graph, hyp_graph):
    """Get raw TPR/FPR/Precision/Recall at each percentage threshold."""
    hyp = clone_graph(hyp_graph)
    d = evaluate_area_under_roc(ref_graph, hyp)
    return {
        "percentages": PERCENTAGES,
        "tpr": [float(x) for x in d["TPR"]],
        "fpr": [float(x) for x in d["FPR"]],
        "precision": [float(x) for x in d["precision"]],
        "recall": [float(x) for x in d["recall"]],
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("Comprehensive Circuit Discovery Evaluation — InterpBench IOI")
    print("=" * 72)

    # Load model and reference
    print("\n[1/4] Loading model and ground-truth graph ...")
    model = load_interpbench_model()
    ref = load_reference_graph()
    print(f"  Model loaded ({time.time()-t0:.0f}s)")
    print(f"  Ground truth: {len(ALL_GT)} circuit edges, {len(BRIDGES)} bridges")
    print(f"  Graph size: {len(ref.edges)} edges")

    # Run attributions
    print("\n[2/4] Running attributions (EAP + EAP-IG-inputs) ...")
    dl_attr = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                           model_name="interpbench", num_examples=100).to_dataloader(batch_size=50)

    all_pipelines = {}
    for src in ["EAP", "EAP-IG-inputs"]:
        g_attr = run_attribution(model, dl_attr, src, ig_steps=5)
        pipe = safe_flow_pipeline(g_attr, use_abs=True)
        all_pipelines[src] = pipe
        diag = pipe["diag"]
        print(f"  {src}: edges={diag.get('n_edges','?')}, "
              f"σ_collapse={diag.get('sigma_collapse_frac',0):.3f}, "
              f"safe_len_max={diag.get('safe_len_max',0)}, "
              f"residual={diag.get('projection_residual_norm',0):.4f}")

    edge_names = list(g_attr.edges.keys())
    print(f"  Total edges: {len(edge_names)}")

    # Build scored graphs for each method
    print("\n[3/4] Computing AUROC, AUPRC, and ROC/PR points ...")
    METHODS = ["raw", "flow", "sigma"]
    SOURCES = ["EAP", "EAP-IG-inputs"]

    results = {}
    for src in SOURCES:
        for var in METHODS:
            label = f"{src}+{var}"
            scores = all_pipelines[src]["scorings"][var]
            g_scored = build_scored_graph(g_attr.cfg, scores)

            # AUROC
            roc = compute_auroc(ref, g_scored)
            # AUPRC
            prc = compute_auprc(ref, g_scored)
            # Raw ROC/PR points
            points = compute_roc_pr_points(ref, g_scored)

            results[label] = {
                "auroc": roc["auroc"],
                "auprc": prc["auprc"],
                "roc": roc,
                "prc": prc,
                "points": points,
            }
            print(f"  {label:<22s} AUROC={roc['auroc']:.4f}  AUPRC={prc['auprc']:.4f}")

    # Task accuracy curves
    print("\n[4/4] Computing task accuracy curves (faithfulness vs edge budget) ...")
    print("  (This runs model forward passes at each of 10 edge budgets)")

    dl_eval = dataloader_full(model, num_examples=200, batch_size=50)
    metric_fn = get_metric("logit_diff", "ioi", model.tokenizer, model)

    for src in SOURCES:
        for var in METHODS:
            label = f"{src}+{var}"
            scores = all_pipelines[src]["scorings"][var]
            g_scored = build_scored_graph(g_attr.cfg, scores)

            tac = compute_task_accuracy_curve(model, g_scored, dl_eval, metric_fn)
            results[label]["task_accuracy"] = tac

            print(f"  {label:<22s} avg_faith={tac['avg_faithfulness']:.4f}  "
                  f"AUC={tac['area_under_curve']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print Summary Tables
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*72}")
    print("RESULTS SUMMARY")
    print(f"{'='*72}")

    # Table 1: AUROC + AUPRC
    print(f"\n{'─'*70}")
    print("Table 1: AUROC and AUPRC (InterpBench IOI, 1,108 edges, 7 GT edges)")
    print(f"{'─'*70}")
    print(f"  {'Method':<24s} {'AUROC':>8s}  {'AUPRC':>8s}  "
          f"{'Published AUROC':>16s}")
    print(f"  {'-'*62}")

    for method_name, pub_auc in sorted(PUBLISHED.items(), key=lambda x: -x[1]):
        marker = " ← best SOTA" if pub_auc == max(PUBLISHED.values()) else ""
        print(f"  {method_name:<24s} {'—':>8s}  {'—':>8s}  {pub_auc:>15.2f}{marker}")
    print(f"  {'─'*62}")

    for src in SOURCES:
        for var in METHODS:
            label = f"{src}+{var}"
            r = results[label]
            best_marker = ""
            if r["auroc"] == max(v["auroc"] for v in results.values()):
                best_marker = " ★ best AUROC"
            if r["auprc"] == max(v["auprc"] for v in results.values()):
                best_marker += " ★ best AUPRC"
            print(f"  {label:<24s} {r['auroc']:>7.4f}  {r['auprc']:>7.4f}{best_marker}")

    # Table 2: ROC curve points (at key thresholds)
    print(f"\n{'─'*80}")
    print("Table 2: ROC/PR Curve Points (selected thresholds)")
    print(f"{'─'*80}")

    for src in SOURCES:
        print(f"\n  --- {src} variants ---")
        for var in METHODS:
            label = f"{src}+{var}"
            pts = results[label]["points"]

            print(f"\n  {label}:")
            print(f"    {'%':>6s}  {'TPR':>7s}  {'FPR':>7s}  {'Prec':>7s}  {'Rec':>7s}")
            for i, pct in enumerate(PERCENTAGES):
                print(f"    {pct*100:>4.0f}%  {pts['tpr'][i]:>6.3f}  {pts['fpr'][i]:>6.3f}  "
                      f"{pts['precision'][i]:>6.3f}  {pts['recall'][i]:>6.3f}")

    # Table 3: Task Accuracy Curve
    print(f"\n{'─'*90}")
    print("Table 3: Task Accuracy (Faithfulness) Curve — logit_diff recovery")
    print(f"{'─'*90}")

    print(f"\n  {'% edges':>7s}  ", end="")
    for src in SOURCES:
        for var in METHODS:
            print(f"{src}+{var:<22s}  ", end="")
    print()

    for i, pct in enumerate(PERCENTAGES):
        print(f"  {pct*100:>4.0f}%     ", end="")
        for src in SOURCES:
            for var in METHODS:
                label = f"{src}+{var}"
                faith = results[label]["task_accuracy"]["faithfulness"][i]
                print(f"{faith:>20.4f}     ", end="")
        print()

    # Best at each threshold
    print(f"\n  Best method at each threshold:")
    for i, pct in enumerate(PERCENTAGES):
        best_method = max(
            [(src, var) for src in SOURCES for var in METHODS],
            key=lambda m: results[f"{m[0]}+{m[1]}"]["task_accuracy"]["faithfulness"][i]
        )
        best_faith = results[f"{best_method[0]}+{best_method[1]}"]["task_accuracy"]["faithfulness"][i]
        print(f"    {pct*100:>4.0f}%: {best_method[0]}+{best_method[1]} ({best_faith:.4f})")

    # Table 4: Edge budget to reach 90% and 95% faithfulness
    print(f"\n{'─'*80}")
    print("Table 4: Edge budget to reach target faithfulness")
    print(f"{'─'*80}")
    print(f"  {'Method':<24s}  {'90% faith at':>15s}  {'95% faith at':>15s}  {'Max faith':>12s}")
    print(f"  {'-'*70}")

    for src in SOURCES:
        for var in METHODS:
            label = f"{src}+{var}"
            faiths = results[label]["task_accuracy"]["faithfulness"]

            pct_90 = None
            pct_95 = None
            for i, f in enumerate(faiths):
                if pct_90 is None and f >= 0.90:
                    pct_90 = PERCENTAGES[i] * 100
                if pct_95 is None and f >= 0.95:
                    pct_95 = PERCENTAGES[i] * 100

            p90_str = f"{pct_90:.0f}%" if pct_90 is not None else ">100%"
            p95_str = f"{pct_95:.0f}%" if pct_95 is not None else ">100%"
            max_f = max(faiths)
            print(f"  {label:<24s}  {p90_str:>15s}  {p95_str:>15s}  {max_f:>11.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Head-to-head comparisons
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("HEAD-TO-HEAD: Safe-Flow vs Raw Attribution")
    print(f"{'='*72}")

    for metric_name, metric_key in [("AUROC", "auroc"), ("AUPRC", "auprc")]:
        print(f"\n  {metric_name} comparison:")
        for src in ["EAP", "EAP-IG-inputs"]:
            raw_val = results[f"{src}+raw"][metric_key]
            flow_val = results[f"{src}+flow"][metric_key]
            sigma_val = results[f"{src}+sigma"][metric_key]

            flow_delta = (flow_val - raw_val) / (raw_val + 1e-12) * 100
            sigma_delta = (sigma_val - raw_val) / (raw_val + 1e-12) * 100
            print(f"    {src}: raw={raw_val:.4f} → flow={flow_val:.4f} ({flow_delta:+.1f}%) "
                  f"→ σ={sigma_val:.4f} ({sigma_delta:+.1f}%)")

    # Faithfulness comparison at key operating points
    print(f"\n  Faithfulness at 1% and 5% edge budgets:")
    for budget_pct in [0.01, 0.05]:
        idx = PERCENTAGES.index(budget_pct)
        print(f"\n    At {budget_pct*100:.0f}% edges ({int(budget_pct*1108)} edges):")
        for src in SOURCES:
            for var in METHODS:
                label = f"{src}+{var}"
                faith = results[label]["task_accuracy"]["faithfulness"][idx]
                print(f"      {label:<22s} {faith:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════════
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output = {
        "test": "comprehensive_evaluation",
        "model": "interpbench_ioi",
        "n_edges": len(edge_names),
        "n_gt_edges": len(ALL_GT),
        "ground_truth_edges": ALL_GT,
        "bridge_edges": BRIDGES,
        "published_auroc": PUBLISHED,
        "results": results,
        "diagnostics": {
            src: all_pipelines[src]["diag"] for src in SOURCES
        },
    }
    with open(f"{OUT}/evaluation_metrics.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {OUT}/evaluation_metrics.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
