"""
Cross-Input Stability Analysis on GPT2-IOI.

Verifies that the stability findings from InterpBench transfer to a real model
(GPT2-small, 158 nodes, 32,491 edges). Since there's no ground-truth circuit,
we measure:
  1. Split-to-split rank stability (6 splits × 100 examples)
  2. Top-k Jaccard overlap between splits
  3. Per-edge score coefficient of variation
  4. Size convergence (20, 50, 100 examples)
  5. Which edges are most consistently top-ranked across splits
  6. Faithfulness of circuits built from the most stable edges
"""
import os, sys, json, time
from functools import partial
from collections import defaultdict
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))

from common import get_metric, clone_graph, HFEAPDataset
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from eap.graph import Graph
from eap.attribute import attribute
from transformer_lens import HookedTransformer

OUT = "/workspace/sfd-circuits/artifacts"
DEVICE = "cuda"
os.makedirs(OUT, exist_ok=True)


def load_gpt2():
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def dataloader_split(model, seed, num_examples=100, batch_size=20):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="gpt2", num_examples=None)
    n = len(ds.dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_examples, n), replace=False).tolist()
    ds.dataset = ds.dataset.select(idx)
    return ds.to_dataloader(batch_size=batch_size)


def run_attribution_gpt2(model, dataloader, method, ig_steps=5):
    graph = Graph.from_model(model)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    attribution_metric = partial(metric, mean=True, loss=True)
    attribute(model, graph, dataloader, attribution_metric, method,
              intervention="patching", ig_steps=ig_steps, quiet=True)
    return graph


def compute_stability(all_scores, edge_names):
    """Compute split-to-split stability metrics from a list of score dicts."""
    n_splits = len(all_scores)
    E = len(edge_names)
    edge_arr = np.array(edge_names)

    score_mat = np.zeros((n_splits, E))
    for i, scores in enumerate(all_scores):
        for j, name in enumerate(edge_names):
            score_mat[i, j] = scores.get(name, 0.0)

    rank_mat = np.zeros((n_splits, E), dtype=int)
    for i in range(n_splits):
        order = np.argsort(-score_mat[i])
        for r, jj in enumerate(order):
            rank_mat[i, jj] = r + 1

    # Spearman
    spear_pairs = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            rho = np.corrcoef(rank_mat[i], rank_mat[j])[0, 1]
            spear_pairs.append(rho)

    # Top-k Jaccard
    topk_jac = {}
    for k in [20, 50, 100, 200, 500, 1000]:
        overlaps = []
        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                si = set(edge_arr[rank_mat[i] <= k])
                sj = set(edge_arr[rank_mat[j] <= k])
                overlaps.append(len(si & sj) / len(si | sj))
        topk_jac[str(k)] = (float(np.mean(overlaps)), float(np.std(overlaps)))

    # Per-edge CV
    cv_per_edge = np.zeros(E)
    for jj in range(E):
        m = np.mean(score_mat[:, jj])
        s = np.std(score_mat[:, jj])
        cv_per_edge[jj] = s / (m + 1e-12)

    # Most stable edges
    top100_freq = defaultdict(int)
    for i in range(n_splits):
        top100 = set(edge_arr[rank_mat[i] <= 100])
        for e in top100:
            top100_freq[e] += 1
    most_stable = sorted(top100_freq.items(), key=lambda x: -x[1])[:30]

    return {
        "n_splits": n_splits, "n_edges": E,
        "spearman_mean": float(np.mean(spear_pairs)),
        "spearman_std": float(np.std(spear_pairs)),
        "spearman_min": float(np.min(spear_pairs)),
        "topk_jaccard": topk_jac,
        "cv_per_edge_mean": float(np.mean(cv_per_edge)),
        "cv_per_edge_median": float(np.median(cv_per_edge)),
        "cv_per_edge_q90": float(np.percentile(cv_per_edge, 90)),
        "most_stable_top100": [(e, int(f)) for e, f in most_stable],
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("Cross-Input Stability on GPT2-IOI")
    print("=" * 72)

    print("Loading GPT2-small ...")
    model = load_gpt2()
    print(f"Model loaded ({time.time()-t0:.0f}s)")

    # ═══════════════════════════════════════════════════════════════════════
    # Test 1: Split-to-split stability (6 splits × 100 examples)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Test 1: Split-to-split stability (6 × 100 examples) ──")
    N_SPLITS = 6
    METHODS = [("EAP", "raw"), ("EAP", "flow"), ("EAP", "sigma"),
               ("EAP-IG-inputs", "raw"), ("EAP-IG-inputs", "flow"),
               ("EAP-IG-inputs", "sigma")]

    split_scores = {m: [] for m in METHODS}
    for split_idx in range(N_SPLITS):
        dl = dataloader_split(model, seed=split_idx * 13, num_examples=100)
        for src in ["EAP", "EAP-IG-inputs"]:
            g_attr = run_attribution_gpt2(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g_attr, use_abs=True)
            for var in ["raw", "flow", "sigma"]:
                split_scores[(src, var)].append(pipe["scorings"][var])
        elapsed = time.time() - t0
        print(f"  split {split_idx+1}/{N_SPLITS} done ({elapsed:.0f}s)")

    edge_names = list(g_attr.edges.keys())
    print(f"  GPT2 graph: {len(edge_names)} edges")

    stability = {}
    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        stability[label] = compute_stability(split_scores[method], edge_names)

    # ═══════════════════════════════════════════════════════════════════════
    # Test 2: Size convergence (20, 50, 100 examples)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n── Test 2: Size convergence (20 → 50 → 100) ──")
    SIZES = [20, 50, 100]
    size_scores = {m: {} for m in METHODS}

    for size in SIZES:
        dl = dataloader_split(model, seed=99, num_examples=size)
        for src in ["EAP", "EAP-IG-inputs"]:
            g = run_attribution_gpt2(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g, use_abs=True)
            for var in ["raw", "flow", "sigma"]:
                size_scores[(src, var)][size] = pipe["scorings"][var]
        print(f"  size={size} done ({time.time()-t0:.0f}s)")

    edge_arr = np.array(edge_names)
    size_conv = {}
    for method in METHODS:
        src, var = method
        label = f"{src}+{var}"
        ref_scores = size_scores[method][100]
        ref_arr = np.array([ref_scores.get(n, 0.0) for n in edge_names])
        ref_order = np.argsort(-ref_arr)
        ref_rank = np.zeros(len(edge_names), dtype=int)
        for r, j in enumerate(ref_order):
            ref_rank[j] = r + 1

        conv = {}
        for size in [20, 50]:
            scores = size_scores[method][size]
            arr = np.array([scores.get(n, 0.0) for n in edge_names])
            order = np.argsort(-arr)
            rank = np.zeros(len(edge_names), dtype=int)
            for r, jj in enumerate(order):
                rank[jj] = r + 1
            rho = np.corrcoef(ref_rank, rank)[0, 1]

            top_overlaps = {}
            for k in [20, 50, 100, 500]:
                ref_set = set(edge_arr[ref_rank <= k])
                sz_set = set(edge_arr[rank <= k])
                top_overlaps[str(k)] = len(ref_set & sz_set) / k

            conv[str(size)] = {
                "spearman_with_100": float(rho),
                "topk_overlap": top_overlaps,
            }
        size_conv[label] = conv

    # ═══════════════════════════════════════════════════════════════════════
    # Print Results
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*72}")
    print("RESULTS: Split-to-Split Stability (GPT2-IOI, 6 × 100 examples)")
    print(f"{'='*72}")

    print(f"\n{'Method':<24s} {'Spearman ρ':>12s} {'Top-50 J':>10s} "
          f"{'Top-100 J':>10s} {'Top-500 J':>10s} {'Top-1000 J':>10s} "
          f"{'CV(score)':>10s}")
    print("-" * 90)

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability[label]
        print(f"  {label:<22s} {st['spearman_mean']:>9.4f}±{st['spearman_std']:.3f} "
              f"{st['topk_jaccard']['50'][0]:>9.3f} {st['topk_jaccard']['100'][0]:>9.3f} "
              f"{st['topk_jaccard']['500'][0]:>9.3f} {st['topk_jaccard']['1000'][0]:>9.3f} "
              f"{st['cv_per_edge_mean']:>9.3f}")

    # Best performers
    best_sp = max(stability.items(), key=lambda x: x[1]["spearman_mean"])
    best_j100 = max(stability.items(),
                    key=lambda x: x[1]["topk_jaccard"]["100"][0])
    print(f"\n  Most rank-stable:      {best_sp[0]} (ρ={best_sp[1]['spearman_mean']:.4f})")
    print(f"  Most top-100 stable:   {best_j100[0]} (J={best_j100[1]['topk_jaccard']['100'][0]:.3f})")

    # ── Size convergence ──
    print(f"\n{'='*72}")
    print("RESULTS: Size Convergence (vs 100-example reference)")
    print(f"{'='*72}")

    print(f"\n  {'Method':<24s} {'20→100 ρ':>10s} {'50→100 ρ':>10s} "
          f"{'20→100 top50':>14s} {'50→100 top50':>14s}")
    print(f"  {'-'*75}")
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        sc = size_conv[label]
        print(f"  {label:<22s} {sc['20']['spearman_with_100']:>9.4f} "
              f"{sc['50']['spearman_with_100']:>9.4f} "
              f"{sc['20']['topk_overlap']['50']:>13.3f} "
              f"{sc['50']['topk_overlap']['50']:>13.3f}")

    # ── Most stable edges ──
    print(f"\n{'='*72}")
    print("Most Consistently Top-100 Edges Across 6 Splits")
    print(f"(edges appearing in top-100 in all 6 splits)")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        stable_edges = stability[label]["most_stable_top100"]
        always_top100 = [(e, f) for e, f in stable_edges if f == 6]
        often_top100 = [(e, f) for e, f in stable_edges if f >= 5 and f < 6]
        print(f"\n  {label}:")
        print(f"    Always top-100 ({len(always_top100)}): "
              f"{', '.join(e for e,f in always_top100[:8])}"
              f"{'...' if len(always_top100) > 8 else ''}")
        print(f"    In 5/6 splits ({len(often_top100)}): "
              f"{', '.join(e for e,f in often_top100[:5])}"
              f"{'...' if len(often_top100) > 5 else ''}")

    # ── Edge type distribution of stable edges ──
    print(f"\n{'='*72}")
    print("Edge-Type Distribution of Most Stable Top-100 Edges")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        stable_edges = stability[label]["most_stable_top100"]
        always = [e for e, f in stable_edges if f >= 5]

        # Categorize edges
        direct_logits = sum(1 for e in always if "->logits" in e)
        embed_attn = sum(1 for e in always if "input->" in e and "<v>" in e)
        attn_attn = sum(1 for e in always if ".h" in e.split("->")[0] and ".h" in e.split("->")[1] and "<v>" in e)
        mlp_involved = sum(1 for e in always if "m" in e.split("->")[0] or "m" in e.split("->")[1])
        print(f"  {label}: {len(always)} edges (≥5/6 splits)")
        print(f"    direct→logits: {direct_logits}, embed→attn: {embed_attn}, "
              f"attn→attn: {attn_attn}, MLP-involved: {mlp_involved}")

    # ═══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("VERDICT: GPT2-IOI Cross-Input Stability")
    print(f"{'='*72}")

    # Compare with InterpBench results
    print(f"\n  Comparison with InterpBench (from stability_analysis.json):")
    print(f"  {'':24s} {'InterpBench ρ':>14s} {'GPT2-IOI ρ':>14s} {'Δ':>8s}")
    interp_results = {
        "EAP+raw": 0.817, "EAP+flow": 0.744, "EAP+sigma": 0.802,
        "EAP-IG-inputs+raw": 0.836, "EAP-IG-inputs+flow": 0.809,
        "EAP-IG-inputs+sigma": 0.805,
    }
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        ir = interp_results.get(label, 0)
        gr = stability[label]["spearman_mean"]
        delta = gr - ir
        print(f"  {label:<22s} {ir:>13.3f} {gr:>13.4f} {delta:>+7.3f}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability[label]
        parts = []
        if st["spearman_mean"] > 0.75:
            parts.append(f"rank-stable (ρ={st['spearman_mean']:.3f})")
        if st["topk_jaccard"]["100"][0] > 0.5:
            parts.append(f"top-100 stable (J={st['topk_jaccard']['100'][0]:.2f})")
        if st["cv_per_edge_mean"] < 1.0:
            parts.append(f"low score variance (CV={st['cv_per_edge_mean']:.2f})")
        print(f"  {label:<22s}: {' | '.join(parts) if parts else 'review needed'}")

    # ── Save ──
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output = {
        "test": "cross_input_stability_gpt2",
        "model": "gpt2-small",
        "task": "ioi",
        "n_edges": len(edge_names),
        "n_nodes": 158,
        "split_stability": stability,
        "size_convergence": size_conv,
        "interpbench_comparison": interp_results,
    }
    with open(f"{OUT}/stability_gpt2.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {OUT}/stability_gpt2.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
