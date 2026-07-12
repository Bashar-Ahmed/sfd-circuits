"""
Cross-Input Stability Analysis on Qwen2.5-0.5B IOI.

Completes the 2×2 model×task grid:
  - GPT2-Small IOI  (done)
  - Qwen2.5-0.5B MCQA (done)
  - Qwen2.5-0.5B IOI  (this script)
  - (InterpBench IOI already done)

IOI train set has 10,000 examples → we can use 6 splits × 100 examples.

Metrics:
  1. Split-to-split Spearman ρ (6 splits × 100 examples)
  2. Top-k Jaccard overlap between splits
  3. Per-edge score CV
  4. Size convergence (20, 50, 100 examples)
  5. Edge-type distribution of stable edges
"""
import os, sys, json, time
from functools import partial
from collections import defaultdict
import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))
MIB_REPO = "/workspace/sfd-circuits/repos/MIB-circuit-track"
sys.path.insert(0, MIB_REPO)

from safeflow_eap import safe_flow_pipeline
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from eap.graph import Graph
from eap.attribute import attribute
from transformer_lens import HookedTransformer

OUT = "/workspace/sfd-circuits/artifacts"
DEVICE = "cuda"
os.makedirs(OUT, exist_ok=True)


def load_qwen():
    model = HookedTransformer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        attn_implementation="eager",
        torch_dtype=torch.bfloat16,
        device=DEVICE,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    return model


def dataloader_split(model, seed, num_examples=100, batch_size=10):
    ds = HFEAPDataset(
        "mib-bench/ioi", model.tokenizer,
        split="train", task="ioi",
        model_name="qwen2.5", num_examples=None,
    )
    n = len(ds.dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_examples, n), replace=False).tolist()
    ds.dataset = ds.dataset.select(idx)
    return ds.to_dataloader(batch_size=batch_size)


def run_attribution_qwen(model, dataloader, method, ig_steps=5):
    graph = Graph.from_model(model)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    attribution_metric = partial(metric, mean=True, loss=True)
    attribute(model, graph, dataloader, attribution_metric, method,
              intervention="patching", ig_steps=ig_steps, quiet=True)
    return graph


def compute_stability(all_scores, edge_names):
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

    spear_pairs = []
    for i in range(n_splits):
        for j in range(i + 1, n_splits):
            rho = np.corrcoef(rank_mat[i], rank_mat[j])[0, 1]
            spear_pairs.append(rho)

    topk_jac = {}
    for k in [20, 50, 100, 200, 500, 1000]:
        overlaps = []
        for i in range(n_splits):
            for j in range(i + 1, n_splits):
                si = set(edge_arr[rank_mat[i] <= k])
                sj = set(edge_arr[rank_mat[j] <= k])
                overlaps.append(len(si & sj) / len(si | sj))
        topk_jac[str(k)] = (float(np.mean(overlaps)), float(np.std(overlaps)))

    cv_per_edge = np.zeros(E)
    for jj in range(E):
        m = np.mean(score_mat[:, jj])
        s = np.std(score_mat[:, jj])
        cv_per_edge[jj] = s / (m + 1e-12)

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
        "spearman_max": float(np.max(spear_pairs)),
        "topk_jaccard": topk_jac,
        "cv_per_edge_mean": float(np.mean(cv_per_edge)),
        "cv_per_edge_median": float(np.median(cv_per_edge)),
        "cv_per_edge_q90": float(np.percentile(cv_per_edge, 90)),
        "most_stable_top100": [(e, int(f)) for e, f in most_stable],
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("Cross-Input Stability on Qwen2.5-0.5B IOI")
    print("=" * 72)

    print("Loading Qwen2.5-0.5B ...")
    model = load_qwen()
    print(f"Model loaded: {model.cfg.n_layers} layers, {model.cfg.n_heads} heads, "
          f"d_model={model.cfg.d_model}  ({time.time()-t0:.0f}s)")

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
            g_attr = run_attribution_qwen(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g_attr, use_abs=True)
            for var in ["raw", "flow", "sigma"]:
                split_scores[(src, var)].append(pipe["scorings"][var])

            if split_idx == 0 and src == "EAP":
                diag = pipe["diag"]
                print(f"  Graph: {diag.get('n_edges','?')} edges, "
                      f"σ_collapse={diag.get('sigma_collapse_frac', 0):.3f}, "
                      f"σ-vs-flow ρ={diag.get('spearman_sigma_flow', 0):.3f}, "
                      f"safe_len_max={diag.get('safe_len_max', 0)}, "
                      f"projection residual={diag.get('projection_residual_norm', 0):.4f}")
        elapsed = time.time() - t0
        print(f"  split {split_idx+1}/{N_SPLITS} done ({elapsed:.0f}s)")

    edge_names = list(g_attr.edges.keys())
    print(f"  Qwen IOI graph: {len(edge_names)} edges")

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
            g = run_attribution_qwen(model, dl, src, ig_steps=5)
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
    print("RESULTS: Split-to-Split Stability (Qwen2.5-0.5B IOI, 6 × 100 examples)")
    print(f"{'='*72}")

    header = (f"{'Method':<24s} {'Spearman ρ':>12s} {'Top-20 J':>10s} "
              f"{'Top-50 J':>10s} {'Top-100 J':>10s} {'Top-500 J':>10s} "
              f"{'CV(score)':>10s}")
    print(f"\n{header}")
    print("-" * len(header))

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability[label]
        print(f"  {label:<22s} {st['spearman_mean']:>9.4f}±{st['spearman_std']:.3f} "
              f"{st['topk_jaccard']['20'][0]:>9.3f} {st['topk_jaccard']['50'][0]:>9.3f} "
              f"{st['topk_jaccard']['100'][0]:>9.3f} {st['topk_jaccard']['500'][0]:>9.3f} "
              f"{st['cv_per_edge_mean']:>9.3f}")

    best_sp = max(stability.items(), key=lambda x: x[1]["spearman_mean"])
    best_j100 = max(stability.items(),
                    key=lambda x: x[1]["topk_jaccard"]["100"][0])
    lowest_cv = min(stability.items(), key=lambda x: x[1]["cv_per_edge_mean"])
    print(f"\n  Most rank-stable:      {best_sp[0]} (ρ={best_sp[1]['spearman_mean']:.4f})")
    print(f"  Most top-100 stable:   {best_j100[0]} (J={best_j100[1]['topk_jaccard']['100'][0]:.3f})")
    print(f"  Lowest score variance: {lowest_cv[0]} (CV={lowest_cv[1]['cv_per_edge_mean']:.3f})")

    # ── Size convergence ──
    print(f"\n{'='*72}")
    print("RESULTS: Size Convergence (vs 100-example reference)")
    print(f"{'='*72}")

    header2 = (f"  {'Method':<24s} {'20→100 ρ':>10s} {'50→100 ρ':>10s} "
               f"{'20→100 top50':>14s} {'50→100 top50':>14s}")
    print(f"\n{header2}")
    print(f"  {'-'*75}")
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        sc = size_conv[label]
        print(f"  {label:<22s} {sc['20']['spearman_with_100']:>9.4f} "
              f"{sc['50']['spearman_with_100']:>9.4f} "
              f"{sc['20']['topk_overlap']['50']:>13.3f} "
              f"{sc['50']['topk_overlap']['50']:>13.3f}")

    best_conv = max(size_conv.items(),
                    key=lambda x: x[1]["20"]["spearman_with_100"])
    print(f"\n  Fastest converger (20→100): {best_conv[0]} "
          f"(ρ={best_conv[1]['20']['spearman_with_100']:.4f})")

    # ── Most stable edges ──
    print(f"\n{'='*72}")
    print("Most Consistently Top-100 Edges Across 6 Splits")
    print(f"(edges appearing in top-100 in all 6 splits)")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        stable_edges = stability[label]["most_stable_top100"]
        all6 = [(e, f) for e, f in stable_edges if f == 6]
        in56 = [(e, f) for e, f in stable_edges if f >= 5]
        print(f"\n  {label}:")
        print(f"    In 6/6 splits ({len(all6)}): "
              f"{', '.join(e for e,f in all6[:8])}"
              f"{'...' if len(all6) > 8 else ''}")
        print(f"    In ≥5/6 splits ({len(in56)}): "
              f"{', '.join(e for e,f in in56[:5])}"
              f"{'...' if len(in56) > 5 else ''}")

    # ── Edge type distribution ──
    print(f"\n{'='*72}")
    print("Edge-Type Distribution of Most Stable Top-100 Edges (≥5/6 splits)")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        stable_edges = stability[label]["most_stable_top100"]
        in56 = [e for e, f in stable_edges if f >= 5]

        direct_logits = sum(1 for e in in56 if "->logits" in e)
        embed_src = sum(1 for e in in56 if "input->" in e or "embed" in e.split("->")[0].lower())
        attn_to_attn = sum(1 for e in in56 if ".h" in e.split("->")[0] and ".h" in e.split("->")[1])
        mlp_involved = sum(1 for e in in56 if "m" in e.split("->")[0] or "m" in e.split("->")[1])

        print(f"  {label}: {len(in56)} edges (≥5/6 splits)")
        print(f"    direct→logits: {direct_logits}, embed→attn: {embed_src}, "
              f"attn→attn: {attn_to_attn}, MLP-involved: {mlp_involved}")

    # ═══════════════════════════════════════════════════════════════════════
    # Cross-model comparison
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("CROSS-MODEL COMPARISON: Qwen2.5-0.5B IOI vs GPT2-IOI vs InterpBench IOI")
    print(f"{'='*72}")

    prior_results = {
        ("InterpBench", "IOI", 1108): {
            "EAP+raw": (0.817, 0.650), "EAP+flow": (0.744, 0.713),
            "EAP+sigma": (0.802, 0.724),
            "EAP-IG-inputs+raw": (0.836, 0.602), "EAP-IG-inputs+flow": (0.809, 0.515),
            "EAP-IG-inputs+sigma": (0.805, 0.522),
        },
        ("GPT2-Small", "IOI", 32491): {
            "EAP+raw": (0.806, 0.528), "EAP+flow": (0.852, 0.044),
            "EAP+sigma": (0.989, 0.043),
            "EAP-IG-inputs+raw": (0.792, 0.529), "EAP-IG-inputs+flow": (0.817, 0.067),
            "EAP-IG-inputs+sigma": (0.982, 0.066),
        },
        ("Qwen2.5-0.5B", "MCQA", 179749): {
            "EAP+raw": (0.957, 0.284), "EAP+flow": (0.977, 0.032),
            "EAP+sigma": (0.985, 0.032),
            "EAP-IG-inputs+raw": (0.963, 0.263), "EAP-IG-inputs+flow": (0.897, 0.033),
            "EAP-IG-inputs+sigma": (0.986, 0.031),
        },
    }

    for (model_name, task, n_edges), methods in prior_results.items():
        print(f"\n  --- {model_name} {task} ({n_edges} edges) ---")
        for m_label, (sp, cv) in methods.items():
            print(f"  {m_label:<28s} ρ={sp:.3f}  CV={cv:.3f}")

    print(f"\n  --- Qwen2.5-0.5B IOI ({len(edge_names)} edges) [NEW] ---")
    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability[label]
        print(f"  {label:<28s} ρ={st['spearman_mean']:.4f}  CV={st['cv_per_edge_mean']:.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("VERDICT: Qwen2.5-0.5B IOI Cross-Input Stability")
    print(f"{'='*72}")

    for method in METHODS:
        label = f"{method[0]}+{method[1]}"
        st = stability[label]
        parts = []
        if st["spearman_mean"] > 0.90:
            parts.append(f"HIGHLY rank-stable (ρ={st['spearman_mean']:.4f})")
        elif st["spearman_mean"] > 0.75:
            parts.append(f"rank-stable (ρ={st['spearman_mean']:.3f})")
        elif st["spearman_mean"] > 0.50:
            parts.append(f"moderately rank-stable (ρ={st['spearman_mean']:.3f})")
        else:
            parts.append(f"low rank stability (ρ={st['spearman_mean']:.3f})")

        if st["topk_jaccard"]["100"][0] > 0.5:
            parts.append(f"top-100 stable (J={st['topk_jaccard']['100'][0]:.2f})")
        if st["cv_per_edge_mean"] < 1.0:
            parts.append(f"low score variance (CV={st['cv_per_edge_mean']:.2f})")
        elif st["cv_per_edge_mean"] < 0.10:
            parts.append(f"VERY low score variance (CV={st['cv_per_edge_mean']:.3f})")

        print(f"  {label:<22s}: {' | '.join(parts) if parts else 'review needed'}")

    raw_rho = stability["EAP+raw"]["spearman_mean"]
    sigma_rho = stability["EAP+sigma"]["spearman_mean"]
    sigma_vs_raw = sigma_rho - raw_rho
    print(f"\n  EAP σ vs EAP raw ρ delta: {sigma_vs_raw:+.4f} "
          f"{'σ WINS' if sigma_vs_raw > 0 else 'raw wins'}")

    raw_cv = stability["EAP+raw"]["cv_per_edge_mean"]
    sigma_cv = stability["EAP+sigma"]["cv_per_edge_mean"]
    cv_ratio = raw_cv / (sigma_cv + 1e-12)
    print(f"  EAP σ vs EAP raw CV ratio: {cv_ratio:.1f}× "
          f"{'σ WINS' if cv_ratio > 1 else 'raw wins'}")

    # ── Save ──
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output = {
        "test": "cross_input_stability_qwen_ioi",
        "model": "qwen2.5-0.5B",
        "task": "ioi",
        "n_edges": len(edge_names),
        "n_layers": model.cfg.n_layers,
        "n_heads": model.cfg.n_heads,
        "d_model": model.cfg.d_model,
        "split_sizes": 100,
        "n_splits": 6,
        "split_stability": stability,
        "size_convergence": size_conv,
        "prior_comparison": {f"{m}_{t}": v
                            for (m, t, _), v in prior_results.items()},
    }
    with open(f"{OUT}/stability_qwen_ioi.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {OUT}/stability_qwen_ioi.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
