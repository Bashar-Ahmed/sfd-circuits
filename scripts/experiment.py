"""
Safe-Circuit evaluation on the MIB InterpBench IOI model.

For each attribution source (EAP, EAP-IG-inputs, IFR) we:
  1. project |scores| onto the conservative non-negative flow cone,
  2. run safe flow decomposition -> per-edge safety score sigma(e)=phi,
  3. compare AUROC and faithfulness of several scorings against the ground-truth
     circuit, and report the degeneracy diagnostics (safe-path lengths, sigma-vs-flow).
"""
import os, sys, json, time, copy
from functools import partial
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interpbench_model, load_reference_graph, get_dataloader,
                    run_attribution, get_metric, auroc_mib_raw, auroc_of_graph, clone_graph,
                    PERCENTAGES)
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.evaluation import evaluate_area_under_curve, evaluate_area_under_roc, compare_graphs
from eap.graph import Graph

OUT = "/workspace/sfd-circuits/artifacts"
os.makedirs(OUT, exist_ok=True)


def precision_recall_curve(reference, hypothesis):
    """Greedy circuits at each pct; precision/recall vs ground-truth edges."""
    hyp = clone_graph(hypothesis)
    n = len(reference.edges)
    prec, rec, sizes = [], [], []
    for pct in PERCENTAGES:
        k = max(1, int(pct * n))
        hyp.apply_greedy(k)
        s = compare_graphs(reference, hyp)
        prec.append(s["precision"]); rec.append(s["recall"]); sizes.append(k)
    return prec, rec, sizes


def faithfulness(model, graph, dataloader, metric):
    g = clone_graph(graph)
    wec, area_under, area_from_1, avg, faiths = evaluate_area_under_curve(
        model, g, dataloader, metric, level="edge", absolute=True, intervention="patching",
        intervention_dataloader=dataloader, quiet=True)
    return {"cpr": float(area_under), "cmd": float(area_from_1),
            "avg": float(avg), "faiths": [float(x) for x in faiths]}


def main():
    t0 = time.time()
    model = load_interpbench_model()
    ref = load_reference_graph()
    dl = get_dataloader(model, split="train", num_examples=100, batch_size=50)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    faith_metric = partial(metric, mean=False, loss=False)

    gt_edges = [name for name, e in ref.edges.items() if bool(e.in_graph)]
    print("GROUND TRUTH circuit edges:", gt_edges)

    SOURCES = ["EAP", "EAP-IG-inputs", "information-flow-routes"]
    SCORINGS = ["raw", "flow", "sigma", "gated", "combo", "lenflow"]
    results = {"ground_truth_edges": gt_edges, "percentages": list(PERCENTAGES),
               "sources": {}}

    for src in SOURCES:
        print(f"\n=================== SOURCE: {src} ===================")
        g_attr = run_attribution(model, dl, src, ig_steps=5)
        # raw baseline auroc directly from attribution graph
        pipe = safe_flow_pipeline(g_attr, use_abs=True)
        diag = pipe["diag"]
        print("projection/safe diagnostics:")
        for k in ["rel_residual", "conservation_err", "flow_value", "n_pos_edges",
                  "n_safe_paths", "safe_len_max", "safe_len_mean", "frac_len_le2",
                  "frac_len_ge3", "n_on_chain_edges", "spearman_sigma_flow", "sigma_collapse_frac"]:
            print(f"    {k:22s} = {diag[k]}")

        src_res = {"diag": diag, "top_safe_paths": pipe["top_paths"], "scorings": {}}
        for sc in SCORINGS:
            key_score = pipe["scorings"][sc]
            g = build_scored_graph(g_attr.cfg, key_score)
            auc, roc = auroc_mib_raw(ref, g)
            prec, rec, sizes = precision_recall_curve(ref, g)
            fth = faithfulness(model, g, dl, faith_metric)
            src_res["scorings"][sc] = {
                "auroc": float(auc), "roc": {k: [float(x) for x in v] for k, v in roc.items()},
                "precision": prec, "recall": rec, "sizes": sizes,
                "faithfulness": fth,
            }
            print(f"  {sc:9s} AUROC={auc:.4f}  CPR={fth['cpr']:.3f} CMD={fth['cmd']:.3f}  "
                  f"P@.5%={prec[2]:.2f} R@.5%={rec[2]:.2f}  P@10%={prec[6]:.2f} R@10%={rec[6]:.2f}")
        results["sources"][src] = src_res

    # ---- headline table ----
    print("\n\n================= AUROC SUMMARY =================")
    hdr = f"{'source':24s} " + " ".join(f"{sc:>8s}" for sc in SCORINGS)
    print(hdr)
    for src in SOURCES:
        row = f"{src:24s} " + " ".join(f"{results['sources'][src]['scorings'][sc]['auroc']:8.4f}" for sc in SCORINGS)
        print(row)
    print("\n================= CPR (faithfulness area, higher=better) =================")
    for src in SOURCES:
        row = f"{src:24s} " + " ".join(f"{results['sources'][src]['scorings'][sc]['faithfulness']['cpr']:8.3f}" for sc in SCORINGS)
        print(row)

    json.dump(results, open(f"{OUT}/results.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/results.json   (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
