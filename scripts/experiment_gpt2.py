"""
Safe-Circuit evaluation on GPT2-small IOI from MIB.

For EAP and EAP-IG-inputs attribution on GPT2-small IOI:
  1. project |scores| onto the conservative non-negative flow cone,
  2. run safe flow decomposition -> per-edge safety score sigma(e)=phi,
  3. compare faithfulness (CPR/CMD) of raw vs flow vs sigma,
  4. report degeneracy diagnostics (safe-path lengths, sigma-vs-flow collapse).
"""
import os, sys, json, time
from functools import partial
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from common import (
    get_metric, clone_graph, PERCENTAGES
)
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from MIB_circuit_track.dataset import HFEAPDataset
from eap.graph import Graph
from eap.attribute import attribute
from transformer_lens import HookedTransformer

OUT = "/workspace/sfd-circuits/artifacts"
os.makedirs(OUT, exist_ok=True)
DEVICE = "cuda"


def load_gpt2():
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def get_dataloader_gpt2(model, split="train", num_examples=100, batch_size=20):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split=split, task="ioi",
                      model_name="gpt2", num_examples=num_examples)
    return ds.to_dataloader(batch_size=batch_size)


def run_attribution_gpt2(model, dataloader, method, ig_steps=5):
    graph = Graph.from_model(model)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    attribution_metric = partial(metric, mean=True, loss=True)
    attribute(model, graph, dataloader, attribution_metric, method,
              intervention="patching", ig_steps=ig_steps, quiet=True)
    return graph


def faithfulness(model, graph, dataloader, metric):
    g = clone_graph(graph)
    wec, area_under, area_from_1, avg, faiths = evaluate_area_under_curve(
        model, g, dataloader, metric, level="edge", absolute=True, intervention="patching",
        intervention_dataloader=dataloader, quiet=True)
    return {"cpr": float(area_under), "cmd": float(area_from_1),
            "avg": float(avg), "faiths": [float(x) for x in faiths]}


def main():
    t0 = time.time()
    print("Loading GPT2-small ...")
    model = load_gpt2()
    dl = get_dataloader_gpt2(model, split="train", num_examples=100, batch_size=20)

    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    faith_metric = partial(metric, mean=False, loss=False)

    SOURCES = ["EAP", "EAP-IG-inputs"]
    SCORINGS = ["raw", "flow", "sigma"]
    results = {"model": "gpt2-small", "task": "ioi", "percentages": list(PERCENTAGES),
               "n_nodes": 0, "n_edges": 0, "sources": {}}

    for src in SOURCES:
        print(f"\n{'='*50}")
        print(f"SOURCE: {src}")
        print(f"{'='*50}")

        # Attribution
        t_attr = time.time()
        g_attr = run_attribution_gpt2(model, dl, src, ig_steps=5)
        print(f"Attribution done in {time.time()-t_attr:.1f}s")

        n_nodes = len(g_attr.nodes)
        n_edges = len(g_attr.edges)
        results["n_nodes"] = n_nodes
        results["n_edges"] = n_edges

        # Safe-flow pipeline
        t_sf = time.time()
        pipe = safe_flow_pipeline(g_attr, use_abs=True)
        diag = pipe["diag"]
        print(f"Safe-flow done in {time.time()-t_sf:.1f}s")
        print("Projection/safe diagnostics:")
        for k in ["rel_residual", "conservation_err", "flow_value", "n_pos_edges",
                  "n_safe_paths", "safe_len_max", "safe_len_mean", "frac_len_le2",
                  "frac_len_ge3", "n_on_chain_edges", "spearman_sigma_flow", "sigma_collapse_frac"]:
            print(f"    {k:22s} = {diag[k]}")

        # Top safe paths
        for sp in pipe["top_paths"][:5]:
            print(f"  safe path len={sp['len']} excess={sp['excess']:.6f}: {' -> '.join(sp['nodes'])}")

        src_res = {"diag": diag, "top_safe_paths": pipe["top_paths"], "scorings": {}}
        for sc in SCORINGS:
            print(f"\n  --- {sc} ---")
            key_score = pipe["scorings"][sc]
            g = build_scored_graph(g_attr.cfg, key_score)

            # Faithfulness
            t_f = time.time()
            fth = faithfulness(model, g, dl, faith_metric)
            src_res["scorings"][sc] = {"faithfulness": fth}
            print(f"  Faithfulness: CPR={fth['cpr']:.4f} CMD={fth['cmd']:.4f} avg={fth['avg']:.4f}")
            print(f"  Faiths per %: {[round(x,3) for x in fth['faiths']]}")
            print(f"  time: {time.time()-t_f:.1f}s")

        results["sources"][src] = src_res

    # Summary table
    print(f"\n\n{'='*60}")
    print("GPT2-IOI SUMMARY")
    print(f"{'='*60}")
    print(f"n_nodes={results['n_nodes']} n_edges={results['n_edges']}")
    for src in SOURCES:
        d = results["sources"][src]["diag"]
        print(f"\n{src}:")
        print(f"  safe_len_max={d['safe_len_max']} safe_len_mean={d['safe_len_mean']:.2f} "
              f"frac_len_ge3={d['frac_len_ge3']:.3f}")
        print(f"  spearman_sigma_flow={d['spearman_sigma_flow']:.3f} "
              f"sigma_collapse_frac={d['sigma_collapse_frac']:.3f}")
        print(f"  n_safe_paths={d['n_safe_paths']} n_pos_edges={d['n_pos_edges']}")
        for sc in SCORINGS:
            f = results["sources"][src]["scorings"][sc]["faithfulness"]
            print(f"  {sc:8s} CPR={f['cpr']:.4f} CMD={f['cmd']:.4f}")

    json.dump(results, open(f"{OUT}/results_gpt2.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/results_gpt2.json  (total {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
