"""Quantify the bridge-rescue effect: mean rank of bridge vs branch ground-truth
edges under raw attribution vs conserved flow vs safe-flow sigma, over seeds."""
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import load_interpbench_model, load_reference_graph, run_attribution, auroc_mib_raw
from safeflow_eap import safe_flow_pipeline, build_scored_graph
from experiment3 import dataloader_seed

OUT = "/workspace/sfd-circuits/artifacts"
BRIDGES = ["input->m0", "a4.h1->logits"]                    # on every circuit input->logits path
BRANCHES = ["m0->a1.h1<v>", "m0->a2.h1<v>", "m0->a4.h1<v>",
            "a1.h1->a2.h1<v>", "a2.h1->a4.h1<v>"]           # fan-out / mid-circuit
SCORINGS = ["raw", "flow", "sigma"]


def ranks_of(edge_names, scoring, all_names):
    arr = np.array([scoring.get(n, 0.0) for n in all_names])
    order = np.argsort(-arr)
    rank = {all_names[order[i]]: i + 1 for i in range(len(all_names))}
    return {e: rank[e] for e in edge_names}


def main():
    model = load_interpbench_model()
    ref = load_reference_graph()
    seeds = range(6)
    agg = {src: {sc: {"bridge": [], "branch": []} for sc in SCORINGS}
           for src in ["EAP", "EAP-IG-inputs"]}
    for seed in seeds:
        dl = dataloader_seed(model, seed)
        for src in agg:
            g = run_attribution(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g, use_abs=True)
            names = list(g.edges.keys())
            for sc in SCORINGS:
                rk = ranks_of(BRIDGES + BRANCHES, pipe["scorings"][sc], names)
                agg[src][sc]["bridge"].append(np.mean([rk[e] for e in BRIDGES]))
                agg[src][sc]["branch"].append(np.mean([rk[e] for e in BRANCHES]))

    print(f"Mean rank / 1108 (lower=better), mean over {len(list(seeds))} seeds")
    print(f"{'source':16s} {'group':7s} " + " ".join(f"{sc:>14s}" for sc in SCORINGS))
    out = {}
    for src in agg:
        out[src] = {}
        for grp in ["bridge", "branch"]:
            cells = []
            out[src][grp] = {}
            for sc in SCORINGS:
                a = np.array(agg[src][sc][grp])
                out[src][grp][sc] = {"mean": float(a.mean()), "std": float(a.std())}
                cells.append(f"{a.mean():6.0f}+/-{a.std():4.0f}")
            print(f"{src:16s} {grp:7s} " + " ".join(f"{c:>14s}" for c in cells))
    json.dump(out, open(f"{OUT}/results4_bridges.json", "w"), indent=2)
    print(f"saved -> {OUT}/results4_bridges.json")


if __name__ == "__main__":
    main()
