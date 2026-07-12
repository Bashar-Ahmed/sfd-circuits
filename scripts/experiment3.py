"""Statistical robustness (bootstrap over data) + ground-truth-edge localization."""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interpbench_model, load_reference_graph, run_attribution,
                    auroc_mib_raw, get_metric, HFEAPDataset)
from safeflow_eap import safe_flow_pipeline, build_scored_graph

OUT = "/workspace/sfd-circuits/artifacts"
SOURCES = ["EAP", "EAP-IG-inputs", "information-flow-routes"]
SCORINGS = ["raw", "flow", "sigma"]


def dataloader_seed(model, seed, num_examples=100, batch_size=50):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="interpbench", num_examples=None)
    n = len(ds.dataset)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(num_examples, n), replace=False).tolist()
    ds.dataset = ds.dataset.select(idx)
    return ds.to_dataloader(batch_size=batch_size)


def main():
    t0 = time.time()
    model = load_interpbench_model()
    ref = load_reference_graph()
    gt_edges = [name for name, e in ref.edges.items() if bool(e.in_graph)]

    n_seeds = 6
    auroc = {src: {sc: [] for sc in SCORINGS} for src in SOURCES}
    localization = None

    for seed in range(n_seeds):
        dl = dataloader_seed(model, seed)
        for src in SOURCES:
            g_attr = run_attribution(model, dl, src, ig_steps=5)
            pipe = safe_flow_pipeline(g_attr, use_abs=True)
            for sc in SCORINGS:
                g = build_scored_graph(g_attr.cfg, pipe["scorings"][sc])
                auc, _ = auroc_mib_raw(ref, g)
                auroc[src][sc].append(float(auc))
            # GT-edge localization on seed 0, EAP-IG
            if seed == 0 and src == "EAP-IG-inputs":
                scorings = pipe["scorings"]
                E = len(g_attr.edges)
                loc = {}
                for sc in SCORINGS:
                    arr = np.array([scorings[sc].get(name, 0.0) for name in g_attr.edges])
                    names = list(g_attr.edges.keys())
                    # percentile rank of each GT edge (1=best)
                    order = np.argsort(-arr)
                    rank = {names[order[i]]: i + 1 for i in range(len(names))}
                    loc[sc] = {ge: {"score": float(scorings[sc].get(ge, 0.0)),
                                    "rank": int(rank[ge]),
                                    "pctile": float(1 - (rank[ge] - 1) / E)} for ge in gt_edges}
                localization = {"n_edges": E, "gt_edges": gt_edges, "by_scoring": loc}

    print("\n==== AUROC bootstrap (mean +/- std over %d data subsets) ====" % n_seeds)
    print(f"{'source':24s} " + " ".join(f"{sc:>16s}" for sc in SCORINGS))
    summ = {}
    for src in SOURCES:
        cells = []
        summ[src] = {}
        for sc in SCORINGS:
            a = np.array(auroc[src][sc])
            summ[src][sc] = {"mean": float(a.mean()), "std": float(a.std()), "vals": a.tolist()}
            cells.append(f"{a.mean():.3f}+/-{a.std():.3f}")
        print(f"{src:24s} " + " ".join(f"{c:>16s}" for c in cells))

    # paired sigma vs raw
    print("\n==== paired delta (sigma - raw) per source ====")
    for src in SOURCES:
        d = np.array(auroc[src]["sigma"]) - np.array(auroc[src]["raw"])
        print(f"  {src:24s} mean_delta={d.mean():+.4f}  (per-seed: {[round(x,3) for x in d.tolist()]})")

    print("\n==== GT-edge localization (EAP-IG, seed 0) rank/1108 (lower=better) ====")
    for ge in localization["gt_edges"]:
        row = "  " + ge.ljust(22)
        for sc in SCORINGS:
            r = localization["by_scoring"][sc][ge]["rank"]
            row += f"  {sc}={r:<5}"
        print(row)

    json.dump({"auroc_bootstrap": summ, "localization": localization},
              open(f"{OUT}/results3.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/results3.json  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
