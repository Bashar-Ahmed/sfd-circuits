"""Responses to the degeneracy of naive safe-flow: (1) sparsified-backbone
safe-flow, (2) the robustness certificate.  Evaluated on InterpBench IOI."""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from common import (load_interpbench_model, load_reference_graph, get_dataloader,
                    run_attribution, auroc_mib_raw)
from safeflow_eap import backbone_safe_flow, robustness_scoring, build_scored_graph

OUT = "/workspace/sfd-circuits/artifacts"


def main():
    t0 = time.time()
    model = load_interpbench_model()
    ref = load_reference_graph()
    dl = get_dataloader(model, split="train", num_examples=100, batch_size=50)

    out = {}
    for src in ["EAP", "EAP-IG-inputs"]:
        print(f"\n============ {src} ============")
        g_attr = run_attribution(model, dl, src, ig_steps=5)

        # ---- backbone sweep ----
        print(" backbone sweep (sparsify -> re-conserve -> safe-flow):")
        _, bb = backbone_safe_flow(g_attr)
        bb_out = {}
        for kf, r in bb.items():
            g = build_scored_graph(g_attr.cfg, r["sigma"])
            auc, _ = auroc_mib_raw(ref, g)
            bb_out[str(kf)] = {**{k: v for k, v in r.items() if k != "sigma"}, "auroc": float(auc)}
            print(f"   keep={kf:<5} k={r['k_edges']:<4} n_pos={r['n_pos']:<4} "
                  f"len_max={r['len_max']} frac_len>=3={r['frac_len_ge3']:.2f} "
                  f"n_paths={r['n_safe_paths']:<4} AUROC={auc:.4f}")

        # ---- robustness certificate ----
        print(" robustness certificate (min forced flow, LP per edge):")
        rob = robustness_scoring(g_attr)
        g = build_scored_graph(g_attr.cfg, rob["scoring"])
        auc_rob, _ = auroc_mib_raw(ref, g)
        print(f"   n_pos={rob['n_pos']} frac_forced_to_zero={rob['frac_forced_zero']:.2f} AUROC={auc_rob:.4f}")

        out[src] = {"backbone": bb_out,
                    "robustness": {"auroc": float(auc_rob),
                                   "frac_forced_zero": rob["frac_forced_zero"],
                                   "n_pos": rob["n_pos"]}}

    json.dump(out, open(f"{OUT}/results2.json", "w"), indent=2)
    print(f"\nsaved -> {OUT}/results2.json  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
