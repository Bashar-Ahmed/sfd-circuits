"""
CPR, CMD, and Task Accuracy Curves for GPT2-IOI.

Computes the MIB standard metrics for circuit discovery on GPT2-Small IOI:
  - CPR (Circuit Probability Recovery): area under faithfulness curve.
    Uses abs=False (signed scores for greedy selection per MIB convention).
  - CMD (Circuit Metric Distance): area between faithfulness and 1.0.
    Uses abs=True (absolute scores per MIB convention).
  - Task Accuracy Curve: faithfulness at each of 10 edge budgets.

The abs nuance: MIB specifies CPR uses signed scores (abs=False) and CMD uses
absolute scores (abs=True). For safe-flow variants (flow, sigma), scores are
always non-negative (projected onto the flow cone), so abs flag doesn't matter.
But for raw EAP scores, sign matters — negative attribution means the edge
is anti-correlated with the task metric.

Methods: EAP raw | flow | sigma — EAP-IG-inputs raw | flow | sigma
"""
import os, sys, json, time
from functools import partial
import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
sys.path.insert(0, os.path.dirname(__file__))
MIB_REPO = "/workspace/sfd-circuits/repos/MIB-circuit-track"
sys.path.insert(0, MIB_REPO)

from safeflow_eap import safe_flow_pipeline, build_scored_graph
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.evaluation import evaluate_area_under_curve
from eap.graph import Graph
from eap.attribute import attribute
from transformer_lens import HookedTransformer

OUT = "/workspace/sfd-circuits/artifacts"
DEVICE = "cuda"
os.makedirs(OUT, exist_ok=True)

PERCENTAGES = [.001, .002, .005, .01, .02, .05, .1, .2, .5, 1]


def load_gpt2():
    model = HookedTransformer.from_pretrained('gpt2-small', device=DEVICE)
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    return model


def dataloader_gpt2(model, num_examples=100, batch_size=10):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="train", task="ioi",
                      model_name="gpt2", num_examples=num_examples)
    return ds.to_dataloader(batch_size=batch_size)


def run_attribution_gpt2(model, dataloader, method, ig_steps=5):
    graph = Graph.from_model(model)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    attribution_metric = partial(metric, mean=True, loss=True)
    attribute(model, graph, dataloader, attribution_metric, method,
              intervention="patching", ig_steps=ig_steps, quiet=True)
    return graph


def evaluate_faithfulness(model, graph, dataloader, metric_fn, absolute, apply_greedy=True):
    """Run MIB evaluate_area_under_curve and return CPR/CMD metrics."""
    w_edge_counts, area_under, area_from_1, avg_faith, faithfulnesses = \
        evaluate_area_under_curve(
            model, graph, dataloader, metric_fn,
            quiet=False, level='edge', absolute=absolute,
            apply_greedy=apply_greedy,
        )
    return {
        "weighted_edge_counts": [int(w) for w in w_edge_counts],
        "area_under": float(area_under),        # CPR when abs=False
        "area_from_1": float(area_from_1),       # CMD when abs=True
        "avg_faithfulness": float(avg_faith),
        "faithfulness": [float(f) for f in faithfulnesses],
        "percentages": PERCENTAGES,
    }


def main():
    t0 = time.time()
    print("=" * 72)
    print("CPR / CMD / Task Accuracy — GPT2-IOI")
    print("=" * 72)

    # Load model
    print("\n[1/4] Loading GPT2-Small ...")
    model = load_gpt2()
    print(f"  Model loaded ({time.time()-t0:.0f}s)")

    # Run attributions
    print("\n[2/4] Running attributions (EAP + EAP-IG-inputs) ...")
    dl_attr = dataloader_gpt2(model, num_examples=100, batch_size=10)

    all_pipelines = {}
    for src in ["EAP", "EAP-IG-inputs"]:
        g_attr = run_attribution_gpt2(model, dl_attr, src, ig_steps=5)

        # Run pipeline TWICE: once with abs=False (for CPR), once with abs=True (for CMD)
        pipe_abs = safe_flow_pipeline(g_attr, use_abs=True)     # for CMD & flow/sigma
        pipe_signed = safe_flow_pipeline(g_attr, use_abs=False)  # for CPR raw

        all_pipelines[src] = {
            "abs": pipe_abs,
            "signed": pipe_signed,
            "graph": g_attr,
        }
        diag = pipe_abs["diag"]
        print(f"  {src}: edges={diag.get('n_edges','?')}, "
              f"σ_collapse={diag.get('sigma_collapse_frac',0):.3f}, "
              f"safe_len_max={diag.get('safe_len_max',0)}, "
              f"residual={diag.get('projection_residual_norm',0):.4f}")

    edge_names = list(g_attr.edges.keys())
    print(f"  Total edges: {len(edge_names)}")

    # Evaluation dataloader (use test split, larger batch for speed)
    print("\n[3/4] Setting up evaluation dataloader ...")
    dl_eval = HFEAPDataset("mib-bench/ioi", model.tokenizer, split="test",
                           task="ioi", model_name="gpt2").to_dataloader(batch_size=100)
    metric_fn = partial(get_metric("logit_diff", "ioi", model.tokenizer, model),
                        mean=False, loss=False)
    print(f"  Test split: {len(dl_eval.dataset)} examples, batch_size=100")

    # ═══════════════════════════════════════════════════════════════════════
    # Evaluate each method
    # ═══════════════════════════════════════════════════════════════════════
    print("\n[4/4] Computing CPR, CMD, and task accuracy curves ...")

    SOURCES = ["EAP", "EAP-IG-inputs"]
    VARIANTS = ["raw", "flow", "sigma"]

    results = {}
    for src in SOURCES:
        g_attr = all_pipelines[src]["graph"]
        pipe_abs = all_pipelines[src]["abs"]
        pipe_signed = all_pipelines[src]["signed"]

        for var in VARIANTS:
            label = f"{src}+{var}"

            # For CPR (abs=False): use signed scores for raw, flow/sigma from abs pipe
            if var == "raw":
                cpr_scores = pipe_signed["scorings"]["raw"]  # signed raw scores
            else:
                cpr_scores = pipe_abs["scorings"][var]  # flow/sigma are always non-negative

            # For CMD (abs=True): use absolute scores for raw, flow/sigma from abs pipe
            cmd_scores = pipe_abs["scorings"][var]  # all from abs pipe

            # Build scored graphs
            g_cpr = build_scored_graph(g_attr.cfg, cpr_scores)
            g_cmd = build_scored_graph(g_attr.cfg, cmd_scores)

            # CPR evaluation: abs=False, apply_greedy=True
            print(f"\n  {label} CPR (abs=False, signed ranking) ...")
            cpr_result = evaluate_faithfulness(
                model, g_cpr, dl_eval, metric_fn,
                absolute=False, apply_greedy=True,
            )

            # CMD evaluation: abs=True, apply_greedy=True
            print(f"  {label} CMD (abs=True, absolute ranking) ...")
            cmd_result = evaluate_faithfulness(
                model, g_cmd, dl_eval, metric_fn,
                absolute=True, apply_greedy=True,
            )

            results[label] = {
                "cpr": cpr_result["area_under"],         # CPR
                "cmd": cmd_result["area_from_1"],         # CMD
                "cpr_details": cpr_result,
                "cmd_details": cmd_result,
            }

            print(f"    CPR={cpr_result['area_under']:.4f}  "
                  f"CMD={cmd_result['area_from_1']:.4f}  "
                  f"avg_faith(CPR)={cpr_result['avg_faithfulness']:.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Print Summary Tables
    # ═══════════════════════════════════════════════════════════════════════

    print(f"\n{'='*72}")
    print("RESULTS: CPR, CMD, and Task Accuracy — GPT2-IOI")
    print(f"{'='*72}")

    # Table 1: CPR and CMD
    print(f"\n{'─'*60}")
    print("Table 1: CPR and CMD Scores (higher CPR = better, lower CMD = better)")
    print(f"{'─'*60}")
    print(f"  {'Method':<24s} {'CPR ↑':>10s} {'CMD ↓':>10s} "
          f"{'Faith(CPR)':>12s} {'Faith(CMD)':>12s}")
    print(f"  {'-'*70}")

    for src in SOURCES:
        for var in VARIANTS:
            label = f"{src}+{var}"
            r = results[label]
            print(f"  {label:<24s} {r['cpr']:>9.4f} {r['cmd']:>9.4f} "
                  f"{r['cpr_details']['avg_faithfulness']:>11.4f} "
                  f"{r['cmd_details']['avg_faithfulness']:>11.4f}")

    # Best of each
    best_cpr = max(results.items(), key=lambda x: x[1]["cpr"])
    best_cmd = min(results.items(), key=lambda x: x[1]["cmd"])
    print(f"\n  Best CPR:  {best_cpr[0]} ({best_cpr[1]['cpr']:.4f})")
    print(f"  Best CMD:  {best_cmd[0]} ({best_cmd[1]['cmd']:.4f})")

    # Table 2: Task Accuracy (Faithfulness) Curves — CPR mode (abs=False)
    print(f"\n{'─'*90}")
    print("Table 2a: Task Accuracy Curve — CPR mode (abs=False, signed ranking)")
    print(f"{'─'*90}")

    # Header
    print(f"\n  {'% edges':>7s}  ", end="")
    for src in SOURCES:
        for var in VARIANTS:
            label = f"{src}+{var}"
            print(f"{label:<20s}  ", end="")
    print()

    for i, pct in enumerate(PERCENTAGES):
        print(f"  {pct*100:>4.0f}%     ", end="")
        for src in SOURCES:
            for var in VARIANTS:
                label = f"{src}+{var}"
                faith = results[label]["cpr_details"]["faithfulness"][i]
                print(f"{faith:>19.4f}  ", end="")
        print()

    # Table 2b: Task Accuracy — CMD mode (abs=True)
    print(f"\n{'─'*90}")
    print("Table 2b: Task Accuracy Curve — CMD mode (abs=True, absolute ranking)")
    print(f"{'─'*90}")

    print(f"\n  {'% edges':>7s}  ", end="")
    for src in SOURCES:
        for var in VARIANTS:
            label = f"{src}+{var}"
            print(f"{label:<20s}  ", end="")
    print()

    for i, pct in enumerate(PERCENTAGES):
        print(f"  {pct*100:>4.0f}%     ", end="")
        for src in SOURCES:
            for var in VARIANTS:
                label = f"{src}+{var}"
                faith = results[label]["cmd_details"]["faithfulness"][i]
                print(f"{faith:>19.4f}  ", end="")
        print()

    # Table 3: CPR vs CMD gap (impact of abs flag)
    print(f"\n{'─'*70}")
    print("Table 3: Impact of abs flag on faithfulness (CPR abs=F vs CMD abs=T)")
    print(f"{'─'*70}")
    print(f"  {'Method':<24s} {'CPR avg_f':>10s} {'CMD avg_f':>10s} "
          f"{'Δ(CPR-CMD)':>12s} {'Interpretation':>20s}")
    print(f"  {'-'*78}")

    for src in SOURCES:
        for var in VARIANTS:
            label = f"{src}+{var}"
            cpr_f = results[label]["cpr_details"]["avg_faithfulness"]
            cmd_f = results[label]["cmd_details"]["avg_faithfulness"]
            delta = cpr_f - cmd_f
            interp = ""
            if abs(delta) < 0.01:
                interp = "no sign effect"
            elif delta > 0.01:
                interp = "signed better"
            else:
                interp = "absolute better"
            print(f"  {label:<24s} {cpr_f:>9.4f} {cmd_f:>9.4f} "
                  f"{delta:>+11.4f} {interp:>20s}")

    # Table 4: Best method at each budget
    print(f"\n{'─'*70}")
    print("Table 4: Best method at each edge budget (CPR mode)")
    print(f"{'─'*70}")

    for i, pct in enumerate(PERCENTAGES):
        n_edges = int(pct * len(edge_names))
        best = max(
            [(s, v) for s in SOURCES for v in VARIANTS],
            key=lambda m: results[f"{m[0]}+{m[1]}"]["cpr_details"]["faithfulness"][i]
        )
        best_faith = results[f"{best[0]}+{best[1]}"]["cpr_details"]["faithfulness"][i]
        runner_up = sorted(
            [(s, v) for s in SOURCES for v in VARIANTS],
            key=lambda m: results[f"{m[0]}+{m[1]}"]["cpr_details"]["faithfulness"][i]
        )[-2]
        ru_faith = results[f"{runner_up[0]}+{runner_up[1]}"]["cpr_details"]["faithfulness"][i]
        print(f"    {pct*100:>4.0f}% ({n_edges:>5d} edges): "
              f"{best[0]}+{best[1]} ({best_faith:.4f})  "
              f"runner-up: {runner_up[0]}+{runner_up[1]} ({ru_faith:.4f})")

    # ═══════════════════════════════════════════════════════════════════════
    # Head-to-head analysis
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*72}")
    print("HEAD-TO-HEAD: Safe-Flow vs Raw on GPT2-IOI")
    print(f"{'='*72}")

    for metric_name, metric_key in [("CPR", "cpr"), ("CMD", "cmd")]:
        direction = "↑" if metric_key == "cpr" else "↓"
        print(f"\n  {metric_name} ({direction}):")
        for src in ["EAP", "EAP-IG-inputs"]:
            raw_val = results[f"{src}+raw"][metric_key]
            flow_val = results[f"{src}+flow"][metric_key]
            sigma_val = results[f"{src}+sigma"][metric_key]

            flow_delta = (flow_val - raw_val) / (abs(raw_val) + 1e-12) * 100
            sigma_delta = (sigma_val - raw_val) / (abs(raw_val) + 1e-12) * 100
            print(f"    {src}: raw={raw_val:.4f} → flow={flow_val:.4f} ({flow_delta:+.1f}%) "
                  f"→ σ={sigma_val:.4f} ({sigma_delta:+.1f}%)")

    # Compare best safe-flow vs best raw at key budgets
    print(f"\n  Faithfulness gain of σ over raw at key budgets (CPR mode):")
    for pct_key in [0.01, 0.02, 0.05, 0.1, 0.2]:
        idx = PERCENTAGES.index(pct_key)
        for src in ["EAP", "EAP-IG-inputs"]:
            raw_f = results[f"{src}+raw"]["cpr_details"]["faithfulness"][idx]
            sigma_f = results[f"{src}+sigma"]["cpr_details"]["faithfulness"][idx]
            delta = sigma_f - raw_f
            print(f"    {pct_key*100:>4.0f}% edges, {src}: raw={raw_f:.4f} σ={sigma_f:.4f} "
                  f"Δ={delta:+.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    # Save
    # ═══════════════════════════════════════════════════════════════════════
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    output = {
        "test": "cpr_cmd_task_accuracy_gpt2",
        "model": "gpt2-small",
        "task": "ioi",
        "n_edges": len(edge_names),
        "percentages": PERCENTAGES,
        "results": results,
    }
    with open(f"{OUT}/evaluation_gpt2.json", "w") as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved → {OUT}/evaluation_gpt2.json")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
