"""Shared setup for Safe-Flow circuit-discovery experiments on the MIB InterpBench IOI model."""
import os, sys, pickle, copy
os.environ.setdefault("HF_HOME", "/workspace/.hf_home")
import torch
from huggingface_hub import hf_hub_download
from transformer_lens import HookedTransformer, HookedTransformerConfig

MIB_REPO = "/workspace/sfd-circuits/repos/MIB-circuit-track"
sys.path.insert(0, MIB_REPO)

from eap.graph import Graph
from eap.attribute import attribute
from MIB_circuit_track.dataset import HFEAPDataset
from MIB_circuit_track.metrics import get_metric
from MIB_circuit_track.evaluation import evaluate_area_under_roc, compare_graphs

DEVICE = "cuda"
PERCENTAGES = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)


def load_interpbench_model():
    hf_cfg = hf_hub_download("mib-bench/interpbench", filename="ll_model_cfg.pkl")
    hf_model = hf_hub_download("mib-bench/interpbench", subfolder="ioi_all_splits",
                               filename="ll_model_100_100_80.pth")
    cfg_dict = pickle.load(open(hf_cfg, "rb"))
    cfg = HookedTransformerConfig.from_dict(cfg_dict) if isinstance(cfg_dict, dict) else cfg_dict
    cfg.device = DEVICE
    cfg.use_hook_mlp_in = True
    cfg.use_attn_result = True
    cfg.use_split_qkv_input = True
    model = HookedTransformer(cfg)
    model.load_state_dict(torch.load(hf_model, map_location=DEVICE))
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True
    if model.tokenizer is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model.set_tokenizer(tok)
    return model


def load_reference_graph():
    return Graph.from_json(hf_hub_download("mib-bench/interpbench", filename="interpbench_graph.json"))


def get_dataloader(model, split="train", num_examples=100, batch_size=50):
    ds = HFEAPDataset("mib-bench/ioi", model.tokenizer, split=split, task="ioi",
                      model_name="interpbench", num_examples=num_examples)
    return ds.to_dataloader(batch_size=batch_size)


def run_attribution(model, dataloader, method, ig_steps=5):
    from functools import partial
    graph = Graph.from_model(model)
    metric = get_metric("logit_diff", "ioi", model.tokenizer, model)
    attribution_metric = partial(metric, mean=True, loss=True)
    attribute(model, graph, dataloader, attribution_metric, method,
              intervention="patching", ig_steps=ig_steps, quiet=True)
    return graph


def clone_graph(g):
    """Cheap non-mutating clone: rebuild topology from cfg, copy the scores tensor."""
    ng = Graph.from_model(dict(g.cfg))
    ng.scores[:] = g.scores
    ng.nodes_in_graph[:] = g.nodes_in_graph
    ng.in_graph[:] = g.in_graph
    return ng


def auroc_of_graph(reference, hypothesis):
    """Replicates MIB: apply_greedy at 10 pcts, compare vs reference, AUC of ROC(FPR,TPR).
    This variant sorts by FPR and anchors at (0,0)-(1,1) for a proper monotone ROC AUC."""
    hyp = clone_graph(hypothesis)
    d = evaluate_area_under_roc(reference, hyp)
    X, Y = d["FPR"], d["TPR"]
    auc = 0.0
    order = sorted(range(len(X)), key=lambda i: (X[i], Y[i]))
    Xs = [0.0] + [X[i] for i in order] + [1.0]
    Ys = [0.0] + [Y[i] for i in order] + [1.0]
    for i in range(len(Xs) - 1):
        auc += (Xs[i+1] - Xs[i]) * (Ys[i+1] + Ys[i]) / 2
    return auc, d


def auroc_mib_raw(reference, hypothesis):
    """AUROC exactly as MIB print_results does: area_under_curve(FPR, TPR) w/o sorting/anchoring."""
    hyp = clone_graph(hypothesis)
    d = evaluate_area_under_roc(reference, hyp)
    X, Y = d["FPR"], d["TPR"]
    auc = 0.0
    for i in range(len(X) - 1):
        x1, x2 = X[i] / X[-1], X[i+1] / X[-1]
        auc += (x2 - x1) * (Y[i+1] + Y[i]) / 2
    return auc, d
