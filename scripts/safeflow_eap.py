"""Bridge between the eap.Graph attribution graph and the safeflow FlowDAG."""
import numpy as np
from eap.graph import Graph
from safeflow import (make_flow_dag, project_to_flow, maximal_safe_paths,
                      edge_safety_features, out_flow, forced_flow_certificate)


def node_topo_order(cfg):
    order = ["input"]
    for L in range(cfg["n_layers"]):
        for h in range(cfg["n_heads"]):
            order.append(f"a{L}.h{h}")
        order.append(f"m{L}")
    order.append("logits")
    return order


def flow_dag_from_graph(graph, use_abs=True):
    """Build a FlowDAG whose edges are exactly the eap graph's real edges,
    keyed by edge name, weighted by |score| (or signed)."""
    order = node_topo_order(graph.cfg)
    edge_list = []
    for name, e in graph.edges.items():
        w = float(abs(e.score) if use_abs else e.score)
        if not np.isfinite(w):          # IFR normalisation can emit NaN/inf
            w = 0.0
        edge_list.append((e.parent.name, e.child.name, name, w))
    dag = make_flow_dag(order, edge_list, "input", "logits")
    # sanity: topo order (all edges forward)
    assert all(u < v for (u, v) in dag.edges), "non-topological edge found"
    return dag


def build_scored_graph(cfg, edge_name_to_score):
    """Return a fresh eap Graph with edge.score set from a name->float dict (0 default)."""
    g = Graph.from_model(cfg)
    for name, e in g.edges.items():
        e.score = float(edge_name_to_score.get(name, 0.0))
    return g


def safe_flow_pipeline(graph, use_abs=True, eps_frac=1e-6, project_kwargs=None):
    """Full pipeline on an attribution graph. Returns dict of per-edge score arrays
    (keyed by edge name) and diagnostics."""
    project_kwargs = project_kwargs or {}
    dag = flow_dag_from_graph(graph, use_abs=use_abs)
    target = np.abs(dag.weight) if use_abs else dag.weight
    flow, diag = project_to_flow(dag, target=target, **project_kwargs)
    eps = max(1e-12, eps_frac * float(flow.max()))
    sps = maximal_safe_paths(dag, flow, eps=eps)
    feats = edge_safety_features(dag, sps, flow)

    keys = dag.edge_keys
    raw = {k: float(abs(w) if use_abs else w) for k, w in zip(keys, dag.weight)}
    flow_d = {k: float(v) for k, v in zip(keys, flow)}
    phi_d = {k: float(v) for k, v in zip(keys, feats["phi"])}
    on_chain = feats["on_chain"]
    safe_len = feats["safe_len"]
    fmax = float(flow.max()) + 1e-12

    # candidate per-edge scorings
    sigma = phi_d
    gated = {k: float(flow[i] * (1.0 + 10.0 * on_chain[i])) for i, k in enumerate(keys)}
    combo = {k: float(np.sqrt(max(flow[i], 0) * feats["phi"][i])) for i, k in enumerate(keys)}
    lenflow = {k: float(safe_len[i] + flow[i] / fmax) for i, k in enumerate(keys)}

    # safe-path length stats & sigma-vs-flow relationship
    lengths = [sp.length for sp in sps]
    pos = flow > eps
    # correlation between sigma and flow on positive-flow edges
    fi = flow[pos]; pi = feats["phi"][pos]
    if len(fi) > 3 and np.std(pi) > 0 and np.std(fi) > 0:
        from scipy.stats import spearmanr
        rho = float(spearmanr(fi, pi).statistic)
        # fraction of positive edges where phi essentially equals flow (sigma collapse)
        collapse = float(np.mean(np.abs(pi - fi) <= 1e-6 * (fi + 1e-12)))
    else:
        rho, collapse = float("nan"), float("nan")

    diag.update({
        "n_safe_paths": len(sps),
        "safe_len_max": int(max(lengths)) if lengths else 0,
        "safe_len_mean": float(np.mean(lengths)) if lengths else 0.0,
        "frac_len_le2": float(np.mean([l <= 2 for l in lengths])) if lengths else 1.0,
        "frac_len_ge3": float(np.mean([l >= 3 for l in lengths])) if lengths else 0.0,
        "n_pos_edges": int(pos.sum()),
        "n_on_chain_edges": int(on_chain.sum()),
        "spearman_sigma_flow": rho,
        "sigma_collapse_frac": collapse,
    })

    scorings = {"raw": raw, "flow": flow_d, "sigma": sigma,
                "gated": gated, "combo": combo, "lenflow": lenflow}
    # keep the maximal safe paths as node-name chains for qualitative inspection
    node_names = dag.nodes
    def path_to_nodes(sp):
        ns = [node_names[dag.edges[sp.edges[0]][0]]]
        for ei in sp.edges:
            ns.append(node_names[dag.edges[ei][1]])
        return ns
    top_paths = sorted(sps, key=lambda s: s.excess, reverse=True)[:25]
    top_paths_repr = [{"nodes": path_to_nodes(sp), "excess": float(sp.excess),
                       "len": sp.length,
                       "edges": [dag.edge_keys[ei] for ei in sp.edges]} for sp in top_paths]

    return {"scorings": scorings, "diag": diag, "dag": dag, "flow": flow,
            "feats": feats, "safe_paths": sps, "top_paths": top_paths_repr}


def backbone_safe_flow(graph, keep_fracs=(0.01, 0.02, 0.05, 0.1, 0.2, 0.5),
                       use_abs=True, eps_frac=1e-6):
    """Sparsify the projected flow to its top-k backbone, re-project to restore
    conservation, then run safe-flow.  Tests whether reducing fan-out yields
    non-degenerate (length >= 3) safe paths.  Returns per-keep_frac stats + sigma."""
    dag = flow_dag_from_graph(graph, use_abs=use_abs)
    target = np.abs(dag.weight)
    flow, _ = project_to_flow(dag, target=target)
    E = len(dag.edges)
    order = np.argsort(-flow)                      # descending flow
    results = {}
    for kf in keep_fracs:
        k = max(1, int(kf * E))
        mask = np.zeros(E, dtype=bool)
        mask[order[:k]] = True
        t2 = np.where(mask, flow, 0.0)
        f2, d2 = project_to_flow(dag, target=t2)   # re-conserve on backbone
        eps = max(1e-12, eps_frac * float(f2.max() + 1e-12))
        sps = maximal_safe_paths(dag, f2, eps=eps)
        feats = edge_safety_features(dag, sps, f2)
        lengths = [sp.length for sp in sps]
        sigma = {k_: float(v) for k_, v in zip(dag.edge_keys, feats["phi"])}
        results[kf] = {
            "k_edges": int(k), "n_pos": int((f2 > eps).sum()),
            "n_safe_paths": len(sps),
            "len_max": int(max(lengths)) if lengths else 0,
            "len_mean": float(np.mean(lengths)) if lengths else 0.0,
            "frac_len_ge3": float(np.mean([l >= 3 for l in lengths])) if lengths else 0.0,
            "sigma": sigma,
        }
    return dag, results


def robustness_scoring(graph, use_abs=True):
    """Per-edge robustness certificate sigma_rob(e) = min forced flow across all
    throughput-consistent flows (LP per positive edge)."""
    dag = flow_dag_from_graph(graph, use_abs=use_abs)
    flow, diag = project_to_flow(dag, target=np.abs(dag.weight))
    subset = [ei for ei in range(len(dag.edges)) if flow[ei] > 1e-12]
    sig = forced_flow_certificate(dag, flow, subset=subset)
    scoring = {k: float(v) for k, v in zip(dag.edge_keys, sig)}
    # how much does the robustness certificate differ from raw flow?
    fpos = flow[flow > 1e-12]; spos = sig[flow > 1e-12]
    frac_zeroed = float(np.mean(spos <= 1e-9)) if len(spos) else float("nan")
    return {"scoring": scoring, "n_pos": len(subset),
            "frac_forced_zero": frac_zeroed,
            "sig": sig, "flow": flow, "dag": dag}
