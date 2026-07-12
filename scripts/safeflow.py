"""
Safe Flow Decomposition for circuit discovery.

A flow network here is a single-source (`input`) single-sink (`logits`) DAG whose
edges carry a non-negative flow that conserves at every internal node
(in-flow == out-flow).  A path P is *safe* iff it is a sub-path of some path in
EVERY flow decomposition of f.  Khan/Rizzi/Tomescu (RECOMB'22) show:

    P = (e_1, ..., e_L) with junction vertices v_1..v_{L-1} is safe  <=>  f_P > 0,
    f_P = f(e_1) - sum_{i=1}^{L-1} ( f_out(v_i) - f(e_{i+1}) ).

Safety is closed under taking sub-paths, so maximal safe paths are well defined.

This module: (1) projects arbitrary signed edge attributions onto the nearest
conservative non-negative s-t flow (Dykstra), (2) enumerates all maximal safe
paths via the excess characterisation, (3) derives per-edge "safety" features.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np


# --------------------------------------------------------------------------- #
#  Flow network container
# --------------------------------------------------------------------------- #
@dataclass
class FlowDAG:
    node_index: Dict[str, int]                 # node name -> topo index
    nodes: List[str]                           # topo-ordered node names
    edges: List[Tuple[int, int]]              # (u_idx, v_idx) per edge
    edge_keys: List[str]                       # external key (e.g. eap edge name) per edge
    source: int
    sink: int
    weight: np.ndarray = field(default=None)   # raw (signed) weight per edge
    flow: np.ndarray = field(default=None)     # projected conservative flow per edge

    # cached adjacency
    out_edges: List[List[int]] = field(default=None)
    in_edges: List[List[int]] = field(default=None)

    def build_adjacency(self):
        self.out_edges = [[] for _ in self.nodes]
        self.in_edges = [[] for _ in self.nodes]
        for ei, (u, v) in enumerate(self.edges):
            self.out_edges[u].append(ei)
            self.in_edges[v].append(ei)


def make_flow_dag(node_names_topo: List[str],
                  edge_list: List[Tuple[str, str, str, float]],
                  source: str, sink: str) -> FlowDAG:
    """edge_list entries: (u_name, v_name, external_key, signed_weight)."""
    node_index = {n: i for i, n in enumerate(node_names_topo)}
    edges, keys, w = [], [], []
    for u, v, key, wt in edge_list:
        edges.append((node_index[u], node_index[v]))
        keys.append(key)
        w.append(wt)
    dag = FlowDAG(node_index=node_index, nodes=list(node_names_topo), edges=edges,
                  edge_keys=keys, source=node_index[source], sink=node_index[sink],
                  weight=np.asarray(w, dtype=np.float64))
    dag.build_adjacency()
    return dag


# --------------------------------------------------------------------------- #
#  Projection onto the conservative non-negative flow cone (Dykstra)
# --------------------------------------------------------------------------- #
def project_to_flow(dag: FlowDAG, target: Optional[np.ndarray] = None,
                    max_iter: int = 20000, tol: float = 1e-11,
                    verbose: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Euclidean projection of `target` (default |weight|) onto
        { f >= 0 : sum_{in v} f = sum_{out v} f  for all internal nodes v }.
    Uses Dykstra's alternating projection between the nonneg orthant and the
    conservation subspace.  Returns (flow, diagnostics).
    """
    if target is None:
        target = np.abs(dag.weight)
    target = np.asarray(target, dtype=np.float64)
    E = len(dag.edges)

    internal = [i for i in range(len(dag.nodes)) if i != dag.source and i != dag.sink]
    # incidence over internal nodes: +1 in-edge, -1 out-edge  (A f = 0)
    A = np.zeros((len(internal), E))
    for r, v in enumerate(internal):
        for ei in dag.in_edges[v]:
            A[r, ei] += 1.0
        for ei in dag.out_edges[v]:
            A[r, ei] -= 1.0
    # subspace projector Pmat = I - A^T (A A^T)^+ A
    AAt_pinv = np.linalg.pinv(A @ A.T)
    Pmat = np.eye(E) - A.T @ AAt_pinv @ A

    x = target.copy()
    p = np.zeros(E)
    q = np.zeros(E)
    last = x.copy()
    for it in range(max_iter):
        # project onto nonneg orthant (C1)
        y = np.maximum(x + p, 0.0)
        p = (x + p) - y
        # project onto conservation subspace (C2)
        x = Pmat @ (y + q)
        q = (y + q) - x
        if it % 5 == 0:
            delta = np.linalg.norm(x - last)
            last = x.copy()
            if delta < tol:
                break
    flow = np.maximum(x, 0.0)          # numerical clean-up (nonneg)
    # residual conservation error after clamp
    cons_err = float(np.abs(A @ flow).max()) if len(internal) else 0.0
    diag = {
        "iters": it + 1,
        "proj_residual": float(np.linalg.norm(flow - target)),
        "target_norm": float(np.linalg.norm(target)),
        "rel_residual": float(np.linalg.norm(flow - target) / (np.linalg.norm(target) + 1e-12)),
        "conservation_err": cons_err,
        "flow_value": float(sum(flow[ei] for ei in dag.out_edges[dag.source])),
        "n_pos_edges": int((flow > 1e-12).sum()),
    }
    dag.flow = flow
    if verbose:
        print("projection:", diag)
    return flow, diag


# --------------------------------------------------------------------------- #
#  Safe path machinery
# --------------------------------------------------------------------------- #
def out_flow(dag: FlowDAG, f: np.ndarray) -> np.ndarray:
    fout = np.zeros(len(dag.nodes))
    for v in range(len(dag.nodes)):
        fout[v] = sum(f[ei] for ei in dag.out_edges[v])
    return fout


def path_excess(dag: FlowDAG, f: np.ndarray, fout: np.ndarray, path_edges: List[int]) -> float:
    """f_P via the incremental excess recurrence."""
    if not path_edges:
        return 0.0
    exc = f[path_edges[0]]
    for k in range(1, len(path_edges)):
        junction = dag.edges[path_edges[k]][0]          # tail of e_{k} == head of e_{k-1}
        exc -= (fout[junction] - f[path_edges[k]])
    return exc


@dataclass
class SafePath:
    edges: List[int]
    excess: float

    @property
    def length(self):
        return len(self.edges)


def maximal_safe_paths(dag: FlowDAG, f: Optional[np.ndarray] = None,
                       eps: float = 1e-9, max_paths: int = 500000) -> List[SafePath]:
    """
    Enumerate all maximal safe paths.  A maximal safe path starts at a *left-end*
    edge (cannot be safely prepended) and is extended right until no extension
    keeps excess > eps.  Safety = excess > eps.
    """
    if f is None:
        f = dag.flow
    fout = out_flow(dag, f)
    pos = f > eps

    # left-end test for edge e=(u,v): u is source, or no in-edge d of u yields
    # 2-path excess  f(d) - (fout[u] - f(e)) > eps.
    def is_left_end(ei):
        u = dag.edges[ei][0]
        if u == dag.source:
            return True
        thresh = fout[u] - f[ei]          # need f(d) > thresh to prepend
        for d in dag.in_edges[u]:
            if pos[d] and f[d] - thresh > eps:
                return False
        return True

    results: List[SafePath] = []
    hit_cap = [False]

    def dfs(path, exc, head):
        # try to extend right
        extended = False
        for e2 in dag.out_edges[head]:
            if not pos[e2]:
                continue
            new_exc = exc - (fout[head] - f[e2])
            if new_exc > eps:
                extended = True
                if len(results) >= max_paths:
                    hit_cap[0] = True
                    return
                dfs(path + [e2], new_exc, dag.edges[e2][1])
        if not extended:
            results.append(SafePath(edges=list(path), excess=exc))

    for ei in range(len(dag.edges)):
        if pos[ei] and is_left_end(ei):
            dfs([ei], f[ei], dag.edges[ei][1])
            if hit_cap[0]:
                break
    return results


def forced_flow_certificate(dag: FlowDAG, f: np.ndarray, subset: Optional[List[int]] = None,
                            eps: float = 1e-12) -> np.ndarray:
    """
    Robustness certificate (the critique's "strongest version"): for each edge e,
        sigma_rob(e) = min_g  g(e)
        s.t.  g >= 0,  out-throughput(v)=f_out(v) and in-throughput(v)=f_in(v) for all v.
    i.e. the flow FORCED through e across every routing consistent with the node
    throughput profile of f.  A per-edge LP (HiGHS).  Edges routable-around -> 0;
    genuine bottlenecks -> high.
    """
    from scipy.optimize import linprog
    from scipy.sparse import csr_matrix
    E = len(dag.edges)
    fout = out_flow(dag, f)
    fin = np.zeros(len(dag.nodes))
    for v in range(len(dag.nodes)):
        fin[v] = sum(f[ei] for ei in dag.in_edges[v])
    # equality constraints: out-throughput per node, in-throughput per node
    rows, cols, data, b = [], [], [], []
    r = 0
    for v in range(len(dag.nodes)):
        if v == dag.sink:            # sink has no out-edges
            continue
        for ei in dag.out_edges[v]:
            rows.append(r); cols.append(ei); data.append(1.0)
        b.append(fout[v]); r += 1
    for v in range(len(dag.nodes)):
        if v == dag.source:          # source has no in-edges
            continue
        for ei in dag.in_edges[v]:
            rows.append(r); cols.append(ei); data.append(1.0)
        b.append(fin[v]); r += 1
    Aeq = csr_matrix((data, (rows, cols)), shape=(r, E))
    beq = np.array(b)
    bounds = [(0, None)] * E

    if subset is None:
        subset = [ei for ei in range(E) if f[ei] > eps]
    sigma = np.zeros(E)
    for ei in subset:
        c = np.zeros(E); c[ei] = 1.0
        res = linprog(c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        sigma[ei] = float(res.fun) if res.success else 0.0
    return sigma


def edge_safety_features(dag: FlowDAG, safe_paths: List[SafePath],
                         f: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    Per-edge features derived from the maximal safe paths:
      phi        : max excess of any maximal safe path containing the edge
      safe_len   : max length of any maximal safe path containing the edge
      n_paths    : number of maximal safe paths containing the edge
      on_chain   : 1 if in some maximal safe path of length >= 2
      reach_len  : (edges to source) + (edges to sink) of the longest containing path
    """
    if f is None:
        f = dag.flow
    E = len(dag.edges)
    phi = np.zeros(E)
    safe_len = np.zeros(E, dtype=int)
    n_paths = np.zeros(E, dtype=int)
    on_chain = np.zeros(E, dtype=bool)
    for sp in safe_paths:
        L = sp.length
        for pos_in_path, ei in enumerate(sp.edges):
            if sp.excess > phi[ei]:
                phi[ei] = sp.excess
            if L > safe_len[ei]:
                safe_len[ei] = L
            n_paths[ei] += 1
            if L >= 2:
                on_chain[ei] = True
    return {"phi": phi, "safe_len": safe_len, "n_paths": n_paths,
            "on_chain": on_chain, "flow": f.copy()}
