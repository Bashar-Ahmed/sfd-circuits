"""
Validate the safe-flow implementation against brute-force enumeration of ALL
unit decompositions of small integral flows.

Ground truth: a path P is safe iff, in EVERY decomposition of f into s-t paths,
some decomposition-path contains P as a contiguous sub-path.  For integral flows
this is decided exactly by enumerating all unit-path decompositions.
"""
import itertools, random
import numpy as np
from safeflow import make_flow_dag, path_excess, maximal_safe_paths, out_flow, project_to_flow


# ---------- brute force over integral flows ---------- #
def st_unit_paths(s, t, remaining, out_adj):
    """All simple s->t paths (as edge-id tuples) using edges with remaining>0."""
    paths = []
    def dfs(node, path, used_nodes):
        if node == t:
            paths.append(tuple(path)); return
        for (ei, v) in out_adj[node]:
            if remaining[ei] > 0 and v not in used_nodes:
                dfs(v, path + [ei], used_nodes | {v})
    dfs(s, [], {s})
    return paths


def all_decompositions(remaining, s, t, out_adj, cap=200000):
    """Yield every unit decomposition (list of edge-id tuples)."""
    results = []
    def rec(rem):
        if len(results) > cap:
            return
        if all(v == 0 for v in rem.values()):
            results.append([]); return []
        decs = []
        for p in st_unit_paths(s, t, rem, out_adj):
            rem2 = dict(rem)
            for ei in p:
                rem2[ei] -= 1
            sub = rec(rem2)
            for d in sub:
                decs.append([p] + d)
        # store leaf assembly differently: we accumulate full decompositions
        return decs
    # rec above returns nested; re-implement iteratively to collect full decs
    full = []
    def rec2(rem, acc):
        if len(full) > cap:
            return
        if all(v == 0 for v in rem.values()):
            full.append(list(acc)); return
        for p in st_unit_paths(s, t, rem, out_adj):
            rem2 = dict(rem)
            for ei in p:
                rem2[ei] -= 1
            rec2(rem2, acc + [p])
    rec2(remaining, [])
    return full


def contiguous_subpaths_of(path):
    subs = set()
    for i in range(len(path)):
        for j in range(i, len(path)):
            subs.add(tuple(path[i:j + 1]))
    return subs


def path_contains(container, sub):
    n, m = len(container), len(sub)
    for i in range(n - m + 1):
        if tuple(container[i:i + m]) == tuple(sub):
            return True
    return False


def brute_safe_set(decompositions, all_candidate_paths):
    safe = set()
    for cand in all_candidate_paths:
        ok = True
        for dec in decompositions:
            if not any(path_contains(dp, cand) for dp in dec):
                ok = False; break
        if ok:
            safe.add(cand)
    return safe


def build(nodes_topo, edge_list):
    """edge_list: (u,v,key,flow). Returns dag(with flow set), out_adj, edge_id map."""
    dag = make_flow_dag(nodes_topo, edge_list, nodes_topo[0], nodes_topo[-1])
    f = np.array([w for *_, w in edge_list], dtype=float)
    dag.flow = f
    out_adj = {i: [] for i in range(len(nodes_topo))}
    for ei, (u, v) in enumerate(dag.edges):
        out_adj[u].append((ei, v))
    return dag, f, out_adj


def check_case(name, nodes_topo, edge_list, verbose=False):
    dag, f, out_adj = build(nodes_topo, edge_list)
    s, t = dag.source, dag.sink
    remaining = {ei: int(round(f[ei])) for ei in range(len(f))}
    decs = all_decompositions(remaining, s, t, out_adj)
    assert len(decs) > 0, f"{name}: no decompositions found"

    # candidate contiguous edge-paths = all subpaths of all support s-t paths
    support_paths = st_unit_paths(s, t, {ei: (1 if f[ei] > 0 else 0) for ei in range(len(f))}, out_adj)
    cands = set()
    for p in support_paths:
        cands |= contiguous_subpaths_of(p)

    brute = brute_safe_set(decs, cands)

    fout = out_flow(dag, f)
    pred_safe = {c for c in cands if path_excess(dag, f, fout, list(c)) > 1e-9}

    ok_char = (brute == pred_safe)

    # maximal safe paths cover exactly the safe set
    msp = maximal_safe_paths(dag, f, eps=1e-9)
    covered = set()
    for sp in msp:
        covered |= contiguous_subpaths_of(sp.edges)
    # every maximal path is right- & left-maximal (excess>0, cannot extend)
    for sp in msp:
        assert sp.excess > 1e-9, f"{name}: maximal path has non-positive excess"
    ok_cover = (covered & cands) == brute  # safe candidates == subpaths of maximal paths

    status = "OK" if (ok_char and ok_cover) else "FAIL"
    print(f"[{status}] {name}: #decomp={len(decs)} #cands={len(cands)} "
          f"#brute_safe={len(brute)} #pred_safe={len(pred_safe)} #maximal={len(msp)}")
    if not (ok_char and ok_cover) or verbose:
        print("   char match:", ok_char, "| cover match:", ok_cover)
        print("   brute - pred:", brute - pred_safe)
        print("   pred - brute:", pred_safe - brute)
    return ok_char and ok_cover


def random_dag_flow(n_nodes, seed, max_w=3):
    rng = random.Random(seed)
    nodes = [f"n{i}" for i in range(n_nodes)]
    nodes[0], nodes[-1] = "s", "t"
    # random forward edges
    cand = [(i, j) for i in range(n_nodes) for j in range(i + 1, n_nodes)]
    rng.shuffle(cand)
    chosen = cand[: max(n_nodes, len(cand) // 3)]
    # build a random integral flow: sum unit s-t paths
    out_adj = {i: [j for (a, j) in [(u, v) for (u, v) in chosen] if a == i] for i in range(n_nodes)}
    out_adj = {i: [v for (u, v) in chosen if u == i] for i in range(n_nodes)}
    fdict = {}
    for _ in range(rng.randint(2, 5)):
        # random s->t path
        node = 0; path = []
        ok = True
        used = {0}
        while node != n_nodes - 1:
            nxts = [v for v in out_adj[node] if v not in used]
            if not nxts:
                ok = False; break
            nn = rng.choice(nxts)
            path.append((node, nn)); used.add(nn); node = nn
        if ok:
            for e in path:
                fdict[e] = fdict.get(e, 0) + 1
    if not fdict:
        return None
    edge_list = [(f"n{u}" if 0 < u < n_nodes - 1 else ("s" if u == 0 else "t"),
                  f"n{v}" if 0 < v < n_nodes - 1 else ("s" if v == 0 else "t"),
                  f"e{u}_{v}", w) for (u, v), w in fdict.items()]
    nodes_topo = ["s"] + [f"n{i}" for i in range(1, n_nodes - 1)] + ["t"]
    return nodes_topo, edge_list


if __name__ == "__main__":
    n_ok = n_tot = 0

    # --- hand cases ---
    # 1) simple branch: s->a(2), a->t1(1), a->t2(1)  [t1,t2 merge to t]
    c1_nodes = ["s", "a", "b", "c", "t"]
    c1_edges = [("s", "a", "sa", 2), ("a", "b", "ab", 1), ("a", "c", "ac", 1),
                ("b", "t", "bt", 1), ("c", "t", "ct", 1)]
    n_tot += 1; n_ok += check_case("branch", c1_nodes, c1_edges)

    # 2) two sources into a, two sinks out -> a->? not forced
    c2_nodes = ["s", "x", "y", "a", "p", "q", "t"]
    c2_edges = [("s", "x", "sx", 1), ("s", "y", "sy", 1),
                ("x", "a", "xa", 1), ("y", "a", "ya", 1),
                ("a", "p", "ap", 1), ("a", "q", "aq", 1),
                ("p", "t", "pt", 1), ("q", "t", "qt", 1)]
    n_tot += 1; n_ok += check_case("bowtie", c2_nodes, c2_edges)

    # 3) bottleneck chain: everything through a->b
    c3_nodes = ["s", "a", "b", "t"]
    c3_edges = [("s", "a", "sa", 3), ("a", "b", "ab", 3), ("b", "t", "bt", 3)]
    n_tot += 1; n_ok += check_case("bottleneck", c3_nodes, c3_edges)

    # 4) diamond with unequal splits
    c4_nodes = ["s", "a", "b", "c", "t"]
    c4_edges = [("s", "a", "sa", 3), ("a", "b", "ab", 2), ("a", "c", "ac", 1),
                ("b", "t", "bt", 2), ("c", "t", "ct", 1)]
    n_tot += 1; n_ok += check_case("diamond", c4_nodes, c4_edges)

    # 5) skip edge (residual-stream-like): s->a->t and s->t direct
    c5_nodes = ["s", "a", "t"]
    c5_edges = [("s", "a", "sa", 2), ("a", "t", "at", 2), ("s", "t", "st", 1)]
    n_tot += 1; n_ok += check_case("skip", c5_nodes, c5_edges)

    # --- random cases ---
    for seed in range(60):
        rc = random_dag_flow(random.Random(seed).randint(4, 7), seed)
        if rc is None:
            continue
        try:
            n_tot += 1; n_ok += check_case(f"rand{seed}", rc[0], rc[1])
        except Exception as e:
            print(f"[ERR ] rand{seed}: {e}")

    # --- projection sanity: a conservative flow projects to itself ---
    dag, f, _ = build(c4_nodes, c4_edges)
    proj, diag = project_to_flow(dag, target=f.copy())
    assert np.allclose(proj, f, atol=1e-6), f"projection changed a valid flow: {proj} vs {f}"
    # projection of a non-flow yields conservation
    dag2, _, _ = build(c4_nodes, c4_edges)
    noisy = np.abs(f + np.array([0.5, -0.3, 0.9, -0.2, 0.4]))
    proj2, diag2 = project_to_flow(dag2, target=noisy)
    print(f"[proj] valid-flow self-projection OK | noisy conservation_err={diag2['conservation_err']:.2e}")

    print(f"\n==== {n_ok}/{n_tot} cases passed ====")
