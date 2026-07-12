# Safe Flow Decomposition for Circuit Discovery

Applying the Khan–Tomescu **safe flow decomposition** framework (RNA-assembly / DAG
flows) to **mechanistic-interpretability circuit discovery**, evaluated on the
**MIB benchmark's InterpBench IOI** model. See `../sfd-circuits/REPORT.md` for the
full write-up and the published artifact for the visual summary.

## TL;DR
The transformer attribution graph is formally the right object for safe-flow, but the
dense residual stream makes safe paths degenerate to length ≤ 2 (σ collapses onto raw
attribution); global AUROC is unchanged. The useful by-product: **conservation
projection is a bottleneck-edge detector** — it rescues ground-truth bridge edges by
10–18× in rank. The safe-flow engine is validated 50/50 against brute-force
decomposition enumeration.

## Layout
```
scripts/
  safeflow.py        core: excess-flow safety, maximal-safe-path enumeration,
                     Dykstra projection onto the conservative-flow cone, robustness LP
  test_safeflow.py   brute-force validation (50 integral flows)  -> 50/50 pass
  safeflow_eap.py    bridge between eap.Graph and the FlowDAG; Safe-Circuit pipeline
  common.py          load InterpBench model + data + AUROC harness
  experiment.py      main: raw vs flow vs sigma vs variants (AUROC + faithfulness + diag)
  experiment2.py     backbone sweep + robustness certificate
  experiment3.py     bootstrap AUROC (6 seeds) + GT-edge localization
  experiment4.py     bridge-vs-branch rank rescue (6 seeds)
artifacts/           results{,2,3,4}.json, report.html
repos/               MIB-circuit-track (+ pinned EAP-IG submodule), MIB, EAP-IG
```

## Reproduce
```bash
source /venv/main/bin/activate
export HF_HOME=/workspace/.hf_home
cd scripts
python test_safeflow.py     # validate the engine (50/50)
python experiment.py        # main AUROC + faithfulness + degeneracy table
python experiment3.py       # bootstrap AUROC + localization
python experiment4.py       # bridge-rescue table
```

## Key numbers (InterpBench IOI, 6-seed bootstrap)
- Degeneracy: max safe-path length = 2; σ==flow on ~88–94% of edges; Spearman(σ,flow) 0.91–0.95.
- AUROC (raw / flow / σ): EAP 0.719 / 0.647 / 0.677 · EAP-IG 0.727 / 0.741 / 0.734 · IFR 0.588 / 0.606 / 0.647.
- Bridge-edge mean rank /1108 (raw→σ): EAP 402→22 · EAP-IG 364→36. Branch edges trade off in reverse.
