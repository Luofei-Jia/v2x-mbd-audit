# Audit-core code: Reassessing V2X Misbehavior Detection (VeReMi NextGen)

Reference implementation for the paper's memorization audit, honest-protocol
evaluation, and operating-point analysis. Every number, table, and figure in
the paper is produced by these scripts from `results/*.json` (included here so
the published numbers are verifiable without downloading the datasets).

## Scripts -> paper artifacts

| Script | Paper artifact | Output (results/) |
|---|---|---|
| `sweep_attacks.py` | Audit table, Figs. 1-2, Sec. IV (P1/P2/P3 protocols) | `sweep_attacks.json`, `sweep_attacks_seeds.json` |
| `sweep_hard3_f5.py` | Sec. VI F5 sweep; core data/feature library (imported by other scripts) | `sweep_hard3_f5.json`, `sweep_hard3_f5_seeds.json` |
| `seq_baselines.py` | Table IV (LSTM/Transformer rows), Fig. 3 | `seq_baselines.json`, `seq_baselines_seeds.json` |
| `seq_baselines_ext.py` | Sec. VII full-attack coverage (remaining 12 attacks) | `seq_baselines_ext_seeds.json` |
| `stgnn_baseline.py` | Table IV (GraphSAGE rows) | `stgnn_baseline.json`, `stgnn_baseline_seeds.json` |
| `gnn_tune.py` | Table IV (GraphSAGE-200ep and no-graph MLP control) | `gnn_tune_*.json` |
| `event_metrics.py` | Event table (event recall, alarm burden) | `event_metrics.json` |
| `operational_metrics.py` | Operational table, Sec. VII (latency, stability, cost) | `operational_metrics.json` |
| `cross_attack_matrix.py` | Fig. 4, LOAO column (unseen-attack transfer) | `cross_attack_matrix.json` |
| `transfer_cross_scenario.py` | Cross-scenario table (highway_2 <-> urban_2) | `transfer_cross_scenario.json` |
| `cross_dataset_extension.py` | Sec. VIII cross-dataset (NextGen <-> VeReMi Extension) | `cross_dataset_extension.json` |
| `hard3_signatures.py` | Sec. VI source-code audit (rcv_lag artifact, dist_sr) | `hard3_signatures.json` |
| `validate_extension_labels.py` | Appendix label-semantics validation (standalone) | stdout tables |
| `make_figures.py` | Figs. 1-4 (reads results JSONs; writes PNGs to cwd) | `fig*.png` |
| `gen_addition_tables.py` | LaTeX table fragments emitted from results JSON | stdout |

## Data

Scripts expect the datasets under `data/` in this layout (paths can be adjusted
in each script's `ROOT` constant):

- **VeReMi NextGen** (Zenodo record 19665762, CC-BY 4.0):
  `data/nextgen/<Scenario>/<Scenario>/{Train,Test}/.../*.json`
  Scenarios used: `InTAS_highway_2`, `InTAS_urban_2`, plus the per-attack
  variants (`InTAS_highway_2_<attack>`) for the 15 attack classes.
- **VeReMi Extension** (Mendeley 10.17632/k62n4z9gdz.1):
  `data/extension_mixalldata_clean.csv` (used by `validate_extension_labels.py`
  and `cross_dataset_extension.py`).

## Environment

Python 3 with `numpy`, `pandas`, `scikit-learn`, `torch` (CPU is sufficient),
`matplotlib`. Runs were performed on CPU; deep baselines are small (2-layer
LSTM / Transformer, 2-layer GraphSAGE). All stochastic models are run with 3
fixed seeds (`0,1,2`).

## Running

Run from this directory. Example:

```bash
python3 sweep_attacks.py                 # protocol audit -> results/
python3 seq_baselines.py                 # Table IV sequences
python3 make_figures.py                  # figures 1-4 (PNG in cwd)
python3 validate_extension_labels.py data/extension_mixalldata_clean.csv
```

Feature sets: F1 (12 instantaneous kinematics) ⊂ F3 (16, + per-sender
consistency) ⊂ F34 (25) ⊂ F34+F5legit (29; F5 = receiver-view `dist_sr`,
`close_rate`, `lat_off`, content `collision`).

## Preregistration protocol

`EXPERIMENT_PROTOCOL_V1.md` is the full preregistration protocol and
decision log referenced by the paper (the sections §9a--§9o cited there
index this file). It is the original working document, written in Chinese,
with dated registrations for every decision, including predictions that
were later refuted. It was added to this repository with v1.0.1
(2026-10-16); its dated entries are contemporaneous records of each
decision as it was made.

## Protocol P2 (sender-disjoint)

Senders appearing in the other split are removed from **both** sides, with
both filters computed on the original frames:

```python
tr_senders = set(tr.sender_id)
tr = tr[~tr.sender_id.isin(set(te.sender_id))]
te = te[~te.sender_id.isin(tr_senders)]
```

P1 is the official temporal split; P3 is a random 70/30 message-level split
(reported only to quantify inflation; never as detection performance).
