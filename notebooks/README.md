# Experiment notebooks (R0–R5)

Paper reproduction notebooks. Each notebook is self-contained: mount or
point at the repo root, load cached data from `data/`, and write figures
to `paper/figures/` or summary JSON to `data/results/`.

| Notebook | Paper artifact | Primary inputs | Primary outputs |
|----------|----------------|----------------|-----------------|
| `R0_replication.ipynb` | Figure 1 | External reasoning checkpoints, eval tasks | `data/results/` metrics JSON |
| `R1_scale_sft.ipynb` | Appendix scale table | `data/sft/r2_*.jsonl`, HF base models | Scale-axis SFT metrics |
| `R2_teacher_axis.ipynb` | Teacher-axis sweeps | `data/sft/r2_*.jsonl` | Teacher comparison JSON |
| `R2_clean.ipynb` | Table 3, Figure 3 | `data/sft/reasoning_sft_v2.jsonl` | `data/results/r2clean.json` |
| `R3_mcq.ipynb` | MCQ SFT + eval | `data/sft/mcq_sft.jsonl`, SciQ eval | MCQ metrics JSON |
| `R4_control.ipynb` | Table 2 / Fig 2 (arith), Table 5 / Fig 3 (MCQ) | Frozen eval sets in `data/eval/` | `data/results/control_arith.json`, `control_mcq.json` |
| `R5_entropy.ipynb` | Table 6, Figure 4 | Generation records with entropies | `data/results/entropy.json` |

Drop the `.ipynb` files into this directory before running. Legacy M2
Colab notebooks live under `notebooks/legacy/`.

Expected runtime: one notebook end-to-end on a Colab T4 with cached
teacher traces is on the order of tens of minutes (no frontier re-generation).
