# Hollow Chains

Code and cached data for reproducing **Hollow Chains: Structural Fidelity
without Semantic Correctness in Sub-Billion Reasoning Models**. Sub-billion
language models can learn to emit well-formed reasoning traces—correct tag
order, teacher-like openings, low-entropy “think” spans—while remaining
semantically wrong. Structural Fidelity (SF) and Semantic Correctness (SC)
therefore decouple; the gap between them (form without substance) is
measurable and persists under scale and corruption sweeps described in the
paper.

## Figure and table map

| Paper artifact | Notebook | Cached result / data |
|----------------|----------|----------------------|
| Figure 1 (replication) | `notebooks/R0_replication.ipynb` | `data/results/` (R0 metrics JSON) |
| Table 2, Figure 2 (arithmetic control) | `notebooks/R4_control.ipynb` | `data/results/control_arith.json` |
| Table 3, Figure 3 (teacher / clean) | `notebooks/R2_clean.ipynb` | `data/results/r2clean.json` |
| Table 5 (MCQ control) | `notebooks/R4_control.ipynb` | `data/results/control_mcq.json` |
| Table 6, Figure 4 (entropy) | `notebooks/R5_entropy.ipynb` | `data/results/entropy.json` |
| Appendix scale table | `notebooks/R1_scale_sft.ipynb` | `data/sft/r2_*.jsonl` + scale metrics |
| Teacher-axis sweeps (supporting) | `notebooks/R2_teacher_axis.ipynb` | `data/sft/r2_*.jsonl` |
| MCQ SFT track | `notebooks/R3_mcq.ipynb` | `data/sft/mcq_sft.jsonl` |

Rebuild PDFs from cached JSON:

```bash
python paper/build_figures.py
# writes paper/figures/fig_{control,teacher,entropy,scale}.pdf
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate  # or Windows equivalent
pip install -e ".[dev,gpu]"
pytest tests/ -q
```

Pin `transformers>=4.51,<5` (5.x breaks GPTNeoX/Pythia loading used in
scale experiments). See `ENVIRONMENT.md` for full pins.

**One notebook on a Colab T4 (cached data):**

1. Clone the repo and `pip install -e ".[gpu]"`.
2. Place release JSONL/JSON under `data/` (see `data/README.md`).
3. Open `notebooks/R4_control.ipynb`, set `REPO_ROOT` to the clone path.
4. Run all cells; confirm `data/results/control_arith.json` and metrics
   match the paper table within floating-point tolerance.

Regenerate frontier teacher traces locally (API key via env only):

```bash
export TOGETHER_API_KEY=...   # never commit
python scripts/gen_teacher_data.py --teacher deepseek-v4 --n 200 --out-dir data/sft
```

Copy `r2_*.jsonl` into `data/sft/` (filename must stay `r2_<teacher>.jsonl`).

## Data

| Path | Contents |
|------|----------|
| `data/sft/reasoning_sft_v2.jsonl` | Combined reasoning distillation set |
| `data/sft/mcq_sft.jsonl` | SciQ-derived MCQ SFT |
| `data/sft/r2_*.jsonl` | Per-teacher caches (`question`, `prompt`, `completion`) |
| `data/eval/*.jsonl` | Frozen arith / MCQ eval sets |
| `data/results/*.json` | Precomputed SF/SC summaries for figures |

**Provenance:** Teacher names in filenames (`qwen3-0p6b`, `qwen3-1p7b`,
`deepseek-v4`, `qwen3p5-397b`). Frontier teachers via Together API; small
Qwen teachers via local GPU (`scripts/gen_teacher_data.py`).

**Licenses:** SciQ (MCQ source) is CC BY-NC 3.0. Teacher outputs are
model-generated; respect upstream model licenses. Code: MIT (`LICENSE`).

## Package layout

```
src/hollow_chains/   metrics (SF/SC/gap), train, eval, data loaders
configs/             experiment YAML
notebooks/           R0–R5 reproduction (+ legacy/ for old M2 notebooks)
scripts/             gen_teacher_data.py, compute_metrics entrypoints
data/                release JSONL + cached results
paper/               build_figures.py, figures/
tests/               CPU metrics tests + training smoke test
```

Metrics CLI (torch-free):

```bash
compute-metrics --records path/to/generations.jsonl --config configs/metrics.yaml --out report.json
```

## Citation

See `CITATION.cff`. Cite the paper title when available; author metadata
will be updated upon de-anonymization.

## Development

```bash
make test   # pytest + metrics coverage gate
make lint   # ruff + black
```
