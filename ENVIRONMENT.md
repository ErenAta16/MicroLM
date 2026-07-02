# Environment

Reproduce paper experiments on a single **16 GB GPU** (Colab T4 is
sufficient when using cached teacher JSONL under `data/sft/`).

## Python

- **3.12** (matches Colab and CI smoke tests)

## Core pins

| Package | Pin | Notes |
|---------|-----|-------|
| `transformers` | `>=4.51,<5` | **Required.** 5.x breaks GPTNeoX/Pythia and several Llama loaders used in scale SFT |
| `torch` | `2.x` + CUDA | CPU-only works for metrics tests; GPU for SFT/generation |
| `accelerate` | `>=0.33,<2` | HF Trainer |
| `datasets` | `>=2.20,<4` | Optional data utilities |
| `safetensors` | `>=0.4,<0.6` | Checkpoint I/O |
| `openai` | `>=1.0,<2` | Together API client for frontier teachers |
| `bitsandbytes` | optional | 4-bit local Qwen ≥1.7B on 4 GB GPUs |

Install:

```bash
pip install -e ".[dev,gpu]"
```

## Seeds

Paper sweeps use seeds **`{0, 1, 2}`** where multiple runs are reported.
Single-seed artifacts in the release bundle use seed **0** (word problems in
`scripts/gen_teacher_data.py`, default `SEED_DEFAULT=0`).

## Hardware notes

| Task | GPU |
|------|-----|
| Metrics / `pytest tests/test_metrics.py` | None |
| Pipeline smoke (`test_pipeline_smoke.py`) | None (CPU tiny model) |
| Scale SFT on Pythia-14m | ≥4 GB |
| Local Qwen3-0.6B teacher gen | ≥4 GB fp16 |
| Local Qwen3-1.7B teacher gen | ≥4 GB with bitsandbytes 4-bit |
| Frontier teachers | API only |

Run **one** model download or teacher job at a time to avoid HuggingFace
hub lock contention on shared caches.

## Verification

```bash
pytest tests/ -q
python scripts/gen_teacher_data.py --logic-check
```
