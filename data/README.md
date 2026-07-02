# Data release layout

Cached distillation traces, frozen eval sets, and precomputed experiment
summaries for paper reproduction. Drop files from the release bundle here;
see row counts after files are present (`wc -l` for JSONL).

## `sft/` — distillation traces (JSONL)

| File | Rows (expected) | Schema keys | Provenance |
|------|-----------------|-------------|------------|
| `reasoning_sft_v2.jsonl` | ~2000 | `question`, `prompt`, `completion` | Mixed teachers (Qwen3 0.6B/1.7B, frontier API teachers); arithmetic word problems, seed 0 |
| `mcq_sft.jsonl` | ~500 | `question`, `prompt`, `completion` | SciQ-derived MCQ SFT; see license note below |
| `r2_deepseek-v4.jsonl` | 200 | `question`, `prompt`, `completion` | Together `deepseek-ai/DeepSeek-V4-Pro` |
| `r2_qwen3p5-397b.jsonl` | 200 | `question`, `prompt`, `completion` | Together `Qwen/Qwen3.5-397B-A17B` |
| `r2_qwen3-0p6b.jsonl` | 200 | `question`, `prompt`, `completion` | Local `Qwen/Qwen3-0.6B` teacher |
| `r2_qwen3-1p7b.jsonl` | 200 | `question`, `prompt`, `completion` | Local `Qwen/Qwen3-1.7B` teacher |

Prompt/completion format matches `scripts/gen_teacher_data.py` (open
`<|begin_of_thought|>` in masked prompt; M1 tag schema in completion).
Regenerate traces with `scripts/gen_teacher_data.py` if needed.

**Licenses:** Teacher outputs are model-generated (follow each base model
license). MCQ source: SciQ (CC BY-NC 3.0) — non-commercial use only.

Files over 90 MB are listed in `.gitignore`; use the download note in the
repo README if a file is omitted from git.

## `eval/` — frozen evaluation sets (JSONL)

| File | Description |
|------|-------------|
| `arith_eval.jsonl` | Arithmetic word problems (`id`, `prompt`, `gold`, `task_type`) |
| `mcq_eval.jsonl` | SciQ-style factual MCQ (`id`, `prompt`, `gold`, `task_type`) |

## `results/` — cached notebook summaries (JSON)

| File | Produced by | Contents |
|------|-------------|----------|
| `control_arith.json` | R4 | SF/SC/gap on arithmetic control harness |
| `control_mcq.json` | R4 | SF/SC/gap on MCQ control harness |
| `r2clean.json` | R2_clean | Teacher-axis clean comparison |
| `entropy.json` | R5 | Token-entropy profile summaries |

These JSON files feed `paper/build_figures.py` for PDF figures.
