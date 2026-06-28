# hollow-chains

Measure **Structural Fidelity (SF)** vs **Semantic Correctness (SC)** in reasoning traces from tiny language models — and the gap between them ("theater": well-formed but wrong).

## Install

```bash
pip install -e ".[dev]"        # M1 metrics only (CPU)
pip install -e ".[dev,gpu]"    # M2 training + generation (Colab / GPU)
```

Target Python: **3.12** (Colab parity).

## M1 — Metrics

Compute metrics on a JSONL file of `GenerationRecord` samples:

```bash
compute-metrics --records data/records.jsonl --config configs/metrics.yaml --out report.json
```

## M2 — Train → Generate → Metrics

```mermaid
flowchart LR
  A[Tokenizer] --> B[Pretrain shard]
  B --> C[Pretrain ladder]
  C --> D[SFT reasoning traces]
  D --> E[generate_records]
  E --> F[GenerationRecord JSONL]
  F --> G[M1 compute-metrics]
```

1. **Tokenizer** — one frozen ByteLevel BPE (`configs/tokenizer.yaml`), reasoning tags imported from M1.
2. **Pretrain** — causal LM on FineWeb-Edu shard for each ladder rung.
3. **SFT** — reasoning traces with M1 tag schema; one-axis sweeps in `configs/sft.yaml`.
4. **Generate** — `generate_records()` writes schema-valid JSONL with optional token entropies.
5. **Metrics** — same M1 CLI; no torch in the metrics layer.

### Model ladder (vocab=16000, ctx=512, tied embeddings)

| Rung | Target | Realized | hidden | layers | heads | intermediate |
|------|--------|----------|--------|--------|-------|--------------|
| tiny_1m | 1,000,000 | 951,720 | 56 | 1 | 1 | 256 |
| small_8m | 8,000,000 | 7,661,120 | 320 | 2 | 5 | 896 |
| mid_50m | 50,000,000 | 47,679,588 | 836 | 4 | 11 | 2304 |
| large_150m | 150,000,000 | 142,774,164 | 1404 | 5 | 18 | 3840 |
| xl_350m | 350,000,000 | 333,041,312 | 1888 | 7 | 16 | 5120 |

Print live table: `python -m hollow_chains.models.ladder`

At **tiny_1m**, ~96% of weights are the shared vocab embedding matrix — expected for this vocab size.

### Colab notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/00_setup_colab.ipynb` | Mount Drive, train tokenizer, materialize pretrain shard |
| `notebooks/01_pretrain_ladder.ipynb` | Pretrain each rung |
| `notebooks/02_sft_sweeps.ipynb` | Build SFT data + run sweep cells |
| `notebooks/03_generate_emergence.ipynb` | Generate JSONL + M1 reports |

Each notebook has a config cell at the top and lists artifacts written at the end.

### Local CPU smoke test

Verifies the full loop without network/GPU:

```bash
pytest tests/test_pipeline_smoke.py -v
```

Runs: micro tokenizer → tiny_1m pretrain (2 steps) → SFT (1 step) → 1 generation → M1 metrics.

## Generation recipes

Validated prompt formats and decoding for external HuggingFace checkpoints (`generate_with_recipe` in `eval/generate.py`).

### Reasoning (`SupraLabs/*-Reasoning`)

| Field | Value |
|-------|-------|
| System | Your role as an assistant involves thoroughly exploring questions through a systematic long thinking process before providing the final precise and accurate solutions. |
| Prompt | `[SYSTEM]: {system}\n\n[USER]: {question}\n\n[ASSISTANT]: <|begin_of_thought|>\n` |
| Decoding | `do_sample=True`, `temperature=0.3`, `top_k=25`, `top_p=0.8`, `repetition_penalty=1.3`, `max_new_tokens=512` |
| Post | Decode with `skip_special_tokens=False`, strip `<s>`/`</s>`, prepend `<|begin_of_thought|>\n` |

### Instruct (`SupraLabs/*-Instruct`)

| Field | Value |
|-------|-------|
| Prompt | Alpaca: `Below is an instruction... ### Instruction:\n{question}\n\n### Response:\n` |
| Decoding | `do_sample=True`, `temperature=0.7`, `top_k=50`, `top_p=0.9`, `repetition_penalty=1.15`, `max_new_tokens=300` |
| Post | No prepend |

### Qwen3 teacher (SFT data)

| Field | Value |
|-------|-------|
| Models | `Qwen/Qwen3-0.6B`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B` |
| Prompt | `apply_chat_template(..., enable_thinking=True)` |
| Decoding | `do_sample=True`, `temperature=0.6`, `top_p=0.95`, `top_k=20`, `max_new_tokens=512` |
| SFT prompt | Same reasoning format as eval (open `<|begin_of_thought|>` in masked prompt) |
| External-base SFT defaults | `epochs=6`, `lr=3e-4`, `batch_size=4`, `max_len=768` (`configs/scale_ladder.yaml`) |

### R2 teacher data (local CLI)

Generate teacher SFT JSONL caches for the R2 Colab notebook:

```bash
pip install -e ".[gpu]"   # torch, transformers, openai, bitsandbytes (optional 4-bit)
export TOGETHER_API_KEY=... # for deepseek-v4 / qwen3p5-397b
python scripts/gen_teacher_data.py --teacher qwen3-0p6b --n 200
python scripts/gen_teacher_data.py --teacher all --n 200 --out-dir ./teacher_cache
```

Writes `r2_{teacher}.jsonl` (e.g. `r2_qwen3-0p6b.jsonl`) to `--out-dir`. Upload these files to Google Drive **`MyDrive/MicroLM/sft_data/`** so the R2 Colab notebook cache-hits and skips re-generation.

Teachers: `qwen3-0p6b`, `qwen3-1p7b` (local GPU), `deepseek-v4`, `qwen3p5-397b` (Together API).

## Development

```bash
make test    # M1 metrics (>90% coverage) + M2 smoke test
make lint    # ruff + black
```

## Milestones

| Milestone | Scope | Status |
|-----------|-------|--------|
| **M1** | Metrics layer, schema, parser, CLI | **Done** |
| **M2** | Tokenizer, ladder, pretrain, SFT, generate, Colab | **Done** |
| **M3** | Bit-flip + quantization corruption | Pending |
| **M4** | Degradation eval, aggregation, visualization | Pending |

## Project structure

```
MicroLM/
├── configs/
│   ├── metrics.yaml
│   ├── tokenizer.yaml
│   ├── model_ladder.yaml
│   ├── pretrain.yaml
│   ├── sft.yaml
│   ├── scale_ladder.yaml
│   ├── generate.yaml
│   └── smoke.yaml
├── notebooks/          # Colab orchestration
├── scripts/
│   ├── compute_metrics.py
│   └── train_tokenizer.py
├── src/hollow_chains/
│   ├── data/           # schema, tokenizer, pretrain_data, build_reasoning_sft
│   ├── metrics/        # M1 — torch-free
│   ├── models/         # ladder (Llama + param solver)
│   ├── train/          # pretrain, sft
│   ├── eval/           # generate, run_emergence
│   └── ...
└── tests/
    ├── test_metrics.py
    └── test_pipeline_smoke.py
```
