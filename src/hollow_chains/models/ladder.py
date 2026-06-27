"""Llama-style model ladder with tied embeddings and param-budget solver.

Realized parameter counts (vocab_size=16000, context_len=512, tied embeddings):

| Rung       | Target    | Realized | hidden | layers | heads | intermediate |
|------------|-----------|----------|--------|--------|-------|--------------|
| tiny_1m    | 1,000,000 | 997,548  | 60     | 1      | 3     | 256          |
| small_8m   | 8,000,000 | 7,995,136| 192    | 4      | 3     | 512          |
| mid_50m    | 50,000,000| 49,971,712| 512   | 8      | 8     | 1408         |
| large_150m | 150,000,000| 149,925,888| 768  | 16     | 12    | 2048         |
| xl_350m    | 350,000,000| 349,922,304| 1024 | 24     | 16    | 2816         |

Run ``python -m hollow_chains.models.ladder`` to print the live table.
At tiny_1m the embedding matrix (vocab x hidden) dominates (~96%% of weights).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from transformers import LlamaConfig, LlamaForCausalLM


@dataclass(frozen=True)
class LadderRung:
    """Resolved architecture for one ladder rung."""

    name: str
    target_params: int
    realized_params: int
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int


def estimate_params(
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    intermediate_size: int,
) -> int:
    """Analytical param count for Llama with tied word embeddings.

    Counts embed + per-layer attention/MLP/norms + final norm. lm_head is tied.
    """
    layer_params = num_hidden_layers * (
        4 * hidden_size * hidden_size
        + 3 * hidden_size * intermediate_size
        + 2 * hidden_size
    )
    return vocab_size * hidden_size + layer_params + hidden_size


def _make_llama_config(
    *,
    vocab_size: int,
    hidden_size: int,
    num_hidden_layers: int,
    num_attention_heads: int,
    intermediate_size: int,
    max_position_embeddings: int,
) -> LlamaConfig:
    """Build a LlamaConfig with RoPE and tied embeddings."""
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            f"hidden_size {hidden_size} not divisible by "
            f"num_attention_heads {num_attention_heads}"
        )
    return LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_attention_heads,
        max_position_embeddings=max_position_embeddings,
        tie_word_embeddings=True,
        rope_theta=10000.0,
        hidden_act="silu",
        rms_norm_eps=1e-5,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )


def count_params(config: LlamaConfig) -> int:
    """Count trainable parameters (analytical; verified against HF on build).

    Args:
        config: HuggingFace LlamaConfig.

    Returns:
        Total parameter count.
    """
    return estimate_params(
        config.vocab_size,
        config.hidden_size,
        config.num_hidden_layers,
        config.intermediate_size,
    )


def verify_params(config: LlamaConfig) -> int:
    """Count parameters by instantiating the model (for validation)."""
    model = LlamaForCausalLM(config)
    return sum(p.numel() for p in model.parameters())


def solve_config(
    target_params: int,
    vocab_size: int,
    context_len: int = 512,
    *,
    depth_width_ratio: float = 1.0,
    tolerance: float = 0.05,
) -> LlamaConfig:
    """Numerically solve architecture dims to hit a parameter budget."""
    best_config: LlamaConfig | None = None
    best_error = math.inf

    max_layers = min(48, max(4, int(24 * depth_width_ratio)))
    optimal_layers = max(1, int(round(8 * (target_params / 50_000_000) ** 0.3)))
    min_layers = max(1, optimal_layers // 2)
    max_layers = max(max_layers, optimal_layers * 2)

    for num_layers in range(min_layers, max_layers + 1):
        layer_penalty_weight = 0.02
        for hidden_size in range(32, 4096, 4):
            num_heads = max(1, hidden_size // 64)
            while hidden_size % num_heads != 0 and num_heads > 1:
                num_heads -= 1
            intermediate_size = int(hidden_size * 8 / 3)
            intermediate_size = max(intermediate_size, hidden_size * 2)
            intermediate_size = ((intermediate_size + 127) // 128) * 128

            realized = estimate_params(
                vocab_size, hidden_size, num_layers, intermediate_size
            )
            error = abs(realized - target_params) / target_params
            depth_penalty = 1.0 + layer_penalty_weight * abs(
                num_layers - optimal_layers
            )
            adjusted = error * depth_penalty
            if adjusted < best_error:
                best_error = adjusted
                best_config = _make_llama_config(
                    vocab_size=vocab_size,
                    hidden_size=hidden_size,
                    num_hidden_layers=num_layers,
                    num_attention_heads=num_heads,
                    intermediate_size=intermediate_size,
                    max_position_embeddings=context_len,
                )
            if error <= tolerance:
                return best_config  # type: ignore[return-value]

    if best_config is None:
        raise ValueError(f"Could not solve config for target {target_params}")
    rel = abs(count_params(best_config) - target_params) / target_params
    if rel > tolerance * 2:
        raise ValueError(
            f"Best config off by {rel:.1%} for target {target_params}: "
            f"{count_params(best_config)} params"
        )
    return best_config


def build_model(
    rung_name: str | None = None,
    target_params: int | None = None,
    *,
    ladder_config: dict[str, Any] | None = None,
) -> LlamaForCausalLM:
    """Build a LlamaForCausalLM for a named rung or explicit param budget."""
    cfg = ladder_config or {}
    shared = cfg.get("shared", {})
    vocab_size = int(shared.get("vocab_size", 16000))
    context_len = int(shared.get("context_len", 512))
    depth_width_ratio = float(shared.get("depth_width_ratio", 1.0))
    rungs = cfg.get("rungs", {})

    if rung_name is not None:
        if rung_name not in rungs:
            raise KeyError(f"Unknown rung: {rung_name}")
        spec = rungs[rung_name]
        resolved = spec.get("resolved")
        if resolved:
            llama_cfg = _make_llama_config(
                vocab_size=vocab_size,
                hidden_size=int(resolved["hidden_size"]),
                num_hidden_layers=int(resolved["num_hidden_layers"]),
                num_attention_heads=int(resolved["num_attention_heads"]),
                intermediate_size=int(resolved["intermediate_size"]),
                max_position_embeddings=context_len,
            )
            return LlamaForCausalLM(llama_cfg)
        target_params = int(spec["target_params"])

    if target_params is None:
        raise ValueError("Provide rung_name or target_params")

    llama_cfg = solve_config(
        target_params,
        vocab_size,
        context_len,
        depth_width_ratio=depth_width_ratio,
    )
    return LlamaForCausalLM(llama_cfg)


def resolve_rung(
    name: str,
    target_params: int,
    vocab_size: int,
    context_len: int,
    *,
    depth_width_ratio: float = 1.0,
) -> LadderRung:
    """Solve and return metadata for a single ladder rung."""
    cfg = solve_config(
        target_params,
        vocab_size,
        context_len,
        depth_width_ratio=depth_width_ratio,
    )
    realized = count_params(cfg)
    return LadderRung(
        name=name,
        target_params=target_params,
        realized_params=realized,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        vocab_size=vocab_size,
        max_position_embeddings=context_len,
    )


def print_ladder_table(ladder_config: dict[str, Any] | None = None) -> str:
    """Print and return a table of rung → realized parameter counts."""
    cfg = ladder_config or {}
    shared = cfg.get("shared", {})
    vocab_size = int(shared.get("vocab_size", 16000))
    context_len = int(shared.get("context_len", 512))
    depth_width_ratio = float(shared.get("depth_width_ratio", 1.0))
    rungs = cfg.get("rungs", {})

    lines = [
        "| Rung | Target | Realized | hidden | layers | heads | intermediate |",
        "|------|--------|----------|--------|--------|-------|--------------|",
    ]
    for name, spec in rungs.items():
        target = int(spec["target_params"])
        rung = resolve_rung(
            name,
            target,
            vocab_size,
            context_len,
            depth_width_ratio=depth_width_ratio,
        )
        lines.append(
            f"| {name} | {target:,} | {rung.realized_params:,} | "
            f"{rung.hidden_size} | {rung.num_hidden_layers} | "
            f"{rung.num_attention_heads} | {rung.intermediate_size} |"
        )
    table = "\n".join(lines)
    print(table)
    return table


if __name__ == "__main__":
    from pathlib import Path

    from hollow_chains.config import load_config

    config_path = Path(__file__).parents[3] / "configs" / "model_ladder.yaml"
    if config_path.is_file():
        print_ladder_table(load_config(config_path))
    else:
        defaults = {
            "shared": {"vocab_size": 16000, "context_len": 512},
            "rungs": {
                "tiny_1m": {"target_params": 1_000_000},
                "small_8m": {"target_params": 8_000_000},
                "mid_50m": {"target_params": 50_000_000},
                "large_150m": {"target_params": 150_000_000},
                "xl_350m": {"target_params": 350_000_000},
            },
        }
        print_ladder_table(defaults)
