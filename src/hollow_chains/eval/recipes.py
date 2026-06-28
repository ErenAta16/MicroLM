"""Validated generation recipes for external HF checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from hollow_chains.data.sft_format import (
    format_instruct_prompt,
    format_reasoning_eval_prompt,
)
from hollow_chains.metrics.parse import TAG_BEGIN_THOUGHT

REASONING_PREPEND = f"{TAG_BEGIN_THOUGHT}\n"


@dataclass(frozen=True)
class GenerationRecipe:
    """Prompt builder, decoding kwargs, and post-process hook name."""

    name: str
    format_prompt: Any
    decoding: dict[str, Any]
    prepend_begin_thought: bool = False


REASONING_RECIPE = GenerationRecipe(
    name="reasoning",
    format_prompt=format_reasoning_eval_prompt,
    decoding={
        "do_sample": True,
        "temperature": 0.3,
        "top_k": 25,
        "top_p": 0.8,
        "repetition_penalty": 1.3,
        "max_new_tokens": 512,
    },
    prepend_begin_thought=True,
)

INSTRUCT_RECIPE = GenerationRecipe(
    name="instruct",
    format_prompt=format_instruct_prompt,
    decoding={
        "do_sample": True,
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "max_new_tokens": 300,
    },
    prepend_begin_thought=False,
)

RECIPES: dict[str, GenerationRecipe] = {
    "reasoning": REASONING_RECIPE,
    "instruct": INSTRUCT_RECIPE,
}


def get_recipe(name: str) -> GenerationRecipe:
    """Return a registered recipe by name."""
    if name not in RECIPES:
        raise KeyError(f"Unknown generation recipe: {name}")
    return RECIPES[name]


def postprocess_generation(text: str, recipe: GenerationRecipe) -> str:
    """Apply recipe-specific post-decode cleanup."""
    cleaned = text.replace("<s>", "").replace("</s>", "").strip()
    if recipe.prepend_begin_thought and not cleaned.startswith(TAG_BEGIN_THOUGHT):
        cleaned = REASONING_PREPEND + cleaned
    return cleaned
