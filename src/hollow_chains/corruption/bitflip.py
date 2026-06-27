"""Bit-flip corruption stubs (M3)."""

from __future__ import annotations

from typing import Any


def apply_bitflip(model: Any, rate: float, **kwargs: Any) -> Any:
    """Apply random bit-flip corruption to model weights (M3).

    Later contract:
        - ``model``: model with weight tensors.
        - ``rate``: per-bit flip probability.
        - Returns corrupted model (same structure, perturbed weights).

    Raises:
        NotImplementedError: M3 milestone not yet implemented.
    """
    raise NotImplementedError("apply_bitflip is stubbed for M3 (corruption milestone).")
