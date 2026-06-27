"""Quantization corruption stubs (M3)."""

from __future__ import annotations

from typing import Any


def quantize(model: Any, bits: int = 8, **kwargs: Any) -> Any:
    """Apply weight quantization to a model (M3).

    Later contract:
        - ``model``: full-precision model.
        - ``bits``: target bit width (e.g. 8, 4).
        - Returns quantized model ready for inference.

    Raises:
        NotImplementedError: M3 milestone not yet implemented.
    """
    raise NotImplementedError("quantize is stubbed for M3 (corruption milestone).")
