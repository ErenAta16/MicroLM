"""Degradation evaluation stubs (M3–M4)."""

from __future__ import annotations

from typing import Any


def run_degradation(config: Any, **kwargs: Any) -> dict[str, Any]:
    """Run degradation-axis evaluation (bit-flip + quantize) (M3–M4).

    Later contract:
        - Apply corruption at increasing severity.
        - Generate traces, compute SF/SC/gap per corruption level.
        - Returns aggregated results dict.

    Raises:
        NotImplementedError: M3–M4 milestone not yet implemented.
    """
    raise NotImplementedError("run_degradation is stubbed for M3–M4.")
