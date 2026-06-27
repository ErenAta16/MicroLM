"""Result aggregation stubs (M2–M4)."""

from __future__ import annotations

from typing import Any


def aggregate(results: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
    """Aggregate evaluation runs into a unified report (M2–M4).

    Later contract:
        - ``results``: list of per-run gap reports.
        - Returns merged summary with curves and statistics.

    Raises:
        NotImplementedError: M2–M4 milestone not yet implemented.
    """
    raise NotImplementedError("aggregate is stubbed for M2–M4.")
