"""Visualization stubs (M4)."""

from __future__ import annotations

from typing import Any


def plot_sf_sc_scatter(report: Any, **kwargs: Any) -> Any:
    """Plot per-record SF vs SC scatter (M4).

    Later contract:
        - ``report``: GapReport or equivalent dict.
        - Returns figure object or saves to path.

    Raises:
        NotImplementedError: M4 milestone not yet implemented.
    """
    raise NotImplementedError("plot_sf_sc_scatter is stubbed for M4.")


def plot_four_way_bars(counts: dict[str, int], **kwargs: Any) -> Any:
    """Plot four-way classification bar chart (M4).

    Raises:
        NotImplementedError: M4 milestone not yet implemented.
    """
    raise NotImplementedError("plot_four_way_bars is stubbed for M4.")
