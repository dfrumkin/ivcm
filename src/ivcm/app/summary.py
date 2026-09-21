"""Per-metric summary of the selected images (pure, Qt-free, unit-tested).

The selected set is the checked records with computed metrics -- the same eligibility rule
:func:`ivcm.app.records.top3` uses. For each metric (``count``, ``density``, ``area``, ``perimeter``)
this reduces those records to what the summary strips draw: the mean, the sample standard deviation,
the min/max, and a histogram of the values over the ``[0, max]`` axis.

Statistics that are not measurable are ``None`` rather than a sentinel number, so the drawing code
has one explicit "nothing to draw" case: everything is ``None`` for an empty selection, and ``sd``
is ``None`` for a single image (a sample SD needs two).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ivcm.app.records import ImageRecord

METRIC_KEYS = ("count", "density", "area", "perimeter")


@dataclass(frozen=True)
class MetricSummary:
    """One metric's statistics over the selected images.

    ``histogram`` holds ``bins`` counts spanning ``[0, maximum]`` and always totals the selected
    count, so every selected image is accounted for in what is drawn. ``mean``/``minimum``/
    ``maximum`` are ``None`` only for an empty selection; ``sd`` is additionally ``None`` for a
    single image.
    """

    mean: float | None
    sd: float | None
    minimum: float | None
    maximum: float | None
    histogram: tuple[int, ...]


@dataclass(frozen=True)
class SelectionSummary:
    """The selected-image count and one :class:`MetricSummary` per key in :data:`METRIC_KEYS`."""

    count: int
    metrics: dict[str, MetricSummary]


def selected_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """The checked records that have computed metrics (the summarised set)."""
    return [r for r in records if r.is_nerve and r.metrics is not None]


def _histogram(values: np.ndarray, maximum: float, bins: int) -> tuple[int, ...]:
    """Counts of ``values`` in ``bins`` equal bins spanning ``[0, maximum]``.

    Bins are half-open (``[lo, hi)``) except the last, which is closed -- ``np.histogram``'s own
    convention -- so the image sitting exactly at ``maximum`` lands in the final bin instead of
    falling outside the histogram. A degenerate ``maximum == 0`` (every selected value is zero) has
    no axis extent to divide by, so all the values are placed in the first bin by hand.
    """
    if maximum <= 0.0:
        return (values.size,) + (0,) * (bins - 1)
    counts, _ = np.histogram(values, bins=bins, range=(0.0, maximum))
    return tuple(int(c) for c in counts)


def _summarise(values: np.ndarray, bins: int) -> MetricSummary:
    if values.size == 0:
        return MetricSummary(mean=None, sd=None, minimum=None, maximum=None, histogram=(0,) * bins)
    maximum = float(values.max())
    # Sample SD (ddof=1): "the standard deviation across the selected images". Undefined for a
    # single image -- numpy would warn and yield nan, so the n < 2 case is answered directly.
    sd = float(values.std(ddof=1)) if values.size > 1 else None
    return MetricSummary(
        mean=float(values.mean()),
        sd=sd,
        minimum=float(values.min()),
        maximum=maximum,
        histogram=_histogram(values, maximum, bins),
    )


def selection_summary(records: list[ImageRecord], *, bins: int) -> SelectionSummary:
    """Summarise the selected images: the count plus per-metric statistics and histogram.

    ``bins`` is the (fixed) number of histogram bins each metric's values are binned into.
    """
    if bins < 1:
        raise ValueError(f"bins must be >= 1; got {bins}.")
    selected = selected_records(records)
    metrics: dict[str, MetricSummary] = {}
    for key in METRIC_KEYS:
        # `selected` guarantees `metrics is not None`; the check keeps the type checker informed.
        values = np.array([r.metrics[key] for r in selected if r.metrics is not None], dtype=float)
        metrics[key] = _summarise(values, bins)
    return SelectionSummary(count=len(selected), metrics=metrics)
