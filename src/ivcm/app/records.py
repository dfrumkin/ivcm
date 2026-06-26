"""Per-image record and the fixed table ordering (pure, Qt-free, unit-tested)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(eq=False)
class ImageRecord:
    """One image in the review table.

    ``eq=False`` keeps identity-based equality/hash: records are unique per file, and the default
    field-wise ``__eq__`` would compare the numpy ``mask`` (an ambiguous truth value). This lets
    ``list.index(record)`` and set/dict membership work by identity.

    ``is_nerve`` is the checkbox state (the manual nerve/not-nerve label). ``mask`` and
    ``metrics`` are filled lazily by segmentation and cached, so they persist even after the
    row is unchecked (the overlay still shows in the preview/large view). ``metrics`` holds the
    ``cell_morphometry`` output (``{"density", "area", "perimeter"}``) once computed.
    """

    path: Path
    is_nerve: bool
    mask: np.ndarray | None = None
    metrics: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return self.path.name


def sort_key(record: ImageRecord) -> tuple[int, float, float, float, str]:
    """Fixed (non-user-changeable) order.

    Checked (nerve) rows come before unchecked rows; within the checked group, by density, then
    area, then perimeter (all descending — most prominent first), then name. Unchecked rows have
    their metrics treated as absent and order by name only, always below checked rows. The key is
    total, so the sort is well-defined regardless of cached metrics on unchecked rows.
    """
    metrics = record.metrics if record.is_nerve and record.metrics is not None else None
    density = metrics["density"] if metrics is not None else 0.0
    area = metrics["area"] if metrics is not None else 0.0
    perimeter = metrics["perimeter"] if metrics is not None else 0.0
    return (
        0 if record.is_nerve else 1,
        -density,
        -area,
        -perimeter,
        record.name.lower(),
    )


def sorted_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """The records in the fixed order (a stable sort, so equal keys keep input order)."""
    return sorted(records, key=sort_key)


def top3(records: list[ImageRecord]) -> list[ImageRecord]:
    """The first three checked rows with computed metrics, in the fixed order."""
    ordered = sorted_records(records)
    eligible = [r for r in ordered if r.is_nerve and r.metrics is not None]
    return eligible[:3]


def selection_after_toggle(
    sorted_records_list: list[ImageRecord],
    record: ImageRecord,
    now_checked: bool,
    prior_row: int,
) -> int | None:
    """Row to select after a checkbox toggle re-sorts ``sorted_records_list``.

    On **check** (``now_checked``), follow the toggled ``record`` to its new sorted position. On
    **uncheck**, keep the prior positional row when it is a valid index so the selection stays
    where the user was; if ``prior_row`` is out of range (e.g. no prior selection), return
    ``None`` for "no selection" rather than landing on an unrelated row. A failed check that
    reverts the record to unchecked arrives here with ``now_checked`` false and so takes the
    uncheck branch.
    """
    if now_checked:
        try:
            return sorted_records_list.index(record)
        except ValueError:
            return None
    if not (0 <= prior_row < len(sorted_records_list)):
        return None
    return prior_row
