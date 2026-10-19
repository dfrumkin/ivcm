"""Per-image record and the fixed table ordering (pure, Qt-free, unit-tested)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_PINNED = 3  # at most this many rows may be pinned at once


@dataclass(eq=False)
class ImageRecord:
    """One image in the review table.

    ``eq=False`` keeps identity-based equality/hash: records are unique per file, and the default
    field-wise ``__eq__`` would compare the numpy ``mask`` (an ambiguous truth value). This lets
    ``list.index(record)`` and set/dict membership work by identity.

    ``is_nerve`` is the checkbox state (the manual nerve/not-nerve label). ``pinned`` is the Pin
    checkbox: a pinned row is one shown in the sample window. It is always used (``pinned`` implies
    ``is_nerve``), at most :data:`MAX_PINNED` rows may be pinned, and it does not affect the table
    order. ``mask`` and ``metrics`` are filled lazily by segmentation and cached, so they persist
    even after the row is unchecked (the overlay still shows in the preview/large view). ``mask``
    is the post-processed mask (cells below the minimum area removed) and ``metrics`` its
    ``postprocess_mask`` measures (``{"count", "density", "area", "perimeter"}``) once computed.
    """

    path: Path
    is_nerve: bool
    pinned: bool = False
    mask: np.ndarray | None = None
    metrics: dict[str, float] | None = None

    @property
    def name(self) -> str:
        return self.path.name


def pinned_count(records: list[ImageRecord]) -> int:
    """How many records are currently pinned (used to enforce :data:`MAX_PINNED`)."""
    return sum(1 for r in records if r.pinned)


def sort_key(record: ImageRecord) -> tuple[int, float, float, float, str]:
    """Fixed (non-user-changeable) order.

    Checked (nerve) rows come before unchecked rows; within the checked group, by count, then area,
    then perimeter (all descending — most prominent first), then name. Unchecked rows have their
    metrics treated as absent and order by name only, always below checked rows. Pins play no part.
    The key is total, so the sort is well-defined regardless of cached metrics on unchecked rows.
    """
    metrics = record.metrics if record.is_nerve and record.metrics is not None else None
    count = metrics["count"] if metrics is not None else 0.0
    area = metrics["area"] if metrics is not None else 0.0
    perimeter = metrics["perimeter"] if metrics is not None else 0.0
    return (
        0 if record.is_nerve else 1,
        -count,
        -area,
        -perimeter,
        record.name.lower(),
    )


def sorted_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """The records in the fixed order (a stable sort, so equal keys keep input order)."""
    return sorted(records, key=sort_key)


def selected_records(records: list[ImageRecord]) -> list[ImageRecord]:
    """The checked records that have computed metrics (the summarised and pinnable set)."""
    return [r for r in records if r.is_nerve and r.metrics is not None]


def _count(record: ImageRecord) -> float:
    return record.metrics["count"] if record.metrics is not None else 0.0


def pinned_display(records: list[ImageRecord]) -> list[ImageRecord]:
    """The pinned records in sample window order: count ascending, then name."""
    return sorted((r for r in records if r.pinned), key=lambda r: (_count(r), r.name.lower()))


def count_targets(records: list[ImageRecord]) -> tuple[float, float, float] | None:
    """The left/center/right slot targets: mean−SD, mean, mean+SD of count over the selected records.

    Uses the summary panel's sample SD; with a single selected record (SD undefined) all three
    targets are its count. ``None`` when nothing is selected.
    """
    counts = np.array([_count(r) for r in selected_records(records)], dtype=float)
    if counts.size == 0:
        return None
    mean = float(counts.mean())
    sd = float(counts.std(ddof=1)) if counts.size > 1 else 0.0
    return (mean - sd, mean, mean + sd)


def _closest(candidates: list[ImageRecord], target: float) -> ImageRecord | None:
    """The candidate whose count is closest to ``target`` (ties: the fixed table order)."""
    if not candidates:
        return None
    return min(candidates, key=lambda r: (abs(_count(r) - target), sort_key(r)))


def initial_pins(records: list[ImageRecord]) -> list[ImageRecord]:
    """Up to three distinct selected records closest in count to mean−SD, mean, and mean+SD."""
    targets = count_targets(records)
    if targets is None:
        return []
    candidates = selected_records(records)
    chosen: list[ImageRecord] = []
    for target in targets:
        pick = _closest([r for r in candidates if r not in chosen], target)
        if pick is None:
            break
        chosen.append(pick)
    return chosen


def replacement_for(records: list[ImageRecord], slot: int, excluded: set[ImageRecord]) -> ImageRecord | None:
    """The record to pin into sample-window ``slot`` (0 left, 1 center, 2 right) after an unpin there.

    Among selected, unpinned records not in ``excluded`` (the images pinned earlier this session),
    the one whose count is closest to the slot's target; ``None`` if nothing qualifies.
    """
    targets = count_targets(records)
    if targets is None:
        return None
    candidates = [r for r in selected_records(records) if not r.pinned and r not in excluded]
    return _closest(candidates, targets[slot])


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
