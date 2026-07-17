"""The selection-summary panel: the selected-image count plus one strip per metric (imports Qt).

Each strip is a small horizontal graph of one metric over a ``0``-to-``max`` axis, where ``max`` is
that metric's largest value across the selected images (so the axis scale is per-metric, and each
row labels its own axis). A strip draws, from back to front: the distribution of the values as a
histogram, the ±1 SD box, the mean, and min/max end ticks. In a small font under the axis it labels
the axis ends (``0`` and the maximum) and the mean under its marker; the SD and the minimum stay
off the strip, where they would cost the vertical space the preview needs, and are on hover.

The panel is a view over :func:`ivcm.app.summary.selection_summary`; all the arithmetic lives in
that Qt-free module.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPaintEvent
from PySide6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ivcm.app.records import ImageRecord
from ivcm.app.summary import METRIC_KEYS, MetricSummary, selection_summary

_BINS = 24  # histogram bins per strip; ~10px each at the panel's launch width
_GRAPH_HEIGHT = 26  # the drawn graph's height; the strip adds the ticks and the mean-text band
_STRIP_MIN_WIDTH = 140
_TICK = 4  # length of the min/max end ticks, in px
_SMALL_POINT_DROP = 1  # points shaved off the default font for the row labels
_AXIS_POINT_DROP = 2  # the under-axis numbers are smaller still, so their band costs little height
_LABEL_GAP = 6  # min px between the mean number and an axis-end number before the mean is dropped


def _shrink(font: QFont, points: int) -> QFont:
    """``font`` with its point size reduced by ``points`` (floored at 1)."""
    smaller = QFont(font)
    smaller.setPointSize(max(1, font.pointSize() - points))
    return smaller


def _small(widget: QLabel) -> QLabel:
    """Shrink a label's font so the row labels stay subordinate to the strips."""
    widget.setFont(_shrink(widget.font(), _SMALL_POINT_DROP))
    return widget


def _display(key: str) -> str:
    """A metric key as shown to the user: ``density`` -> ``Density`` (matching the table)."""
    return key.capitalize()


class _MetricStrip(QWidget):
    """One metric's horizontal distribution strip; hover shows the exact statistics.

    The widget's height is the graph, plus the tick overhang above and below it, plus a band at the
    bottom reserved for the under-axis numbers. The band is reserved rather than merely added: the
    graph is laid out above it, so the numbers cannot be clipped or drawn over.
    """

    def __init__(self, key: str) -> None:
        super().__init__()
        self._key = key
        self._summary: MetricSummary | None = None
        self._axis_font = _shrink(self.font(), _AXIS_POINT_DROP)
        self._band = QFontMetrics(self._axis_font).height()
        self.setFixedHeight(_GRAPH_HEIGHT + 2 * _TICK + self._band)
        self.setMinimumWidth(_STRIP_MIN_WIDTH)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_summary(self, summary: MetricSummary) -> None:
        self._summary = summary
        self.setToolTip(self._tooltip(summary))
        self.update()

    def _tooltip(self, summary: MetricSummary) -> str:
        if summary.mean is None or summary.minimum is None or summary.maximum is None:
            return f"{_display(self._key)}: no images selected"
        spread = "—" if summary.sd is None else f"{summary.sd:.2f}"
        return (
            f"{_display(self._key)}\n"
            f"mean ± SD: {summary.mean:.2f} ± {spread}\n"
            f"min–max: {summary.minimum:.2f}–{summary.maximum:.2f}\n"
            f"axis: 0–{summary.maximum:.2f}"
        )

    def _value_to_x(self, value: float, maximum: float, rect: QRect) -> float:
        """Position of ``value`` on the ``[0, maximum]`` axis, clamped to the strip.

        A degenerate ``maximum <= 0`` (every selected value is zero) has no axis extent, so every
        value collapses onto the origin rather than dividing by a zero range.
        """
        if maximum <= 0.0:
            return float(rect.left())
        fraction = min(max(value / maximum, 0.0), 1.0)
        return rect.left() + fraction * rect.width()

    def paintEvent(self, event: QPaintEvent) -> None:
        summary = self._summary
        if summary is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # The graph is inset by the tick length top and bottom so the baseline's end ticks stay
        # inside the widget, and by the text band so the numbers below have room of their own.
        rect = self.rect().adjusted(1, _TICK, -2, -(_TICK + self._band))
        palette = self.palette()

        baseline = rect.bottom()
        if summary.maximum is not None:
            self._paint_histogram(painter, summary, rect, baseline, palette.mid().color())
            self._paint_marks(painter, summary, rect, baseline, palette.highlight().color())
            self._paint_axis_text(painter, summary, rect, baseline, palette.text().color())

        painter.setPen(palette.mid().color())
        painter.drawLine(rect.left(), baseline, rect.right(), baseline)
        painter.end()

    def _paint_histogram(
        self, painter: QPainter, summary: MetricSummary, rect: QRect, baseline: int, color: QColor
    ) -> None:
        """Bars over ``[0, max]``, scaled so the tallest bin fills the strip."""
        tallest = max(summary.histogram)
        if tallest == 0:
            return
        bars = QColor(color)
        bars.setAlpha(150)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bars)
        bin_width = rect.width() / len(summary.histogram)
        for i, count in enumerate(summary.histogram):
            if count == 0:
                continue
            height = round(rect.height() * count / tallest)
            left = round(rect.left() + i * bin_width)
            width = max(1, round(rect.left() + (i + 1) * bin_width) - left - 1)
            painter.fillRect(QRect(left, baseline - height, width, height), bars)

    def _paint_marks(
        self, painter: QPainter, summary: MetricSummary, rect: QRect, baseline: int, color: QColor
    ) -> None:
        """The ±1 SD box, the mean line, and the min/max end ticks."""
        if summary.mean is None or summary.minimum is None or summary.maximum is None:
            return
        maximum = summary.maximum

        # The SD box is drawn only when the SD is measurable: a zero-width box at n = 1 would read
        # as "no spread" rather than "not measurable".
        if summary.sd is not None:
            box_left = self._value_to_x(summary.mean - summary.sd, maximum, rect)
            box_right = self._value_to_x(summary.mean + summary.sd, maximum, rect)
            box = QColor(color)
            box.setAlpha(60)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.fillRect(
                QRect(round(box_left), rect.top(), max(1, round(box_right - box_left)), rect.height()),
                box,
            )

        painter.setPen(color)
        for value in (summary.minimum, summary.maximum):
            tick_x = round(self._value_to_x(value, maximum, rect))
            painter.drawLine(tick_x, baseline - _TICK, tick_x, baseline + _TICK)

        mean_x = round(self._value_to_x(summary.mean, maximum, rect))
        painter.drawLine(mean_x, rect.top(), mean_x, baseline)

    def _paint_axis_text(
        self, painter: QPainter, summary: MetricSummary, rect: QRect, baseline: int, color: QColor
    ) -> None:
        """The band under the axis: ``0`` at its start, the maximum at its end, the mean centred.

        The mean is drawn only where it fits centred under its marker without running into an end
        number -- a mean that has collapsed onto an axis end (a single image, or an all-zero metric)
        is already the end number itself, and nudging its label aside would only point it at a value
        it does not sit on. The tooltip carries the figure regardless.
        """
        if summary.mean is None or summary.maximum is None:
            return
        metrics = QFontMetrics(self._axis_font)
        painter.setFont(self._axis_font)
        painter.setPen(color)

        top = baseline + _TICK
        zero_width = metrics.horizontalAdvance("0")
        maximum_text = f"{summary.maximum:.2f}"
        maximum_width = metrics.horizontalAdvance(maximum_text)
        painter.drawText(QRect(rect.left(), top, zero_width, self._band), Qt.AlignmentFlag.AlignCenter, "0")
        painter.drawText(
            QRect(rect.right() - maximum_width, top, maximum_width, self._band),
            Qt.AlignmentFlag.AlignCenter,
            maximum_text,
        )

        mean_text = f"{summary.mean:.2f}"
        mean_width = metrics.horizontalAdvance(mean_text)
        left = round(self._value_to_x(summary.mean, summary.maximum, rect) - mean_width / 2)
        fits = (
            left >= rect.left() + zero_width + _LABEL_GAP
            and left + mean_width <= rect.right() - maximum_width - _LABEL_GAP
        )
        if fits:
            painter.drawText(QRect(left, top, mean_width, self._band), Qt.AlignmentFlag.AlignCenter, mean_text)


class SummaryPanel(QWidget):
    """The count centred on top, one :class:`_MetricStrip` per metric stacked beneath it.

    :meth:`refresh` recomputes the summary from the records and repaints; with nothing selected the
    strips give way to a placeholder. The panel's height is fixed -- including when the placeholder
    is showing -- so the preview below it keeps the rest of the pane and does not resize as images
    are selected and deselected.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

        self._count = QLabel()
        self._count.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._strips: dict[str, _MetricStrip] = {}
        self._rows = QWidget()
        grid = QGridLayout(self._rows)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setVerticalSpacing(2)
        for row, key in enumerate(METRIC_KEYS):
            strip = _MetricStrip(key)
            self._strips[key] = strip
            grid.addWidget(_small(QLabel(_display(key))), row, 0)
            grid.addWidget(strip, row, 1)
        grid.setColumnStretch(1, 1)

        self._placeholder = QLabel("No images selected")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._count)
        layout.addWidget(self._rows)
        layout.addWidget(self._placeholder)

        # The rows give way to the (shorter) placeholder when nothing is selected, so the height is
        # pinned to the count plus the rows: the panel keeps one size whatever is showing.
        self.setFixedHeight(self._count.sizeHint().height() + layout.spacing() + self._rows.sizeHint().height())
        self.refresh([])

    def refresh(self, records: list[ImageRecord]) -> None:
        """Recompute the summary from ``records`` and repaint the panel."""
        summary = selection_summary(records, bins=_BINS)
        self._count.setText(f"Count: {summary.count}")
        self._rows.setVisible(summary.count > 0)
        self._placeholder.setVisible(summary.count == 0)
        for key, metric in summary.metrics.items():
            self._strips[key].set_summary(metric)
