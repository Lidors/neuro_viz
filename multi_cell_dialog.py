"""
multi_cell_dialog.py  –  Standalone window for comparing multiple units.

Layout:
  Top bar    : Normalize checkbox  |  Merge cells checkbox  |  event checkboxes  |  Refresh
  Splitter:
    Upper    : Mean PSTH  – one PlotWidget per selected event, all cells overlaid
                           (or single merged average when Merge is checked)
    Lower    : Block metrics – max / min / range / t_max / t_min vs block #
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QGridLayout, QSplitter,
    QWidget, QLabel, QCheckBox, QPushButton, QScrollArea, QFrame,
    QFileDialog
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from psth_panel import _render_widget_to_pdf

BG_COLOR   = '#16213e'
FG_COLOR   = '#e0e0e0'
ZERO_COLOR = '#555577'

CELL_COLORS = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0',
    '#00BCD4', '#FF5722', '#8BC34A', '#E91E63', '#607D8B',
    '#CDDC39', '#795548',
]

METRIC_KEYS   = ['amplitude', 'best_lag_ms']
METRIC_LABELS = ['Amplitude', 'Best Lag (ms)']


def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _styled_pw(title: str, x_label: str, y_label: str,
               xlim=None, small: bool = False) -> pg.PlotWidget:
    pw = pg.PlotWidget()
    pw.setBackground(BG_COLOR)
    pw.showGrid(x=True, y=True, alpha=0.15)
    fs = '8pt' if small else '10pt'
    for ax in ('bottom', 'left'):
        pw.getAxis(ax).setPen(FG_COLOR)
        pw.getAxis(ax).setTextPen(FG_COLOR)
        if small:
            pw.getAxis(ax).setStyle(tickFont=QFont('Arial', 7))
    pw.setLabel('bottom', x_label, color=FG_COLOR)
    pw.setLabel('left',   y_label, color=FG_COLOR)
    pw.setTitle(title, color=FG_COLOR, size=fs)
    if xlim:
        pw.setXRange(*xlim, padding=0)
    return pw


def _clear_grid(grid: QGridLayout):
    while grid.count():
        item = grid.takeAt(0)
        w = item.widget()
        if w:
            w.hide()
            w.deleteLater()


class MultiCellDialog(QDialog):
    """Non-modal window: select events via checkboxes, click Refresh to update."""

    def __init__(self, parent, data, get_selection_fn,
                 get_xlim_fn, get_smooth_fn, get_block_size_fn):
        super().__init__(parent)
        self.setWindowTitle("Compare selected units")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1100, 750)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._data             = data
        self._get_selection    = get_selection_fn
        self._get_xlim         = get_xlim_fn          # callable → (lo, hi)
        self._get_smooth       = get_smooth_fn        # callable → int
        self._get_block_size   = get_block_size_fn    # callable → int
        self._ev_checks: dict[str, QCheckBox] = {}

        self._build_ui()
        self._draw()

    # ── UI ────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Top control bar ──────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(10)

        self._norm_cb = QCheckBox("Normalize (0–1)")
        self._norm_cb.setStyleSheet(
            f"QCheckBox {{ color: {FG_COLOR}; font-size: 9pt; }}")
        bar.addWidget(self._norm_cb)
        bar.addSpacing(8)

        self._merge_cb = QCheckBox("Merge cells")
        self._merge_cb.setStyleSheet(
            f"QCheckBox {{ color: #FFD54F; font-size: 9pt; }}"
            f"QCheckBox::indicator:checked   {{ background: #FFD54F; "
            f"  border: 2px solid #FFD54F; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #1a1a3e; "
            f"  border: 2px solid #FFD54F; border-radius: 3px; }}"
        )
        bar.addWidget(self._merge_cb)
        bar.addSpacing(16)

        # Event checkboxes in a horizontal scroll area
        bar.addWidget(self._make_label("Events:"))
        ev_widget = QWidget()
        ev_layout = QHBoxLayout(ev_widget)
        ev_layout.setContentsMargins(0, 0, 0, 0)
        ev_layout.setSpacing(8)
        for key in self._data.event_keys:
            cb = self._make_ev_checkbox(key)
            ev_layout.addWidget(cb)
            self._ev_checks[key] = cb
        if self._ev_checks:
            list(self._ev_checks.values())[0].setChecked(True)

        scroll = QScrollArea()
        scroll.setWidget(ev_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(32)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        bar.addWidget(scroll, stretch=1)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(26)
        refresh_btn.setStyleSheet(
            "QPushButton { background: #1a1a3e; color: #e0e0e0; "
            "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background: #2a2a5e; }"
        )
        refresh_btn.clicked.connect(self._draw)
        bar.addWidget(refresh_btn)

        pdf_btn = QPushButton("Save PDF")
        pdf_btn.setFixedHeight(26)
        pdf_btn.setStyleSheet(
            "QPushButton { background: #1a2a1a; color: #a5d6a7; "
            "border: 1px solid #2a5a2a; border-radius: 3px; padding: 2px 10px; }"
            "QPushButton:hover { background: #1e3a1e; }"
        )
        pdf_btn.clicked.connect(self._save_pdf)
        bar.addWidget(pdf_btn)
        outer.addLayout(bar)

        # ── Cell-colour legend ───────────────────────────────────────────────
        self._legend_lbl = QLabel()
        self._legend_lbl.setStyleSheet(f"color: {FG_COLOR}; font-size: 9pt;")
        self._legend_lbl.setWordWrap(True)
        outer.addWidget(self._legend_lbl)

        # ── Main splitter: PSTH (top) | metrics (bottom) ─────────────────────
        splitter = self._splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")

        # Top: mean PSTH grid
        self._psth_container = QWidget()
        self._psth_container.setStyleSheet(f"background: {BG_COLOR};")
        self._psth_grid = QGridLayout(self._psth_container)
        self._psth_grid.setContentsMargins(0, 0, 0, 0)
        self._psth_grid.setSpacing(4)
        splitter.addWidget(self._psth_container)

        # Bottom: block-metric mini-plots
        self._metric_container = QWidget()
        self._metric_container.setStyleSheet(f"background: {BG_COLOR};")
        self._metric_grid = QGridLayout(self._metric_container)
        self._metric_grid.setContentsMargins(0, 0, 0, 0)
        self._metric_grid.setSpacing(4)
        splitter.addWidget(self._metric_container)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter, stretch=1)

        # Connect checkboxes to auto-redraw
        self._norm_cb.stateChanged.connect(self._draw)
        self._merge_cb.stateChanged.connect(self._draw)
        for cb in self._ev_checks.values():
            cb.stateChanged.connect(self._on_ev_changed)

    def _make_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {FG_COLOR}; font-weight: bold; font-size: 9pt;")
        return lbl

    def _make_ev_checkbox(self, key: str) -> QCheckBox:
        label = self._data.get_event_label(key)
        color = self._data.get_event_color(key)
        cb = QCheckBox(label)
        cb.setChecked(False)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {color}; font-weight: bold; font-size: 9pt; }}"
            f"QCheckBox::indicator:checked   {{ background: {color}; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #1a1a3e; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
        )
        return cb

    def _on_ev_changed(self):
        # Prevent unchecking the last event
        checked = [k for k, cb in self._ev_checks.items() if cb.isChecked()]
        if not checked:
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
        self._draw()

    # ── PDF export ────────────────────────────────────────────────────────────
    def _save_pdf(self):
        unit_indices = self._get_selection()
        suffix  = '_'.join(f"U{i}" for i in unit_indices) if unit_indices else 'compare'
        default = f"compare_{suffix}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", default, "PDF files (*.pdf)")
        if not path:
            return
        _render_widget_to_pdf(self._splitter, path, self)

    # ── Drawing ───────────────────────────────────────────────────────────────
    def _draw(self):
        _clear_grid(self._psth_grid)
        _clear_grid(self._metric_grid)

        unit_indices = self._get_selection()
        if not unit_indices:
            self._legend_lbl.setText("No units selected — Ctrl+click rows in the unit table.")
            return

        event_keys = [k for k, cb in self._ev_checks.items() if cb.isChecked()]
        if not event_keys:
            return

        # Read current settings from main window at draw time
        xlim       = self._get_xlim()
        smooth     = self._get_smooth()
        block_size = self._get_block_size()

        normalize = self._norm_cb.isChecked()
        merge     = self._merge_cb.isChecked()
        y_lbl     = 'Norm FR' if normalize else 'FR (Hz)'
        n_ev      = len(event_keys)
        cols      = min(n_ev, 2)

        # ── Legend ────────────────────────────────────────────────────────────
        parts = []
        for ci, uid in enumerate(unit_indices):
            color = CELL_COLORS[ci % len(CELL_COLORS)]
            lbl   = f"U{uid} {self._data.cell_type_label[uid]}"
            parts.append(f'<span style="color:{color}">&#9632; {lbl}</span>')
        if merge:
            parts.append(f'<span style="color:#FFD54F">&#9632; Merged average</span>')
        self._legend_lbl.setText('&nbsp;&nbsp;'.join(parts))

        # ── Mean PSTH plots (create before inserting into layout) ─────────────
        psth_pws = []
        for key in event_keys:
            title = self._data.get_event_label(key)
            pw = _styled_pw(title=title, x_label='Time (s)',
                            y_label=y_lbl, xlim=xlim)
            leg = pw.addLegend(offset=(-10, 10))
            leg.setLabelTextColor(FG_COLOR)
            pw.addItem(pg.InfiniteLine(pos=0, angle=90,
                                       pen=pg.mkPen(ZERO_COLOR, width=1,
                                                    style=Qt.DashLine)))
            psth_pws.append(pw)
        for j, pw in enumerate(psth_pws):
            self._psth_grid.addWidget(pw, j // cols, j % cols)

        # ── Metric plots (5 mini-plots, same row) ─────────────────────────────
        fr_unit = 'norm' if normalize else 'Hz'
        metric_labels_cur = [
            f'Max ({fr_unit})', f'Min ({fr_unit})', f'Range ({fr_unit})',
            't Max (s)', 't Min (s)',
        ]
        metric_pws = []
        for lbl in metric_labels_cur:
            pw = _styled_pw(title=lbl, x_label='Block #',
                            y_label='', small=True)
            metric_pws.append(pw)
        for j, pw in enumerate(metric_pws):
            self._metric_grid.addWidget(pw, 0, j)

        # ── Draw data per cell ────────────────────────────────────────────────
        # Collect per-cell mean PSTHs for potential merging
        # psth_data[ji][ci] = (mean, sem, tax)
        psth_data = {ji: [] for ji in range(len(event_keys))}

        for ci, unit_idx in enumerate(unit_indices):
            color   = CELL_COLORS[ci % len(CELL_COLORS)]
            r, g, b = _hex_to_rgb(color)
            pen     = pg.mkPen(color=(r, g, b), width=2)
            brush   = pg.mkBrush(r, g, b)
            cell_lbl = f"U{unit_idx}"

            for ji, key in enumerate(event_keys):
                mean, sem, tax = self._data.get_mean_psth(unit_idx, key, smooth)

                if normalize:
                    mn, mx = mean.min(), mean.max()
                    rng = mx - mn
                    if rng > 1e-9:
                        mean = (mean - mn) / rng
                        sem  = sem / rng

                psth_data[ji].append((mean, sem, tax))

                if not merge:
                    if sem is not None and np.any(sem > 0):
                        upper = pg.PlotDataItem(tax, mean + sem)
                        lower = pg.PlotDataItem(tax, mean - sem)
                        psth_pws[ji].addItem(
                            pg.FillBetweenItem(upper, lower,
                                               brush=pg.mkBrush(r, g, b, 40)))
                    psth_pws[ji].plot(
                        tax, mean, pen=pen,
                        name=cell_lbl if ji == 0 else None
                    )

            if not merge:
                # Individual block metrics
                try:
                    metrics = self._data.compute_block_metrics(
                        unit_idx, event_keys[0],
                        block_size, smooth,
                        xlim=xlim)
                    if normalize:
                        metrics = dict(metrics)   # shallow copy
                        for k in ('max', 'min', 'range'):
                            v = metrics[k]
                            v_max = np.abs(v).max()
                            if v_max > 1e-9:
                                metrics[k] = v / v_max
                    bn = metrics['block_nums']
                    for mi, key in enumerate(METRIC_KEYS):
                        metric_pws[mi].plot(
                            bn, metrics[key],
                            pen=pen,
                            symbol='o', symbolSize=5,
                            symbolBrush=brush,
                            symbolPen=pg.mkPen(None),
                            name=cell_lbl if mi == 0 else None
                        )
                except Exception:
                    pass

        # ── Merged mode: overlay averaged PSTH + merged block metrics ─────────
        if merge and unit_indices:
            merge_pen   = pg.mkPen(color='#FFD54F', width=2.5)
            merge_brush = pg.mkBrush(255, 213, 79)

            for ji, key in enumerate(event_keys):
                cells = psth_data[ji]
                if not cells:
                    continue
                tax       = cells[0][2]
                mean_mat  = np.stack([c[0] for c in cells], axis=0)   # (n_cells, n_time)
                merged    = mean_mat.mean(axis=0)
                psth_pws[ji].plot(
                    tax, merged,
                    pen=merge_pen,
                    name='Merged' if ji == 0 else None
                )

            # Merged block metrics (first event, xlim-restricted)
            try:
                first_key = event_keys[0]
                block_mats = []
                ref_tax    = None
                ref_bn     = None
                for unit_idx in unit_indices:
                    bm, tax_b = self._data.compute_block_psth(
                        unit_idx, first_key, block_size, smooth)
                    if ref_tax is None:
                        ref_tax = tax_b
                        ref_bn  = np.arange(1, bm.shape[1] + 1)
                    if normalize:
                        # normalise each block column to [0,1] using its own range
                        col_min = bm.min(axis=0, keepdims=True)
                        col_max = bm.max(axis=0, keepdims=True)
                        rng = col_max - col_min
                        rng[rng < 1e-9] = 1.0
                        bm = (bm - col_min) / rng
                    # Align to shortest block count
                    if block_mats and bm.shape[1] != block_mats[0].shape[1]:
                        n = min(bm.shape[1], block_mats[0].shape[1])
                        bm = bm[:, :n]
                        block_mats = [b[:, :n] for b in block_mats]
                        ref_bn = ref_bn[:n]
                    block_mats.append(bm)

                merged_bm = np.stack(block_mats, axis=0).mean(axis=0)  # (n_time, n_blocks)

                # Apply xlim restriction
                if xlim is not None and ref_tax is not None:
                    t_lo, t_hi = xlim
                    mask = (ref_tax >= t_lo) & (ref_tax <= t_hi)
                    if mask.any():
                        merged_bm = merged_bm[mask, :]

                idx_max = merged_bm.argmax(axis=0)
                idx_min = merged_bm.argmin(axis=0)
                tax_w   = ref_tax[(ref_tax >= xlim[0]) & (ref_tax <= xlim[1])] if xlim is not None else ref_tax
                m_metrics = {
                    'max':   merged_bm.max(axis=0),
                    'min':   merged_bm.min(axis=0),
                    'range': merged_bm.max(axis=0) - merged_bm.min(axis=0),
                    't_max': tax_w[idx_max] if len(tax_w) > 0 else np.zeros_like(ref_bn, dtype=float),
                    't_min': tax_w[idx_min] if len(tax_w) > 0 else np.zeros_like(ref_bn, dtype=float),
                }
                for mi, key in enumerate(METRIC_KEYS):
                    metric_pws[mi].plot(
                        ref_bn, m_metrics[key],
                        pen=merge_pen,
                        symbol='o', symbolSize=5,
                        symbolBrush=merge_brush,
                        symbolPen=pg.mkPen(None),
                        name='Merged' if mi == 0 else None
                    )
            except Exception:
                pass
