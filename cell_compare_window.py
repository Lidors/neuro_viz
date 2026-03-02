"""
cell_compare_window.py – Single-cell cross-session comparison.

Each imported cell carries its own SessionData so events are always
resolved correctly.  All PSTHs and both metric traces land on shared
axes so the user can directly compare response shapes and dynamics
across sessions.

Usage
-----
  win = CellCompareWindow()
  win.add_cell(CellEntry(label="U12 sess-A", color="#2196F3",
                         unit_indices=[11], session_data=sd, session_name="A"))
  win.show()

Grouping
--------
  Select 2+ cells in the sidebar list and press G (or click "Group").
  The group is drawn as a mean ± SEM across units.
  Select a group and click "Ungroup" to restore individual cells.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QFont, QKeySequence
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QCheckBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QAbstractItemView, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QLabel, QSplitter, QColorDialog,
    QInputDialog, QMessageBox, QShortcut,
)

from population_viewer import (
    BG_COLOR, FG_COLOR, ZERO_COLOR, GROUP_COLORS,
    _hex_to_rgb, _lbl, _btn, _COMBO_STYLE,
)

_SB_STYLE = (
    f"QSpinBox, QDoubleSpinBox {{ background: #0d1b2a; color: {FG_COLOR}; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 1px 4px; }}"
)
# Line styles cycle per-event so different conditions are visually distinct
_LINE_STYLES = [Qt.SolidLine, Qt.DashLine, Qt.DotLine, Qt.DashDotLine]


@dataclass
class CellEntry:
    label:        str           # display name, e.g. "U12 sess-A"
    color:        str           # hex color, e.g. "#2196F3"
    unit_indices: list          # list of unit indices into session_data
    session_data: object        # SessionData instance
    session_name: str           # human-readable session identifier
    hidden:       bool  = field(default=False)
    is_group:     bool  = field(default=False)
    members:      list  = field(default_factory=list)  # original CellEntry list (for ungroup)


def _make_plot(title: str = '') -> pg.PlotWidget:
    """Create a dark-themed PlotWidget."""
    w = pg.PlotWidget(title=title)
    w.setBackground(BG_COLOR)
    w.showGrid(x=True, y=True, alpha=0.15)
    for ax in ('bottom', 'left'):
        w.getAxis(ax).setPen(FG_COLOR)
        w.getAxis(ax).setTextPen(FG_COLOR)
        w.getAxis(ax).setStyle(tickFont=QFont('Arial', 8))
    if title:
        w.setTitle(title, color=FG_COLOR, size='9pt')
    return w


class CellCompareWindow(QDialog):
    """Compare individual units across sessions on shared axes.

    Layout
    ------
      Controls bar  (Smooth / Win / Block / Normalize)
      Event area    (one checkbox row per session, coloured by event)
      ─────────────────────────────────────────────────────────────
      │ Cell list   │  PSTH  (all cells × all selected events)    │
      │ (sidebar)   ├────────────────────────────────────────────│
      │ [Hide][Clr] │  Amplitude  │  Best Lag                     │
      │ [Grp] [Ugrp]└────────────────────────────────────────────┘
      │ [Remove]
      └─────────────
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cell Comparison")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1200, 800)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._cells:    list[CellEntry] = []
        self._sessions: list[tuple[str, object]] = []  # (name, SessionData), unique
        self._ev_checks: dict[int, dict[str, QCheckBox]] = {}

        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_cell(self, entry: CellEntry):
        """Add one cell.  Rebuilds event rows if a new session appears."""
        si = next((i for i, (_, d) in enumerate(self._sessions)
                   if d is entry.session_data), -1)
        new_session = (si < 0)
        if new_session:
            self._sessions.append((entry.session_name, entry.session_data))

        self._cells.append(entry)

        if new_session:
            self._rebuild_event_checks()

        self._refresh_cell_list()
        self._refresh()
        self.raise_()
        self.activateWindow()

    # ── UI building ───────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Controls bar ──────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(_lbl("Smooth:"))
        self._smooth_sb = QSpinBox()
        self._smooth_sb.setRange(0, 30)
        self._smooth_sb.setValue(3)
        self._smooth_sb.setFixedSize(54, 24)
        self._smooth_sb.setStyleSheet(_SB_STYLE)
        self._smooth_sb.valueChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._smooth_sb)
        bar.addSpacing(4)

        bar.addWidget(_lbl("Win (s):"))
        self._win_sb = QDoubleSpinBox()
        self._win_sb.setRange(0.5, 10.0)
        self._win_sb.setValue(3.0)
        self._win_sb.setSingleStep(0.5)
        self._win_sb.setFixedSize(60, 24)
        self._win_sb.setStyleSheet(_SB_STYLE)
        self._win_sb.valueChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._win_sb)
        bar.addSpacing(4)

        bar.addWidget(_lbl("Block:"))
        self._blk_sb = QSpinBox()
        self._blk_sb.setRange(1, 200)
        self._blk_sb.setValue(10)
        self._blk_sb.setFixedSize(54, 24)
        self._blk_sb.setStyleSheet(_SB_STYLE)
        self._blk_sb.valueChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._blk_sb)
        bar.addSpacing(8)

        bar.addWidget(_lbl("Normalize:"))
        self._norm_combo = QComboBox()
        self._norm_combo.setFixedHeight(24)
        self._norm_combo.setStyleSheet(_COMBO_STYLE)
        for lbl_txt, val in [("None", 'none'), ("Z-score", 'zscore'),
                              ("Peak", 'peak'), ("Max", 'max')]:
            self._norm_combo.addItem(lbl_txt, userData=val)
        self._norm_combo.currentIndexChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._norm_combo)

        bar.addStretch(1)

        scatter_btn = _btn("Scatter ↗")
        scatter_btn.setToolTip(
            "Open scatter window: compare amplitude or lag between two signals")
        scatter_btn.clicked.connect(self._open_scatter)
        bar.addWidget(scatter_btn)

        clear_btn = _btn("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        bar.addWidget(clear_btn)

        refresh_btn = _btn("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        bar.addWidget(refresh_btn)
        outer.addLayout(bar)

        # ── Event area (per-session rows) ─────────────────────────────────────
        self._ev_area_widget = QWidget()
        self._ev_area_vbox   = QVBoxLayout(self._ev_area_widget)
        self._ev_area_vbox.setContentsMargins(4, 2, 4, 2)
        self._ev_area_vbox.setSpacing(2)
        self._ev_scroll = QScrollArea()
        self._ev_scroll.setWidget(self._ev_area_widget)
        self._ev_scroll.setWidgetResizable(True)
        self._ev_scroll.setFixedHeight(32)
        self._ev_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._ev_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._ev_scroll.setFrameShape(QFrame.NoFrame)
        self._ev_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        outer.addWidget(self._ev_scroll)

        # ── Main area ─────────────────────────────────────────────────────────
        main = QHBoxLayout()
        main.setSpacing(0)

        # Sidebar: cell list + action buttons
        side = QWidget()
        side.setFixedWidth(210)
        side.setStyleSheet(f"background: {BG_COLOR};")
        side_l = QVBoxLayout(side)
        side_l.setContentsMargins(6, 6, 6, 6)
        side_l.setSpacing(4)
        side_l.addWidget(_lbl("Imported cells  (Ctrl+click to multi-select)"))

        self._cell_list = QListWidget()
        self._cell_list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; border: 1px solid #3a3a6a;")
        self._cell_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._cell_list.currentRowChanged.connect(lambda _: self._update_sidebar_btns())
        self._cell_list.itemSelectionChanged.connect(self._update_sidebar_btns)
        side_l.addWidget(self._cell_list, stretch=1)

        # Row 1: Hide/Show + Color
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self._hide_btn = _btn("Hide")
        self._hide_btn.clicked.connect(self._toggle_visibility)
        row1.addWidget(self._hide_btn)
        color_btn = _btn("Color…")
        color_btn.clicked.connect(self._change_color)
        row1.addWidget(color_btn)
        side_l.addLayout(row1)

        # Row 2: Group + Ungroup
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        self._group_btn = _btn("Group (G)")
        self._group_btn.clicked.connect(self._group_selected)
        row2.addWidget(self._group_btn)
        self._ungroup_btn = _btn("Ungroup")
        self._ungroup_btn.clicked.connect(self._ungroup_selected)
        row2.addWidget(self._ungroup_btn)
        side_l.addLayout(row2)

        # Row 3: Remove
        rm_btn = _btn("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        side_l.addWidget(rm_btn)

        # Row 4: Export to Console
        console_btn = _btn("Export to Console")
        console_btn.setToolTip("Export selected cell data to the Python console")
        console_btn.clicked.connect(self._export_to_console)
        side_l.addWidget(console_btn)

        main.addWidget(side)

        # G shortcut (works even when list has focus)
        grp_sc = QShortcut(QKeySequence('G'), self)
        grp_sc.activated.connect(self._group_selected)

        # Plot area: PSTH on top, amp + lag on bottom
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")

        self._psth_plot = _make_plot("PSTH")
        self._psth_plot.setLabel('bottom', 'Time (s)', color=FG_COLOR)
        self._psth_plot.setLabel('left', 'FR (Hz)', color=FG_COLOR)
        self._psth_plot.addLine(
            x=0, pen=pg.mkPen(ZERO_COLOR, width=1, style=Qt.DashLine))
        splitter.addWidget(self._psth_plot)

        metrics_w = QWidget()
        metrics_l = QHBoxLayout(metrics_w)
        metrics_l.setContentsMargins(0, 0, 0, 0)
        metrics_l.setSpacing(4)
        self._amp_plot = _make_plot("Amplitude")
        self._lag_plot = _make_plot("Best Lag (ms)")
        self._amp_plot.setLabel('bottom', 'Block #', color=FG_COLOR)
        self._lag_plot.setLabel('bottom', 'Block #', color=FG_COLOR)
        metrics_l.addWidget(self._amp_plot)
        metrics_l.addWidget(self._lag_plot)
        splitter.addWidget(metrics_w)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter, stretch=1)
        outer.addLayout(main, stretch=1)

    # ── Event checkboxes ──────────────────────────────────────────────────────

    def _rebuild_event_checks(self):
        """Create one labelled checkbox row per session."""
        while self._ev_area_vbox.count():
            item = self._ev_area_vbox.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._ev_checks.clear()

        if not self._sessions:
            return

        multi = len(self._sessions) > 1
        for si, (sname, sess_data) in enumerate(self._sessions):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(8)

            if multi:
                lbl = QLabel(f"[{sname}]")
                lbl.setStyleSheet(
                    "color: #8888bb; font-size: 8pt; font-style: italic;")
                lbl.setFixedWidth(110)
                row_l.addWidget(lbl)

            self._ev_checks[si] = {}
            for key in sess_data.event_keys:
                cb = self._make_ev_cb(key, sess_data)
                cb.stateChanged.connect(lambda _: self._refresh())
                row_l.addWidget(cb)
                self._ev_checks[si][key] = cb
            row_l.addStretch()
            self._ev_area_vbox.addWidget(row_w)

        # Default: first event of every session checked
        for si in self._ev_checks:
            if self._ev_checks[si]:
                next(iter(self._ev_checks[si].values())).setChecked(True)

        h = max(28, min(len(self._sessions) * 26 + 4, 104))
        self._ev_scroll.setFixedHeight(h)

    def _make_ev_cb(self, key: str, data) -> QCheckBox:
        label = data.get_event_label(key)
        color = data.get_event_color(key)
        cb = QCheckBox(label)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {color}; font-size: 9pt; }}"
            f"QCheckBox::indicator:checked   {{ background: {color}; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #1a1a3e; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
        )
        return cb

    # ── Cell list ─────────────────────────────────────────────────────────────

    def _refresh_cell_list(self):
        prev_row = self._cell_list.currentRow()
        self._cell_list.clear()
        multi = len(self._sessions) > 1
        for cell in self._cells:
            suffix = f"  [{cell.session_name}]" if multi else ''
            if cell.is_group:
                text = f"◆ {cell.label}  (n={len(cell.unit_indices)}){suffix}"
            else:
                text = f"● {cell.label}{suffix}"
            item = QListWidgetItem(text)
            if cell.hidden:
                item.setForeground(QColor(70, 70, 90))
                font = item.font()
                font.setStrikeOut(True)
                item.setFont(font)
            else:
                r, g, b = _hex_to_rgb(cell.color)
                item.setForeground(QColor(r, g, b))
            self._cell_list.addItem(item)
        if 0 <= prev_row < self._cell_list.count():
            self._cell_list.setCurrentRow(prev_row)
        self._update_sidebar_btns()

    def _update_sidebar_btns(self):
        """Sync Hide/Show and Group/Ungroup labels with the current selection."""
        selected = self._selected_rows()
        n = len(selected)

        # Hide/Show: label reflects the focused cell's state
        row = self._cell_list.currentRow()
        if 0 <= row < len(self._cells) and self._cells[row].hidden:
            self._hide_btn.setText("Show")
        else:
            self._hide_btn.setText("Hide")

        # Group: needs 2+ selected, all from same session, none already a group
        can_group = (
            n >= 2
            and len({id(self._cells[r].session_data) for r in selected}) == 1
        )
        self._group_btn.setEnabled(can_group)

        # Ungroup: needs exactly 1 selected group
        can_ungroup = (
            n == 1
            and 0 <= selected[0] < len(self._cells)
            and self._cells[selected[0]].is_group
            and bool(self._cells[selected[0]].members)
        )
        self._ungroup_btn.setEnabled(can_ungroup)

    def _selected_rows(self) -> list[int]:
        """Return sorted list of currently selected row indices."""
        return sorted(
            self._cell_list.row(item)
            for item in self._cell_list.selectedItems()
        )

    # ── Sidebar actions ───────────────────────────────────────────────────────

    def _remove_selected(self):
        for row in reversed(self._selected_rows()):
            self._cells.pop(row)
        self._refresh_cell_list()
        self._refresh()

    def _toggle_visibility(self):
        row = self._cell_list.currentRow()
        if row < 0:
            return
        cell = self._cells[row]
        cell.hidden = not cell.hidden
        self._refresh_cell_list()
        self._refresh()

    def _change_color(self):
        row = self._cell_list.currentRow()
        if row < 0:
            return
        cell = self._cells[row]
        r, g, b = _hex_to_rgb(cell.color)
        new_color = QColorDialog.getColor(QColor(r, g, b), self, "Cell color")
        if not new_color.isValid():
            return
        cell.color = (f"#{new_color.red():02x}"
                      f"{new_color.green():02x}"
                      f"{new_color.blue():02x}")
        self._refresh_cell_list()
        self._refresh()

    def _group_selected(self):
        rows = self._selected_rows()
        if len(rows) < 2:
            return
        cells = [self._cells[r] for r in rows]

        # All cells must be from the same session
        if len({id(c.session_data) for c in cells}) > 1:
            QMessageBox.warning(
                self, "Cannot group",
                "All selected cells must be from the same session.\n"
                "Use normalization (Z-score / Peak) to compare across sessions.")
            return

        # Ask for a label
        default_label = f"Group ({len(cells)} cells)"
        label, ok = QInputDialog.getText(
            self, "Group cells", "Group label:", text=default_label)
        if not ok or not label.strip():
            return

        # Flatten unit indices (handles grouping a group)
        unit_indices = []
        for c in cells:
            unit_indices.extend(c.unit_indices)

        group = CellEntry(
            label        = label.strip(),
            color        = cells[0].color,
            unit_indices = unit_indices,
            session_data = cells[0].session_data,
            session_name = cells[0].session_name,
            is_group     = True,
            members      = list(cells),
        )

        # Replace the selected cells with the group at the first selected position
        first_row = rows[0]
        for r in reversed(rows):
            self._cells.pop(r)
        self._cells.insert(first_row, group)

        self._refresh_cell_list()
        self._cell_list.setCurrentRow(first_row)
        self._refresh()

    def _ungroup_selected(self):
        rows = self._selected_rows()
        if len(rows) != 1:
            return
        row  = rows[0]
        cell = self._cells[row]
        if not cell.is_group or not cell.members:
            return

        self._cells.pop(row)
        for i, member in enumerate(cell.members):
            self._cells.insert(row + i, member)

        self._refresh_cell_list()
        self._refresh()

    def _clear_all(self):
        self._cells.clear()
        self._sessions.clear()
        self._rebuild_event_checks()
        self._refresh_cell_list()
        self._psth_plot.clear()
        self._psth_plot.addLine(
            x=0, pen=pg.mkPen(ZERO_COLOR, width=1, style=Qt.DashLine))
        self._amp_plot.clear()
        self._lag_plot.clear()

    def _open_scatter(self):
        """Open a Metric Scatter window using the currently visible signals."""
        win = MetricScatterWindow(self)
        win.show()

    def _export_to_console(self):
        """Export selected (or all) cells' data to the Python console."""
        from PyQt5.QtWidgets import QApplication as _QA
        from app_window import MainWindow
        main_win = next(
            (w for w in _QA.topLevelWidgets() if isinstance(w, MainWindow)),
            None)
        if main_win is None:
            return
        main_win._open_console()
        rows = self._selected_rows()
        if not rows:
            rows = list(range(len(self._cells)))
        smooth  = self._smooth_sb.value()
        win_sec = self._win_sb.value()
        for row in rows:
            cell = self._cells[row]
            safe = cell.label.replace(' ', '_').replace('[', '').replace(']', '')
            data_dict = {}
            for key in cell.session_data.event_keys:
                try:
                    mean, sem, tax = self._get_psth(
                        cell, key, smooth, win_sec, 'none')
                    data_dict[key] = {'mean': mean, 'sem': sem, 'time': tax}
                except Exception:
                    pass
            main_win._console_win.inject(safe, data_dict)

    # ── Data helpers ──────────────────────────────────────────────────────────

    def _session_idx_of(self, cell: CellEntry) -> int:
        return next((i for i, (_, d) in enumerate(self._sessions)
                     if d is cell.session_data), -1)

    def _get_cell_psth(self, cell: CellEntry, key: str,
                       smooth: int, window_sec: float,
                       normalize: str) -> tuple:
        """Trial-level SEM for single cells (calls get_mean_psth directly)."""
        mean, sem, tax = cell.session_data.get_mean_psth(
            cell.unit_indices[0], key, smooth, window_sec)

        if normalize == 'zscore':
            sigma = float(mean.std())
            if sigma > 0:
                mean = (mean - float(mean.mean())) / sigma
                sem  = sem / sigma
        elif normalize == 'peak':
            shifted = mean - float(mean.min())
            mx = float(shifted.max())
            if mx > 0:
                mean = shifted / mx
                sem  = sem / mx
        elif normalize == 'max':
            # Divide by peak absolute value — preserves mean/DC offset
            mx = float(np.abs(mean).max())
            if mx > 0:
                mean = mean / mx
                sem  = sem / mx

        return mean, sem, tax

    def _get_psth(self, cell: CellEntry, key: str,
                  smooth: int, window_sec: float,
                  normalize: str) -> tuple:
        """Dispatch to the right PSTH computation based on entry type."""
        if cell.is_group:
            # 'max' is not handled by the backend — fetch unnormalised then apply
            backend_norm = normalize if normalize != 'max' else 'none'
            mean, sem, tax = cell.session_data.get_group_mean_psth(
                cell.unit_indices, key, smooth, window_sec, backend_norm)
            if normalize == 'max':
                mx = float(np.abs(mean).max())
                if mx > 0:
                    mean = mean / mx
                    sem  = sem  / mx
            return mean, sem, tax
        else:
            # Trial-level SEM (meaningful for single cells)
            return self._get_cell_psth(cell, key, smooth, window_sec, normalize)

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._cells:
            return

        smooth     = self._smooth_sb.value()
        window_sec = self._win_sb.value()
        block_size = self._blk_sb.value()
        normalize  = self._norm_combo.currentData()
        y_label    = 'FR (norm)' if normalize != 'none' else 'FR (Hz)'

        _amp_anns: list[str] = []
        _lag_anns: list[str] = []

        # Clear all plots and re-add decorations
        self._psth_plot.clear()
        self._psth_plot.setLabel('left', y_label, color=FG_COLOR)
        self._psth_plot.addLine(
            x=0, pen=pg.mkPen(ZERO_COLOR, width=1, style=Qt.DashLine))
        self._amp_plot.clear()
        self._lag_plot.clear()

        # Legends are recreated fresh each refresh
        self._psth_plot.addLegend(
            offset=(-10, 10), labelTextColor=FG_COLOR,
            brush=pg.mkBrush(BG_COLOR))
        self._amp_plot.addLegend(
            offset=(-10, 10), labelTextColor=FG_COLOR,
            brush=pg.mkBrush(BG_COLOR))
        self._lag_plot.addLegend(
            offset=(-10, 10), labelTextColor=FG_COLOR,
            brush=pg.mkBrush(BG_COLOR))

        for cell in self._cells:
            if cell.hidden:
                continue
            si = self._session_idx_of(cell)
            if si < 0:
                continue
            r, g, b = _hex_to_rgb(cell.color)
            ev_items = list(self._ev_checks.get(si, {}).items())

            for ev_idx, (key, cb) in enumerate(ev_items):
                if not cb.isChecked():
                    continue

                ev_label   = self._sessions[si][1].get_event_label(key)
                curve_name = f"{cell.label}  [{ev_label}]"
                style      = _LINE_STYLES[ev_idx % len(_LINE_STYLES)]
                pen        = pg.mkPen(color=(r, g, b), width=2, style=style)

                # ── PSTH ──────────────────────────────────────────────────
                try:
                    mean, sem, tax = self._get_psth(
                        cell, key, smooth, window_sec, normalize)
                except Exception:
                    continue

                upper = self._psth_plot.plot(tax, mean + sem,
                                             pen=pg.mkPen(None))
                lower = self._psth_plot.plot(tax, mean - sem,
                                             pen=pg.mkPen(None))
                self._psth_plot.addItem(
                    pg.FillBetweenItem(upper, lower,
                                       brush=pg.mkBrush(r, g, b, 45)))
                self._psth_plot.plot(tax, mean, pen=pen, name=curve_name)

                # ── Block metrics ─────────────────────────────────────────
                try:
                    m = cell.session_data.compute_group_block_metrics(
                        cell.unit_indices, key,
                        block_size, smooth, window_sec, None, 'percell')
                except Exception:
                    continue
                if not m:
                    continue

                bn    = m['block_nums']
                brush = pg.mkBrush(r, g, b)
                for plot, mk, sk, ann_list in [
                    (self._amp_plot, 'amplitude_mean',   'amplitude_sem',   _amp_anns),
                    (self._lag_plot, 'best_lag_ms_mean', 'best_lag_ms_sem', _lag_anns),
                ]:
                    mean_m = m[mk]
                    sem_m  = m[sk]
                    plot.plot(bn, mean_m, pen=pen, symbol='o', symbolSize=5,
                              symbolBrush=brush, symbolPen=pg.mkPen(None),
                              name=curve_name)
                    plot.addItem(pg.ErrorBarItem(
                        x=bn, y=mean_m, top=sem_m, bottom=sem_m,
                        pen=pg.mkPen(color=(r, g, b), width=1)))
                    # Pearson r annotation
                    try:
                        from scipy.stats import pearsonr as _pr
                        bn_f = np.asarray(bn, float)
                        v_f  = np.asarray(mean_m, float)
                        mask = np.isfinite(bn_f) & np.isfinite(v_f)
                        if mask.sum() >= 3:
                            rval, pval = _pr(bn_f[mask], v_f[mask])
                            p_str = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'
                            ann_list.append(f'r={rval:.2f} {p_str}')
                    except Exception:
                        pass

        # Update metric plot titles with Pearson annotations
        for plot, base_title, anns in [
            (self._amp_plot, 'Amplitude',     _amp_anns),
            (self._lag_plot, 'Best Lag (ms)', _lag_anns),
        ]:
            if anns:
                plot.setTitle(base_title + '  |  ' + '  |  '.join(anns),
                              color=FG_COLOR, size='9pt')
            else:
                plot.setTitle(base_title, color=FG_COLOR, size='9pt')

        self._update_sidebar_btns()


# ── Metric Scatter Window ─────────────────────────────────────────────────────

class MetricScatterWindow(QDialog):
    """Scatter plot: amplitude or lag of signal A vs signal B (per block).

    Each point = one trial block, coloured by temporal order (plasma ramp).
    Pearson r and p-value are shown below the plot.
    The two signals and the metric are chosen via drop-downs.
    """

    def __init__(self, parent_win: CellCompareWindow):
        super().__init__(parent_win)
        self._pw = parent_win          # reference to the parent CellCompareWindow
        self._items: list[tuple] = []  # (label, CellEntry, event_key)
        self.setWindowTitle("Metric Scatter")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(620, 580)
        self.setStyleSheet(f"background: {BG_COLOR};")
        self._build_ui()
        self._update_combos()
        self._replot()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(6)

        ctrl.addWidget(_lbl("X:"))
        self._x_combo = QComboBox()
        self._x_combo.setFixedHeight(24)
        self._x_combo.setStyleSheet(_COMBO_STYLE)
        self._x_combo.setMinimumWidth(180)
        self._x_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._x_combo)

        ctrl.addSpacing(6)
        ctrl.addWidget(_lbl("Y:"))
        self._y_combo = QComboBox()
        self._y_combo.setFixedHeight(24)
        self._y_combo.setStyleSheet(_COMBO_STYLE)
        self._y_combo.setMinimumWidth(180)
        self._y_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._y_combo)

        ctrl.addSpacing(6)
        ctrl.addWidget(_lbl("Metric:"))
        self._metric_combo = QComboBox()
        self._metric_combo.setFixedHeight(24)
        self._metric_combo.setStyleSheet(_COMBO_STYLE)
        self._metric_combo.addItem("Amplitude",     'amplitude')
        self._metric_combo.addItem("Best Lag (ms)", 'best_lag_ms')
        self._metric_combo.currentIndexChanged.connect(self._replot)
        ctrl.addWidget(self._metric_combo)

        ctrl.addStretch()
        refresh_btn = _btn("Refresh signals")
        refresh_btn.setToolTip("Re-read signals from the parent Compare window")
        refresh_btn.clicked.connect(self._update_combos_and_replot)
        ctrl.addWidget(refresh_btn)
        outer.addLayout(ctrl)

        # Plot
        self._pw_plot = pg.PlotWidget()
        self._pw_plot.setBackground(BG_COLOR)
        self._pw_plot.showGrid(x=True, y=True, alpha=0.15)
        for ax in ('bottom', 'left'):
            self._pw_plot.getAxis(ax).setPen(FG_COLOR)
            self._pw_plot.getAxis(ax).setTextPen(FG_COLOR)
            self._pw_plot.getAxis(ax).setStyle(tickFont=QFont('Arial', 8))
        outer.addWidget(self._pw_plot, stretch=1)

        # Annotation label (r, p, n)
        self._ann_lbl = QLabel()
        self._ann_lbl.setAlignment(Qt.AlignCenter)
        self._ann_lbl.setStyleSheet(
            f"color: {FG_COLOR}; font-size: 12pt; font-weight: bold;")
        outer.addWidget(self._ann_lbl)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _signal_items(self) -> list[tuple]:
        """All visible (cell, event) pairs from the parent window."""
        items = []
        pw = self._pw
        for cell in pw._cells:
            if cell.hidden:
                continue
            si = pw._session_idx_of(cell)
            if si < 0:
                continue
            for key, cb in pw._ev_checks.get(si, {}).items():
                if cb.isChecked():
                    ev_lbl = pw._sessions[si][1].get_event_label(key)
                    items.append((f"{cell.label}  [{ev_lbl}]", cell, key))
        return items

    def _update_combos(self):
        self._items = self._signal_items()
        for combo in (self._x_combo, self._y_combo):
            combo.blockSignals(True)
            combo.clear()
            for label, _, _ in self._items:
                combo.addItem(label)
            combo.blockSignals(False)
        # Default: X = first item, Y = last item (or same if only one)
        if len(self._items) >= 2:
            self._y_combo.setCurrentIndex(len(self._items) - 1)

    def _update_combos_and_replot(self):
        self._update_combos()
        self._replot()

    def _fetch_blocks(self, cell: CellEntry, key: str,
                      metric_key: str):
        """Return (block_nums, values) for the chosen metric, or (None, None)."""
        pw = self._pw
        try:
            m = cell.session_data.compute_group_block_metrics(
                cell.unit_indices, key,
                pw._blk_sb.value(),
                pw._smooth_sb.value(),
                pw._win_sb.value(),
                None, 'percell')
        except Exception:
            return None, None
        if not m:
            return None, None
        # group metrics use '_mean' suffix; single-cell dicts do not
        mk = metric_key + '_mean' if (metric_key + '_mean') in m else metric_key
        if mk not in m:
            return None, None
        return np.asarray(m['block_nums'], float), np.asarray(m[mk], float)

    # ── Plot ──────────────────────────────────────────────────────────────────

    def _replot(self):
        from scipy.stats import pearsonr

        self._pw_plot.clear()
        self._ann_lbl.setText('')

        if not self._items:
            self._ann_lbl.setText("No signals — click 'Refresh signals'.")
            return

        xi = self._x_combo.currentIndex()
        yi = self._y_combo.currentIndex()
        if xi < 0 or yi < 0 or xi >= len(self._items) or yi >= len(self._items):
            return

        x_lbl, x_cell, x_key = self._items[xi]
        y_lbl, y_cell, y_key = self._items[yi]
        metric_key = self._metric_combo.currentData()
        metric_lbl = self._metric_combo.currentText()

        bx, vx = self._fetch_blocks(x_cell, x_key, metric_key)
        by, vy = self._fetch_blocks(y_cell, y_key, metric_key)
        if bx is None or by is None:
            self._ann_lbl.setText("No data for one or both signals.")
            return

        # Match blocks by block number
        common = np.intersect1d(bx, by)
        if len(common) < 3:
            self._ann_lbl.setText(f"Only {len(common)} common blocks — need ≥ 3.")
            return

        ix = [int(np.where(bx == b)[0][0]) for b in common]
        iy = [int(np.where(by == b)[0][0]) for b in common]
        vx_m = vx[ix]
        vy_m = vy[iy]
        mask  = np.isfinite(vx_m) & np.isfinite(vy_m)
        if mask.sum() < 3:
            self._ann_lbl.setText("Too few finite data points.")
            return

        # Scatter: colour points by temporal order (plasma)
        import matplotlib.pyplot as _plt
        cmap = _plt.get_cmap('plasma')
        n = len(common)
        for i in range(n):
            if not mask[i]:
                continue
            c   = cmap(i / max(n - 1, 1))
            rgb = (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            self._pw_plot.plot(
                [float(vx_m[i])], [float(vy_m[i])],
                symbol='o', symbolSize=9,
                symbolBrush=pg.mkBrush(*rgb),
                symbolPen=pg.mkPen('w', width=0.5),
                pen=None,
            )

        # Regression line
        vx_v, vy_v = vx_m[mask], vy_m[mask]
        coeffs = np.polyfit(vx_v, vy_v, 1)
        x_line = np.linspace(vx_v.min(), vx_v.max(), 200)
        y_line = np.polyval(coeffs, x_line)
        self._pw_plot.plot(x_line, y_line,
                           pen=pg.mkPen('#ffffff', width=1.5,
                                        style=Qt.DashLine))

        # Identity line (dotted grey)
        lo = min(vx_v.min(), vy_v.min())
        hi = max(vx_v.max(), vy_v.max())
        self._pw_plot.plot([lo, hi], [lo, hi],
                           pen=pg.mkPen('#555577', width=1, style=Qt.DotLine))

        # Pearson r + p
        rval, pval = pearsonr(vx_v, vy_v)
        p_str = 'p < 0.001' if pval < 0.001 else f'p = {pval:.3f}'
        self._ann_lbl.setText(
            f"r = {rval:.3f}    {p_str}    (n = {int(mask.sum())} blocks)")

        # Axis labels
        def _short(s, n=50): return s[:n] + '…' if len(s) > n else s
        self._pw_plot.setLabel('bottom', f"{metric_lbl}  —  {_short(x_lbl)}",
                                color=FG_COLOR)
        self._pw_plot.setLabel('left',   f"{metric_lbl}  —  {_short(y_lbl)}",
                                color=FG_COLOR)
        self._pw_plot.setTitle(f"Metric Scatter: {metric_lbl}",
                                color=FG_COLOR, size='10pt')
