"""
compare_window.py – Cross-session comparison window.

Groups are imported from PopulationViewer ("Export to Compare →") or from
the main window's unit selection ("Export selection →").  Each CompareGroup
carries its own SessionData reference so no external state is needed.

Usage
-----
  win = CompareWindow()
  win.add_groups([CompareGroup(...), ...])   # called repeatedly as user exports
  win.show()
"""

from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget,
    QCheckBox, QScrollArea, QFrame, QListWidget, QListWidgetItem,
    QAbstractItemView, QComboBox, QSpinBox, QDoubleSpinBox,
    QPushButton, QLabel,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

# Reuse all shared widgets and helpers from population_viewer
from population_viewer import (
    CompareGroup, GroupSpec,
    _PSTHGroupView, _BlockGroupView, _EventMappingPanel,
    _compute_display_slots,
    BG_COLOR, FG_COLOR, GROUP_COLORS,
    _hex_to_rgb, _lbl, _btn, _COMBO_STYLE,
)

_SB_STYLE = (
    f"QSpinBox, QDoubleSpinBox {{ background: #0d1b2a; color: {FG_COLOR}; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 1px 4px; }}"
)


class CompareWindow(QDialog):
    """Persistent cross-session comparison window.

    Groups from different sessions are imported via ``add_groups()``.
    The window owns its own smooth / window / block-size controls so it is
    independent of the main window state.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cross-Session Comparison")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1300, 820)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._groups:   list[CompareGroup] = []
        self._sessions: list[tuple[str, object]] = []   # unique (name, SessionData)
        self._ev_checks: dict[int, dict[str, QCheckBox]] = {}

        self._build_ui()

    # ── Public API ────────────────────────────────────────────────────────────

    def add_groups(self, new_groups: list[CompareGroup]):
        """Import groups.  May add new sessions and rebuild the event bar."""
        new_sessions = False
        for g in new_groups:
            si = next((i for i, (_, d) in enumerate(self._sessions)
                       if d is g.session_data), -1)
            if si < 0:
                self._sessions.append((g.session_name, g.session_data))
                new_sessions = True
            self._groups.append(g)

        if new_sessions or not self._ev_checks:
            self._rebuild_event_checks()

        if len(self._sessions) > 1:
            self._map_panel.update_sessions(self._sessions)
            self._map_panel.setVisible(True)

        self._refresh_group_list()
        self._refresh()
        self.raise_()
        self.activateWindow()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Controls bar ──────────────────────────────────────────────────────
        bar = QHBoxLayout()
        bar.setSpacing(8)

        # Normalize
        bar.addWidget(_lbl("Normalize:"))
        self._norm_combo = QComboBox()
        self._norm_combo.setFixedHeight(24)
        self._norm_combo.setStyleSheet(_COMBO_STYLE)
        for label, val in [("None", 'none'), ("Z-score", 'zscore'), ("Peak", 'peak')]:
            self._norm_combo.addItem(label, userData=val)
        self._norm_combo.currentIndexChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._norm_combo)
        bar.addSpacing(8)

        # Smooth
        bar.addWidget(_lbl("Smooth:"))
        self._smooth_sb = QSpinBox()
        self._smooth_sb.setRange(0, 30)
        self._smooth_sb.setValue(0)
        self._smooth_sb.setFixedSize(54, 24)
        self._smooth_sb.setStyleSheet(_SB_STYLE)
        self._smooth_sb.valueChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._smooth_sb)
        bar.addSpacing(4)

        # Window (s)
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

        # Block size
        bar.addWidget(_lbl("Block:"))
        self._blk_sb = QSpinBox()
        self._blk_sb.setRange(1, 200)
        self._blk_sb.setValue(10)
        self._blk_sb.setFixedSize(54, 24)
        self._blk_sb.setStyleSheet(_SB_STYLE)
        self._blk_sb.valueChanged.connect(lambda _: self._refresh())
        bar.addWidget(self._blk_sb)
        bar.addSpacing(12)

        bar.addStretch(1)

        clear_btn = _btn("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        bar.addWidget(clear_btn)

        refresh_btn = _btn("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        bar.addWidget(refresh_btn)
        outer.addLayout(bar)

        # ── Event area (per-session rows, rebuilt dynamically) ─────────────────
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

        # ── Event mapping panel ───────────────────────────────────────────────
        self._map_panel = _EventMappingPanel()
        self._map_panel.setVisible(False)
        self._map_panel.mapping_changed.connect(self._refresh)
        outer.addWidget(self._map_panel)

        # ── Main area: group sidebar | plots ──────────────────────────────────
        main = QHBoxLayout()
        main.setSpacing(0)

        # Sidebar: group list + remove button
        side = QWidget()
        side.setFixedWidth(220)
        side.setStyleSheet(f"background: {BG_COLOR};")
        side_l = QVBoxLayout(side)
        side_l.setContentsMargins(6, 6, 6, 6)
        side_l.setSpacing(4)
        side_l.addWidget(_lbl("Imported groups"))
        self._group_list = QListWidget()
        self._group_list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; border: 1px solid #3a3a6a;")
        self._group_list.setSelectionMode(QAbstractItemView.SingleSelection)
        side_l.addWidget(self._group_list, stretch=1)
        rm_btn = _btn("Remove selected")
        rm_btn.clicked.connect(self._remove_selected)
        side_l.addWidget(rm_btn)
        main.addWidget(side)

        # Plot area: PSTH (top) + block metrics (bottom)
        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")
        self._psth_view  = _PSTHGroupView()
        self._block_view = _BlockGroupView()
        self._block_view._ev_combo.currentIndexChanged.connect(
            lambda _: self._block_view._draw())
        for btn in self._block_view._mode_group.buttons():
            btn.toggled.connect(lambda _: self._block_view._draw())
        for btn in self._block_view._xaxis_group.buttons():
            btn.toggled.connect(lambda _: self._block_view._draw())
        splitter.addWidget(self._psth_view)
        splitter.addWidget(self._block_view)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        main.addWidget(splitter, stretch=1)

        outer.addLayout(main, stretch=1)

    # ── Event checks ──────────────────────────────────────────────────────────

    def _rebuild_event_checks(self):
        """Rebuild per-session event checkbox rows."""
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

        # Default: check first event of session 0
        if 0 in self._ev_checks and self._ev_checks[0]:
            first_key = next(iter(self._ev_checks[0]))
            self._ev_checks[0][first_key].setChecked(True)

        # Adjust scroll-area height: ~26 px per session, max 104 px
        h = max(28, min(len(self._sessions) * 26 + 4, 104))
        self._ev_scroll.setFixedHeight(h)

    def _make_ev_cb(self, key: str, data) -> QCheckBox:
        label = data.get_event_label(key)
        color = data.get_event_color(key)
        cb    = QCheckBox(label)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {color}; font-size: 9pt; }}"
            f"QCheckBox::indicator:checked   {{ background: {color}; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background: #1a1a3e; "
            f"  border: 2px solid {color}; border-radius: 3px; }}"
        )
        return cb

    def _get_display_slots(self) -> list[dict[int, str]]:
        return _compute_display_slots(self._ev_checks, self._map_panel, self._sessions)

    # ── Group list ────────────────────────────────────────────────────────────

    def _refresh_group_list(self):
        self._group_list.clear()
        multi_sess = len(self._sessions) > 1
        for g in self._groups:
            suffix = f"  [{g.session_name}]" if multi_sess else ''
            item   = QListWidgetItem(
                f"● {g.label}  (n={len(g.unit_indices)}){suffix}")
            r, gv, b = _hex_to_rgb(g.color)
            item.setForeground(QColor(r, gv, b))
            self._group_list.addItem(item)

    def _remove_selected(self):
        row = self._group_list.currentRow()
        if row < 0:
            return
        self._groups.pop(row)
        self._refresh_group_list()
        self._refresh()

    def _clear_all(self):
        self._groups.clear()
        self._sessions.clear()
        self._rebuild_event_checks()
        self._refresh_group_list()
        self._map_panel.setVisible(False)
        self._psth_view._gw.clear()
        self._block_view._p_amp.clear()
        self._block_view._p_lag.clear()

    # ── Conversion ────────────────────────────────────────────────────────────

    def _to_group_specs(self) -> list[GroupSpec]:
        specs = []
        for g in self._groups:
            si = next((i for i, (_, d) in enumerate(self._sessions)
                       if d is g.session_data), 0)
            specs.append(GroupSpec(
                label=g.label, color=g.color,
                unit_indices=list(g.unit_indices),
                session_idx=si))
        return specs

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _refresh(self):
        if not self._groups or not self._sessions:
            return
        specs      = self._to_group_specs()
        slots      = self._get_display_slots()
        smooth     = self._smooth_sb.value()
        window_sec = self._win_sb.value()
        block_size = self._blk_sb.value()
        normalize  = self._norm_combo.currentData()
        y_label    = 'FR (norm)' if normalize != 'none' else 'FR (Hz)'

        self._psth_view.refresh(
            self._sessions, specs, slots,
            smooth, window_sec, None, normalize, y_label)

        self._block_view.set_events(
            self._sessions, slots, smooth, window_sec, None, block_size)
        self._block_view.refresh(specs)
