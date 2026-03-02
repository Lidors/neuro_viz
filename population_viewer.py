"""
population_viewer.py – Population-level PSTH analysis window.

Defines groups of cells (by cell type or manual selection) and displays:
  • Group mean PSTH ± SEM  (one plot per event, all groups overlaid)
  • Block metrics averaged across the group  (Amplitude + Lag vs block #)

Cross-session support
---------------------
  Load additional sessions with "Load session…".  Each group can belong to
  any loaded session (chosen in GroupDialog).

  Event area shows one row of checkboxes per session.  The Event Mapping
  panel (collapsible, shown when >1 session) lets you link events across
  sessions so that mapped events share one PSTH column.

  Each "display slot" is a dict {session_idx: actual_event_key} that
  represents one column in the plot.

Block metric mode
-----------------
  Per-cell    : compute metrics per cell → average + SEM across cells
  Population  : average cell PSTHs first → single population trace (no SEM)
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field

import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget, QLabel,
    QCheckBox, QPushButton, QScrollArea, QFrame, QLineEdit, QListWidget,
    QListWidgetItem, QAbstractItemView, QTabWidget, QComboBox,
    QDialogButtonBox, QRadioButton, QButtonGroup, QSizePolicy,
    QTableWidget, QTableWidgetItem, QTableWidgetSelectionRange, QHeaderView,
    QFileDialog, QMessageBox, QGridLayout,
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QColor

BG_COLOR    = '#16213e'
FG_COLOR    = '#e0e0e0'
ZERO_COLOR  = '#555577'
PANEL_COLOR = '#1a1a3e'

GROUP_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800',
    '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A',
    '#E91E63', '#607D8B', '#CDDC39', '#795548',
]

_LS = f"color: {FG_COLOR}; font-size: 9pt;"   # label style
_BTN = (
    "QPushButton { background: #1a1a3e; color: #e0e0e0; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px 8px; }"
    "QPushButton:hover { background: #2a2a5e; }"
)
_COMBO_STYLE = (
    f"background: #0d1b2a; color: {FG_COLOR}; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 1px 4px;"
)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _hex_to_rgb(h: str) -> tuple:
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _lbl(text: str) -> QLabel:
    w = QLabel(text)
    w.setStyleSheet(_LS)
    return w


def _btn(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setFixedHeight(24)
    b.setStyleSheet(_BTN)
    return b


def _styled_plot(gw, row: int = 0, col: int = 0, title: str = ''):
    p = gw.addPlot(row=row, col=col, title=title)
    p.getViewBox().setBackgroundColor(BG_COLOR)
    p.showGrid(x=True, y=True, alpha=0.15)
    for ax in ('bottom', 'left'):
        p.getAxis(ax).setPen(FG_COLOR)
        p.getAxis(ax).setTextPen(FG_COLOR)
        p.getAxis(ax).setStyle(tickFont=QFont('Arial', 8))
    if title:
        p.setTitle(title, color=FG_COLOR, size='9pt')
    return p


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GroupSpec:
    label: str
    color: str
    unit_indices: list = field(default_factory=list)
    session_idx: int = 0   # index into PopulationViewer._sessions

    @property
    def n(self) -> int:
        return len(self.unit_indices)


@dataclass
class CompareGroup:
    """A group exported to the Cross-Session Comparison window.

    Carries its own SessionData reference so it is fully self-contained.
    """
    label: str
    color: str
    unit_indices: list
    session_data: object   # SessionData
    session_name: str


# ─── Display-slot helper ──────────────────────────────────────────────────────

def _compute_display_slots(
    ev_checks: dict,          # {session_idx: {key: QCheckBox}}
    map_panel: object,        # _EventMappingPanel instance
    sessions: list,           # list of (name, SessionData)
) -> list[dict[int, str]]:
    """Build event display slots from checked events and the mapping panel.

    Returns a list of dicts, each mapping ``session_idx → actual_event_key``.
    One entry = one column in the PSTH plot.

    Algorithm
    ---------
    1. Start with session-0 checked events (primary slots).
    2. For each primary key, look up mapped keys in other sessions; if those
       events are also checked, include them in the same slot.
    3. Any checked events from other sessions not yet claimed get their own
       slot (session-specific event, shown only for that session's groups).
    """
    checked: dict[int, set[str]] = {}
    for si, ev_dict in ev_checks.items():
        for key, cb in ev_dict.items():
            if cb.isChecked():
                checked.setdefault(si, set()).add(key)

    if not checked:
        return []

    slots: list[dict[int, str]] = []
    used:  set[tuple[int, str]] = set()

    # Session-0 events first — these define the primary slots
    for key in ev_checks.get(0, {}).keys():          # preserve original order
        if key not in checked.get(0, set()):
            continue
        if (0, key) in used:
            continue
        slot: dict[int, str] = {0: key}
        used.add((0, key))
        for si in range(1, len(sessions)):
            mapped = map_panel.get_mapped_key(key, si)
            if mapped and mapped in checked.get(si, set()):
                slot[si] = mapped
                used.add((si, mapped))
        slots.append(slot)

    # Events from other sessions that were not captured above
    for si in range(1, len(sessions)):
        for key in ev_checks.get(si, {}).keys():    # preserve original order
            if key in checked.get(si, set()) and (si, key) not in used:
                slots.append({si: key})
                used.add((si, key))

    return slots


# ─── Color picker button ──────────────────────────────────────────────────────

class _ColorButton(QPushButton):
    color_changed = pyqtSignal(str)

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._color = color
        self.setFixedSize(22, 22)
        self._apply()
        self.clicked.connect(self._pick)

    def _apply(self):
        self.setStyleSheet(
            f"QPushButton {{ background: {self._color}; "
            f"border: 1px solid #888; border-radius: 3px; }}"
        )

    def _pick(self):
        from PyQt5.QtWidgets import QColorDialog
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._apply()
            self.color_changed.emit(self._color)

    @property
    def color(self) -> str:
        return self._color

    @color.setter
    def color(self, c: str):
        self._color = c
        self._apply()


# ─── Sortable numeric table item ─────────────────────────────────────────────

class _NumItem(QTableWidgetItem):
    """QTableWidgetItem that sorts by numeric value, not string."""
    def __init__(self, value: float, display: str | None = None):
        super().__init__(display if display is not None else str(int(value)))
        self._val = float(value)

    def __lt__(self, other):
        if isinstance(other, _NumItem):
            return self._val < other._val
        return super().__lt__(other)


# ─── Group definition dialog ──────────────────────────────────────────────────

class GroupDialog(QDialog):
    """Create or edit a GroupSpec.

    Tab 1 – Cell Type : pick one or more cell types; all matching cells included.
    Tab 2 – Manual    : pick individual cells from a searchable table.

    When multiple sessions are loaded a session selector is shown at the top so
    the user can pick which session the group should belong to.
    """

    def __init__(self, sessions: list, existing: GroupSpec | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Define Group")
        self.setMinimumSize(420, 480)
        self.setStyleSheet(
            f"background: {BG_COLOR}; color: {FG_COLOR};"
            "QLineEdit, QListWidget { background: #0d1b2a; "
            f"color: {FG_COLOR}; border: 1px solid #3a3a6a; }}"
        )
        self._sessions       = sessions
        self._cur_sess_idx   = existing.session_idx if existing else 0
        self._data           = sessions[self._cur_sess_idx][1]

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        # ── Session selector (only if >1 session) ─────────────────────────────
        if len(sessions) > 1:
            sess_row = QHBoxLayout()
            sess_row.addWidget(_lbl("Session:"))
            self._sess_combo = QComboBox()
            self._sess_combo.setFixedHeight(24)
            self._sess_combo.setStyleSheet(_COMBO_STYLE)
            for name, _ in sessions:
                self._sess_combo.addItem(name)
            self._sess_combo.setCurrentIndex(self._cur_sess_idx)
            self._sess_combo.currentIndexChanged.connect(self._on_sess_changed)
            sess_row.addWidget(self._sess_combo, stretch=1)
            layout.addLayout(sess_row)
        else:
            self._sess_combo = None

        # ── Label + color ─────────────────────────────────────────────────────
        row = QHBoxLayout()
        row.addWidget(_lbl("Label:"))
        self._label_edit = QLineEdit(existing.label if existing else "Group")
        self._label_edit.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; "
            "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px 6px;")
        row.addWidget(self._label_edit, stretch=1)
        row.addSpacing(6)
        row.addWidget(_lbl("Color:"))
        self._color_btn = _ColorButton(existing.color if existing else GROUP_COLORS[0])
        row.addWidget(self._color_btn)
        layout.addLayout(row)

        # ── Tabs ──────────────────────────────────────────────────────────────
        tabs_style = (
            f"QTabWidget::pane {{ border: 1px solid #3a3a6a; background: {BG_COLOR}; }}"
            f"QTabBar::tab {{ background: #1a1a3e; color: {FG_COLOR}; "
            f"  padding: 4px 14px; border: 1px solid #3a3a6a; }}"
            f"QTabBar::tab:selected {{ background: #2a2a5e; }}"
        )
        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(tabs_style)

        # Tab 1: cell type ─────────────────────────────────────────────────────
        ct_w = QWidget()
        ct_l = QVBoxLayout(ct_w)
        ct_l.addWidget(_lbl("Select cell type(s) — all matching cells will be included:"))
        self._ct_list = QListWidget()
        self._ct_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self._ct_list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; border: 1px solid #3a3a6a;")
        self._populate_ct()
        ct_l.addWidget(self._ct_list)
        self._tabs.addTab(ct_w, "Cell Type")

        # Tab 2: manual ────────────────────────────────────────────────────────
        man_w = QWidget()
        man_l = QVBoxLayout(man_w)
        man_l.addWidget(_lbl("Select individual cells — click column headers to sort:"))
        self._search = QLineEdit()
        self._search.setPlaceholderText("Filter…")
        self._search.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; "
            "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px 6px;")
        self._search.textChanged.connect(self._filter_manual)
        man_l.addWidget(self._search)

        self._manual_tbl = QTableWidget(0, 3)
        self._manual_tbl.setHorizontalHeaderLabels(["Unit #", "Depth (µm)", "Cell Type"])
        self._manual_tbl.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._manual_tbl.setSelectionMode(QAbstractItemView.MultiSelection)
        self._manual_tbl.setSortingEnabled(True)
        self._manual_tbl.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._manual_tbl.verticalHeader().setVisible(False)
        self._manual_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self._manual_tbl.horizontalHeader().setDefaultAlignment(Qt.AlignLeft)
        self._manual_tbl.setStyleSheet(
            f"QTableWidget {{ background: #0d1b2a; color: {FG_COLOR}; "
            f"  border: 1px solid #3a3a6a; gridline-color: #2a2a5a; }}"
            f"QHeaderView::section {{ background: #1a1a3e; color: {FG_COLOR}; "
            f"  border: 1px solid #3a3a6a; padding: 2px 4px; }}"
            f"QTableWidget::item:selected {{ background: #2a4a8a; }}"
        )
        self._populate_manual()
        man_l.addWidget(self._manual_tbl)
        self._tabs.addTab(man_w, "Manual")

        layout.addWidget(self._tabs)

        if existing:
            self._restore(existing)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.setStyleSheet(f"color: {FG_COLOR};")
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # ── session switch ────────────────────────────────────────────────────────

    def _on_sess_changed(self, idx: int):
        self._cur_sess_idx = idx
        self._data = self._sessions[idx][1]
        self._populate_ct()
        self._populate_manual()

    # ── populate helpers ──────────────────────────────────────────────────────

    def _populate_ct(self):
        self._ct_list.clear()
        labels = getattr(self._data, 'cell_type_label', []) or []
        for ct in sorted(set(t for t in labels if t)):
            count = sum(1 for t in labels if t == ct)
            item  = QListWidgetItem(f"{ct}  (n={count})")
            item.setData(Qt.UserRole, ct)
            self._ct_list.addItem(item)

    def _populate_manual(self):
        labels = getattr(self._data, 'cell_type_label', []) or []
        depths = getattr(self._data, 'probe_depth', None)
        self._manual_tbl.setSortingEnabled(False)
        self._manual_tbl.setRowCount(self._data.n_units)
        for i in range(self._data.n_units):
            ct    = labels[i] if i < len(labels) else ''
            depth = int(depths[i]) if depths is not None else 0
            uid_item = _NumItem(i)
            uid_item.setData(Qt.UserRole, i)
            self._manual_tbl.setItem(i, 0, uid_item)
            self._manual_tbl.setItem(i, 1, _NumItem(depth))
            self._manual_tbl.setItem(i, 2, QTableWidgetItem(ct))
        self._manual_tbl.setSortingEnabled(True)
        self._manual_tbl.resizeColumnToContents(0)
        self._manual_tbl.resizeColumnToContents(1)

    def _filter_manual(self, text: str):
        low = text.lower()
        for r in range(self._manual_tbl.rowCount()):
            if not low:
                self._manual_tbl.setRowHidden(r, False)
            else:
                row_text = ' '.join(
                    self._manual_tbl.item(r, c).text()
                    for c in range(self._manual_tbl.columnCount())
                    if self._manual_tbl.item(r, c)
                ).lower()
                self._manual_tbl.setRowHidden(r, low not in row_text)

    def _restore(self, spec: GroupSpec):
        self._label_edit.setText(spec.label)
        self._color_btn.color = spec.color
        idx_set = set(spec.unit_indices)
        labels  = getattr(self._data, 'cell_type_label', []) or []
        ct_units: dict[str, set] = {}
        for i, t in enumerate(labels):
            ct_units.setdefault(t, set()).add(i)
        whole = {t for t, ids in ct_units.items() if ids <= idx_set}
        if whole:
            self._tabs.setCurrentIndex(0)
            for r in range(self._ct_list.count()):
                self._ct_list.item(r).setSelected(
                    self._ct_list.item(r).data(Qt.UserRole) in whole)
        else:
            self._tabs.setCurrentIndex(1)
            for r in range(self._manual_tbl.rowCount()):
                uid = self._manual_tbl.item(r, 0).data(Qt.UserRole)
                self._manual_tbl.setRangeSelected(
                    QTableWidgetSelectionRange(r, 0, r, 2), uid in idx_set)

    # ── result ────────────────────────────────────────────────────────────────

    def get_spec(self) -> GroupSpec | None:
        label = self._label_edit.text().strip()
        if not label:
            return None
        color  = self._color_btn.color
        labels = getattr(self._data, 'cell_type_label', []) or []

        if self._tabs.currentIndex() == 0:
            chosen  = {self._ct_list.item(r).data(Qt.UserRole)
                       for r in range(self._ct_list.count())
                       if self._ct_list.item(r).isSelected()}
            indices = [i for i, t in enumerate(labels) if t in chosen]
        else:
            seen, indices = set(), []
            for idx in self._manual_tbl.selectedIndexes():
                r   = idx.row()
                uid = self._manual_tbl.item(r, 0).data(Qt.UserRole)
                if uid not in seen:
                    seen.add(uid)
                    indices.append(uid)

        return (GroupSpec(label=label, color=color,
                          unit_indices=sorted(indices),
                          session_idx=self._cur_sess_idx)
                if indices else None)


# ─── Group manager sidebar ────────────────────────────────────────────────────

class _GroupManager(QWidget):
    groups_changed = pyqtSignal()

    def __init__(self, sessions: list, parent=None):
        super().__init__(parent)
        self._sessions: list = sessions
        self._groups: list[GroupSpec] = []
        self.setFixedWidth(240)
        self.setStyleSheet(f"background: {BG_COLOR};")
        self._build()

    def set_sessions(self, sessions: list):
        self._sessions = sessions

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        layout.addWidget(_lbl("Groups"))

        self._list = QListWidget()
        self._list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; border: 1px solid #3a3a6a;")
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        layout.addWidget(self._list, stretch=1)

        row = QHBoxLayout()
        for text, slot in [("Add", self._add), ("Edit", self._edit),
                            ("Remove", self._remove)]:
            b = _btn(text)
            b.clicked.connect(slot)
            row.addWidget(b)
        layout.addLayout(row)

        auto_btn = _btn("Auto from cell types")
        auto_btn.clicked.connect(self._auto)
        layout.addWidget(auto_btn)

        clear_btn = _btn("Clear all")
        clear_btn.clicked.connect(self._clear_all)
        layout.addWidget(clear_btn)

    def _refresh_list(self):
        self._list.clear()
        multi = len(self._sessions) > 1
        for g in self._groups:
            sess_name = (self._sessions[g.session_idx][0]
                         if g.session_idx < len(self._sessions) else '?')
            suffix    = f"  [{sess_name}]" if multi else ''
            item      = QListWidgetItem(f"● {g.label}  (n={g.n}){suffix}")
            r, gv, b  = _hex_to_rgb(g.color)
            item.setForeground(QColor(r, gv, b))
            self._list.addItem(item)

    def _add(self):
        dlg = GroupDialog(self._sessions, parent=self)
        if dlg.exec_() == QDialog.Accepted:
            spec = dlg.get_spec()
            if spec:
                self._groups.append(spec)
                self._refresh_list()
                self.groups_changed.emit()

    def _edit(self):
        row = self._list.currentRow()
        if row < 0:
            return
        dlg = GroupDialog(self._sessions, existing=self._groups[row], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            spec = dlg.get_spec()
            if spec:
                self._groups[row] = spec
                self._refresh_list()
                self.groups_changed.emit()

    def _remove(self):
        row = self._list.currentRow()
        if row >= 0:
            self._groups.pop(row)
            self._refresh_list()
            self.groups_changed.emit()

    def _auto(self):
        labels = getattr(self._sessions[0][1], 'cell_type_label', []) or []
        types  = sorted(set(t for t in labels if t))
        if not types:
            return
        base = len(self._groups)
        for i, ct in enumerate(types):
            idxs  = [j for j, t in enumerate(labels) if t == ct]
            color = GROUP_COLORS[(base + i) % len(GROUP_COLORS)]
            self._groups.append(
                GroupSpec(label=ct, color=color, unit_indices=idxs, session_idx=0))
        self._refresh_list()
        self.groups_changed.emit()

    def _clear_all(self):
        self._groups.clear()
        self._refresh_list()
        self.groups_changed.emit()

    @property
    def groups(self) -> list[GroupSpec]:
        return list(self._groups)


# ─── Event mapping panel ──────────────────────────────────────────────────────

class _EventMappingPanel(QFrame):
    """Collapsible table that maps primary-session event keys to each session.

    Used to decide which events share a PSTH column across sessions.
    Auto-matched by label on session load; user-editable via dropdowns.
    """

    mapping_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            f"QFrame {{ background: #111128; border: 1px solid #2a2a6a; }}")
        self._combos: dict[str, list[QComboBox | None]] = {}
        self._sessions: list = []
        self._build()

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 4)
        layout.setSpacing(2)

        hdr = QHBoxLayout()
        self._toggle = QPushButton("▶ Event Mapping  (cross-session)")
        self._toggle.setCheckable(True)
        self._toggle.setFlat(True)
        self._toggle.setStyleSheet(
            f"QPushButton {{ color: #aaaacc; font-size: 9pt; "
            f"  text-align: left; border: none; padding: 0; }}"
            f"QPushButton:hover {{ color: {FG_COLOR}; }}"
        )
        self._toggle.toggled.connect(self._on_toggle)
        hdr.addWidget(self._toggle)
        hdr.addStretch()
        layout.addLayout(hdr)

        self._body = QWidget()
        self._body.setVisible(False)
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 4, 0, 0)
        layout.addWidget(self._body)

    def _on_toggle(self, checked: bool):
        self._body.setVisible(checked)
        self._toggle.setText(
            ("▼" if checked else "▶") + " Event Mapping  (cross-session)")

    def update_sessions(self, sessions: list):
        self._sessions = sessions

        while self._body_layout.count():
            item = self._body_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._combos.clear()

        if len(sessions) <= 1:
            return

        primary_data = sessions[0][1]
        primary_keys = primary_data.event_keys

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(160)
        scroll.setStyleSheet(
            f"QScrollArea {{ background: {BG_COLOR}; border: none; }}"
        )

        grid_w = QWidget()
        grid_w.setStyleSheet(f"background: {BG_COLOR};")
        grid   = QGridLayout(grid_w)
        grid.setSpacing(4)
        grid.setContentsMargins(4, 4, 4, 4)

        hdr_lbl = _lbl("<b>Event</b>")
        hdr_lbl.setTextFormat(Qt.RichText)
        grid.addWidget(hdr_lbl, 0, 0)
        for si, (sname, _) in enumerate(sessions):
            sh = _lbl(f"<b>{sname}</b>")
            sh.setTextFormat(Qt.RichText)
            grid.addWidget(sh, 0, si + 1)

        for ri, pkey in enumerate(primary_keys):
            plabel = primary_data.get_event_label(pkey)
            grid.addWidget(_lbl(plabel), ri + 1, 0)
            self._combos[pkey] = [None] * len(sessions)

            for si, (_, sess_data) in enumerate(sessions):
                cb = QComboBox()
                cb.setFixedHeight(22)
                cb.setStyleSheet(_COMBO_STYLE)
                cb.addItem("(none)", userData=None)
                for sk in sess_data.event_keys:
                    cb.addItem(sess_data.get_event_label(sk), userData=sk)

                if si == 0:
                    idx = cb.findData(pkey)
                else:
                    idx = -1
                    for j in range(1, cb.count()):
                        if cb.itemText(j).lower() == plabel.lower():
                            idx = j
                            break
                    if idx < 0:
                        idx = cb.findData(pkey)
                if idx >= 0:
                    cb.setCurrentIndex(idx)

                cb.currentIndexChanged.connect(self._emit_changed)
                grid.addWidget(cb, ri + 1, si + 1)
                self._combos[pkey][si] = cb

        scroll.setWidget(grid_w)
        self._body_layout.addWidget(scroll)

    def _emit_changed(self):
        self.mapping_changed.emit()

    def get_mapped_key(self, primary_key: str, session_idx: int) -> str | None:
        if session_idx == 0:
            return primary_key
        combos = self._combos.get(primary_key)
        if combos and session_idx < len(combos) and combos[session_idx] is not None:
            return combos[session_idx].currentData()
        return None


# ─── Mean PSTH view ───────────────────────────────────────────────────────────

class _PSTHGroupView(QWidget):
    """One plot per event slot; each plot shows group mean ± SEM ribbons."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        layout.addWidget(self._gw)

    def refresh(self, sessions: list, groups: list[GroupSpec],
                event_slots: list[dict],
                smooth: int, window_sec: float, xlim,
                normalize: str, y_label: str):
        """
        Parameters
        ----------
        sessions    : list of (name, SessionData)
        event_slots : list of {session_idx: actual_event_key} dicts.
                      One dict = one PSTH column.
        """
        self._gw.clear()
        if not groups or not event_slots:
            return

        multi  = len(sessions) > 1
        plots  = []

        for col, slot in enumerate(event_slots):
            # Column label: first session's event name; add [session] tag if
            # this slot covers fewer than all sessions (so it's unambiguous)
            first_si, first_key = next(iter(slot.items()))
            ev_label = sessions[first_si][1].get_event_label(first_key)
            if multi and len(slot) < len(sessions):
                ev_label += f"  [{sessions[first_si][0]}]"

            p = _styled_plot(self._gw, row=0, col=col, title=ev_label)
            p.setLabel('bottom', 'Time (s)', color=FG_COLOR)
            p.setLabel('left', y_label, color=FG_COLOR)
            p.addItem(pg.InfiniteLine(
                pos=0, angle=90,
                pen=pg.mkPen(ZERO_COLOR, width=1, style=Qt.DashLine)))

            for g in groups:
                if not g.unit_indices:
                    continue
                sess_idx   = min(g.session_idx, len(sessions) - 1)
                actual_key = slot.get(sess_idx)
                if not actual_key:
                    continue
                sess_data = sessions[sess_idx][1]
                try:
                    mean, sem, tax = sess_data.get_group_mean_psth(
                        g.unit_indices, actual_key, smooth, window_sec, normalize)
                except Exception:
                    continue
                r, gv, b = _hex_to_rgb(g.color)
                pen = pg.mkPen(color=(r, gv, b), width=2)

                upper = p.plot(tax, mean + sem, pen=pg.mkPen(None))
                lower = p.plot(tax, mean - sem, pen=pg.mkPen(None))
                p.addItem(pg.FillBetweenItem(
                    upper, lower, brush=pg.mkBrush(r, gv, b, 45)))
                p.plot(tax, mean, pen=pen, name=f"{g.label} (n={g.n})")

            if xlim:
                p.setXRange(*xlim, padding=0)
            p.addLegend(offset=(-10, 10),
                        labelTextColor=FG_COLOR,
                        brush=pg.mkBrush(BG_COLOR))
            plots.append(p)

        for p in plots[1:]:
            p.setXLink(plots[0])


# ─── Block metrics view ───────────────────────────────────────────────────────

class _BlockGroupView(QWidget):
    """Amplitude and Lag vs block # for each group."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        ctrl.addWidget(_lbl("Event:"))
        self._ev_combo = QComboBox()
        self._ev_combo.setFixedHeight(24)
        self._ev_combo.setStyleSheet(_COMBO_STYLE)
        ctrl.addWidget(self._ev_combo)
        ctrl.addSpacing(16)
        ctrl.addWidget(_lbl("Mode:"))

        self._mode_group = QButtonGroup(self)
        for i, txt in enumerate(["Per-cell (mean ± SEM)", "Population"]):
            rb = QRadioButton(txt)
            rb.setStyleSheet(_LS)
            if i == 0:
                rb.setChecked(True)
            self._mode_group.addButton(rb, i)
            ctrl.addWidget(rb)

        ctrl.addSpacing(16)
        ctrl.addWidget(_lbl("X-axis:"))
        self._xaxis_group = QButtonGroup(self)
        for i, txt in enumerate(["Block #", "Normalized (0–1)"]):
            rb = QRadioButton(txt)
            rb.setStyleSheet(_LS)
            if i == 0:
                rb.setChecked(True)
            self._xaxis_group.addButton(rb, i)
            ctrl.addWidget(rb)

        ctrl.addStretch()
        ctrl_w = QWidget()
        ctrl_w.setLayout(ctrl)
        layout.addWidget(ctrl_w)

        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        layout.addWidget(self._gw)
        self._p_amp = _styled_plot(self._gw, row=0, col=0, title='Amplitude')
        self._p_lag = _styled_plot(self._gw, row=0, col=1, title='Best Lag (ms)')
        self._p_amp.setLabel('bottom', 'Block #', color=FG_COLOR)
        self._p_lag.setLabel('bottom', 'Block #', color=FG_COLOR)

        self._sessions:     list  = []
        self._groups:       list  = []
        self._smooth:       int   = 0
        self._window_sec:   float = 3.0
        self._xlim                = None
        self._block_size:   int   = 10

    def set_events(self, sessions: list, event_slots: list[dict],
                   smooth: int, window_sec: float, xlim, block_size: int):
        """Update which event slots are available in the combo."""
        self._sessions   = sessions
        self._smooth     = smooth
        self._window_sec = window_sec
        self._xlim       = xlim
        self._block_size = block_size

        multi = len(sessions) > 1
        prev  = self._ev_combo.currentText()
        self._ev_combo.blockSignals(True)
        self._ev_combo.clear()
        for slot in event_slots:
            first_si, first_key = next(iter(slot.items()))
            label = sessions[first_si][1].get_event_label(first_key)
            if multi and len(slot) < len(sessions):
                label += f"  [{sessions[first_si][0]}]"
            self._ev_combo.addItem(label, userData=slot)
        idx = self._ev_combo.findText(prev)
        if idx >= 0:
            self._ev_combo.setCurrentIndex(idx)
        self._ev_combo.blockSignals(False)

    def refresh(self, groups: list[GroupSpec]):
        self._groups = groups
        self._draw()

    def _draw(self):
        self._p_amp.clear()
        self._p_lag.clear()
        if not self._sessions or not self._groups or self._ev_combo.count() == 0:
            return

        slot    = self._ev_combo.currentData()   # {session_idx: key}
        mode    = 'population' if self._mode_group.checkedId() == 1 else 'percell'
        x_label = ('Fraction of session'
                   if self._xaxis_group.checkedId() == 1 else 'Block #')
        for p in (self._p_amp, self._p_lag):
            p.setLabel('bottom', x_label, color=FG_COLOR)

        for g in self._groups:
            if not g.unit_indices:
                continue
            sess_idx   = min(g.session_idx, len(self._sessions) - 1)
            actual_key = slot.get(sess_idx)
            if not actual_key:
                continue
            sess_data = self._sessions[sess_idx][1]
            try:
                m = sess_data.compute_group_block_metrics(
                    g.unit_indices, actual_key,
                    self._block_size, self._smooth, self._window_sec,
                    self._xlim, mode)
            except Exception:
                continue
            if not m:
                continue

            r, gv, b = _hex_to_rgb(g.color)
            pen   = pg.mkPen(color=(r, gv, b), width=2)
            brush = pg.mkBrush(r, gv, b)
            bn    = m['block_nums']

            if self._xaxis_group.checkedId() == 1:
                n  = len(bn)
                bn = bn / n if n > 0 else bn

            for p, key_mean, key_sem in [
                (self._p_amp, 'amplitude_mean',   'amplitude_sem'),
                (self._p_lag, 'best_lag_ms_mean', 'best_lag_ms_sem'),
            ]:
                mean = m[key_mean]
                sem  = m[key_sem]
                p.plot(bn, mean, pen=pen, symbol='o', symbolSize=5,
                       symbolBrush=brush, symbolPen=pg.mkPen(None),
                       name=g.label)
                err = pg.ErrorBarItem(
                    x=bn, y=mean, top=sem, bottom=sem,
                    pen=pg.mkPen(color=(r, gv, b), width=1))
                p.addItem(err)

        for p in (self._p_amp, self._p_lag):
            p.addLegend(offset=(-10, 10),
                        labelTextColor=FG_COLOR,
                        brush=pg.mkBrush(BG_COLOR))


# ─── Main window ──────────────────────────────────────────────────────────────

class PopulationViewer(QDialog):
    """Population-level PSTH analysis with optional cross-session comparison."""

    def __init__(self, parent, data,
                 get_xlim_fn, get_smooth_fn, get_block_size_fn,
                 get_window_sec_fn=None,
                 initial_units: list[int] | None = None,
                 export_fn=None):
        super().__init__(parent)
        self.setWindowTitle("Population Viewer")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1200, 800)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._sessions: list[tuple[str, object]] = [("Primary", data)]
        self._get_xlim       = get_xlim_fn
        self._get_smooth     = get_smooth_fn
        self._get_block_size = get_block_size_fn
        self._get_window_sec = get_window_sec_fn or (lambda: 3.0)
        self._export_fn      = export_fn
        # ev_checks: {session_idx: {event_key: QCheckBox}}
        self._ev_checks: dict[int, dict[str, QCheckBox]] = {}

        self._build_ui()

        if initial_units:
            color = GROUP_COLORS[0]
            spec  = GroupSpec(label="Selection", color=color,
                              unit_indices=sorted(initial_units),
                              session_idx=0)
            self._group_mgr._groups.append(spec)
            self._group_mgr._refresh_list()
            self._refresh()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Controls bar ──────────────────────────────────────────────────────
        ctrl_bar = QHBoxLayout()
        ctrl_bar.setSpacing(10)

        ctrl_bar.addWidget(_lbl("Normalize:"))
        self._norm_combo = QComboBox()
        self._norm_combo.setFixedHeight(24)
        self._norm_combo.setStyleSheet(_COMBO_STYLE)
        for label, value in [("None", 'none'), ("Z-score", 'zscore'), ("Peak", 'peak')]:
            self._norm_combo.addItem(label, userData=value)
        self._norm_combo.currentIndexChanged.connect(lambda _: self._refresh())
        ctrl_bar.addWidget(self._norm_combo)
        ctrl_bar.addStretch()

        load_btn = _btn("Load session…")
        load_btn.clicked.connect(self._load_session)
        ctrl_bar.addWidget(load_btn)

        if self._export_fn is not None:
            export_btn = QPushButton("Export to Compare →")
            export_btn.setFixedHeight(24)
            export_btn.setStyleSheet(
                "QPushButton { background: #2a1a3e; color: #ce93d8; "
                "border: 1px solid #6a3a8a; border-radius: 3px; padding: 2px 8px; }"
                "QPushButton:hover { background: #3a2a5e; }"
            )
            export_btn.clicked.connect(self._export)
            ctrl_bar.addWidget(export_btn)

        refresh_btn = _btn("Refresh")
        refresh_btn.clicked.connect(self._refresh)
        ctrl_bar.addWidget(refresh_btn)
        outer.addLayout(ctrl_bar)

        # ── Event area: one checkbox row per session ───────────────────────────
        ev_header = QHBoxLayout()
        ev_header.addWidget(_lbl("Events:"))
        ev_header.addStretch()
        outer.addLayout(ev_header)

        self._ev_area = QWidget()
        self._ev_area_vbox = QVBoxLayout(self._ev_area)
        self._ev_area_vbox.setContentsMargins(4, 0, 4, 0)
        self._ev_area_vbox.setSpacing(2)

        self._ev_scroll = QScrollArea()
        self._ev_scroll.setWidget(self._ev_area)
        self._ev_scroll.setWidgetResizable(True)
        self._ev_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._ev_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._ev_scroll.setFrameShape(QFrame.NoFrame)
        self._ev_scroll.setStyleSheet("QScrollArea { background: transparent; }")
        outer.addWidget(self._ev_scroll)

        self._rebuild_event_checks()   # populate from current sessions

        # ── Event mapping panel ───────────────────────────────────────────────
        self._map_panel = _EventMappingPanel()
        self._map_panel.setVisible(False)
        self._map_panel.mapping_changed.connect(self._refresh)
        outer.addWidget(self._map_panel)

        # ── Main layout: group manager | plots ────────────────────────────────
        main = QHBoxLayout()
        main.setSpacing(0)

        self._group_mgr = _GroupManager(self._sessions, parent=self)
        self._group_mgr.groups_changed.connect(self._on_groups_changed)
        main.addWidget(self._group_mgr)

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

    # ── Event area management ─────────────────────────────────────────────────

    def _rebuild_event_checks(self):
        """Rebuild per-session event checkbox rows from self._sessions."""
        # Clear existing rows
        while self._ev_area_vbox.count():
            item = self._ev_area_vbox.takeAt(0)
            if item.widget():
                item.widget().setParent(None)
        self._ev_checks.clear()

        multi = len(self._sessions) > 1
        for si, (sname, sess_data) in enumerate(self._sessions):
            row_w = QWidget()
            row_l = QHBoxLayout(row_w)
            row_l.setContentsMargins(0, 0, 0, 0)
            row_l.setSpacing(6)

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
        return _compute_display_slots(self._ev_checks, self._map_panel,
                                      self._sessions)

    # ── Session loading ───────────────────────────────────────────────────────

    def _load_session(self):
        from data_loader import SessionData

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "", "MAT files (*.mat)")
        if not path:
            return

        name = os.path.splitext(os.path.basename(path))[0]
        try:
            sess = SessionData(path)
        except Exception as e:
            QMessageBox.critical(self, "Load Error",
                                 f"Could not load session:\n{e}")
            return

        if len(self._sessions) == 1:
            self._sessions[0] = (
                os.path.splitext(os.path.basename(
                    getattr(self._sessions[0][1], '_res_path', 'Primary')))[0],
                self._sessions[0][1]
            )

        self._sessions.append((name, sess))
        self._group_mgr.set_sessions(self._sessions)
        self._group_mgr._refresh_list()
        self._rebuild_event_checks()
        self._map_panel.update_sessions(self._sessions)
        self._map_panel.setVisible(True)
        self._refresh()

    # ── Refresh ───────────────────────────────────────────────────────────────

    def _on_groups_changed(self):
        self._refresh()

    def _refresh(self):
        groups     = self._group_mgr.groups
        slots      = self._get_display_slots()
        smooth     = self._get_smooth()
        window_sec = self._get_window_sec()
        xlim       = self._get_xlim()
        block_size = self._get_block_size()
        normalize  = self._norm_combo.currentData()
        y_label    = 'FR (norm)' if normalize != 'none' else 'FR (Hz)'

        self._psth_view.refresh(
            self._sessions, groups, slots,
            smooth, window_sec, xlim, normalize, y_label)

        self._block_view.set_events(
            self._sessions, slots, smooth, window_sec, xlim, block_size)
        self._block_view.refresh(groups)

    def _export(self):
        if not self._export_fn:
            return
        from compare_window import CompareGroup
        compare_groups = []
        for g in self._group_mgr.groups:
            if not g.unit_indices:
                continue
            si            = min(g.session_idx, len(self._sessions) - 1)
            sname, sdata  = self._sessions[si]
            compare_groups.append(CompareGroup(
                label        = g.label,
                color        = g.color,
                unit_indices = list(g.unit_indices),
                session_data = sdata,
                session_name = sname,
            ))
        if compare_groups:
            self._export_fn(compare_groups)
