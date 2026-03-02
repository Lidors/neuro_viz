"""
unit_table.py
Left-panel widget – shows a sortable list of units with metadata.

Selection model
---------------
* Single-click a row  → unit_selected(int)  – loads that unit in the PSTH panel
* Checkbox column (★) → units_selected(list) – drives the compare-set independently
  Checkboxes persist across row-selection changes, so Ctrl/Shift issues are avoided.
* Click a Pair cell   → pair_compare_requested(list) – opens compare for the CS/SS pair
"""

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidget, QTableWidgetItem,
    QAbstractItemView, QLineEdit, QLabel, QHBoxLayout, QHeaderView,
    QPushButton
)
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QColor, QFont

_CHECK_COL = 0   # compare-toggle checkbox column
_PAIR_COL  = 4   # pair column (shifted +1 because of checkbox col)


class _NumericTableItem(QTableWidgetItem):
    """QTableWidgetItem that sorts numerically when the text is a number."""
    def __lt__(self, other):
        try:
            return float(self.text()) < float(other.text())
        except ValueError:
            return self.text() < other.text()


class UnitTableWidget(QWidget):
    """
    Displays all units in a sortable table.

    Columns: ★ | # | Type | Depth (µm) | Pair | n R1 | n R2 | n NR1 | n NR2
    """
    unit_selected          = pyqtSignal(int)   # primary selection (single unit)
    units_selected         = pyqtSignal(list)  # compare-set changed (checkbox driven)
    pair_compare_requested = pyqtSignal(list)  # [unit_idx, partner_idx] → open compare

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self._filtered_indices = list(range(data.n_units))
        self._compare_indices: set = set()   # units checked for comparison
        self._build_ui()
        self._populate()

    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Filter box
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Filter:"))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("cell type / depth…")
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit)
        layout.addLayout(filter_row)

        # Unit count + select-all + clear-compare buttons
        info_row = QHBoxLayout()
        self._count_label = QLabel()
        self._count_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        info_row.addWidget(self._count_label, stretch=1)

        _mini_style = (
            "QPushButton { background: #1a1a3e; color: #aaaacc; "
            "border: 1px solid #3a3a6a; border-radius: 2px; font-size: 8pt; padding: 0 4px; }"
            "QPushButton:hover { background: #2a2a5e; }"
        )

        self._selall_btn = QPushButton("★ All")
        self._selall_btn.setFixedHeight(18)
        self._selall_btn.setToolTip("Check all currently visible units")
        self._selall_btn.setStyleSheet(_mini_style)
        self._selall_btn.clicked.connect(self._select_all_filtered)
        info_row.addWidget(self._selall_btn)

        self._clear_btn = QPushButton("Clear ★")
        self._clear_btn.setFixedHeight(18)
        self._clear_btn.setStyleSheet(_mini_style)
        self._clear_btn.clicked.connect(self._clear_compare)
        info_row.addWidget(self._clear_btn)
        layout.addLayout(info_row)

        # Table — trial-count columns are built from whatever events exist
        self._ev_trial_keys = list(self.data.event_keys)
        cols = (['★', '#', 'Type', 'Depth (µm)', 'Pair']
                + [f'n {k}' for k in self._ev_trial_keys])
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.setAlternatingRowColors(True)
        self._table.setSortingEnabled(True)
        self._table.itemSelectionChanged.connect(self._on_selection)
        self._table.itemChanged.connect(self._on_item_changed)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.horizontalHeader().sortIndicatorChanged.connect(self._sync_order)

        # Make the table compact
        self._table.verticalHeader().setDefaultSectionSize(22)
        font = QFont()
        font.setPointSize(9)
        self._table.setFont(font)

        layout.addWidget(self._table)

    # ------------------------------------------------------------------
    def _populate(self, indices=None):
        if indices is None:
            indices = list(range(self.data.n_units))
        self._filtered_indices = indices

        # Block itemChanged so checkbox restores don't fire _on_item_changed
        self._table.itemChanged.disconnect(self._on_item_changed)
        self._table.setSortingEnabled(False)
        self._table.setRowCount(len(indices))

        n_trials = self.data.n_trials
        pair_map = getattr(self.data, 'pair_map', {})

        for row, unit_idx in enumerate(indices):
            depth = (self.data.probe_depth[unit_idx]
                     if self.data.probe_depth[unit_idx] > 0
                     else self.data.probe_pos[unit_idx])

            # Resolve pair
            if unit_idx in pair_map:
                partner   = pair_map[unit_idx]
                pair_text = str(partner) if partner is not None else '\u2013'
            else:
                partner   = None
                pair_text = ''

            is_beh = (hasattr(self.data, 'beh_unit_indices')
                      and unit_idx in self.data.beh_unit_indices)

            # ── Column 0: compare checkbox ────────────────────────────────
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            chk.setCheckState(
                Qt.Checked if unit_idx in self._compare_indices else Qt.Unchecked)
            chk.setData(Qt.UserRole, unit_idx)   # real index lives here
            chk.setTextAlignment(Qt.AlignCenter)
            chk.setToolTip("Check to add to compare set")
            if is_beh:
                chk.setBackground(QColor('#0d2b25'))
            self._table.setItem(row, _CHECK_COL, chk)

            # ── Columns 1+: unit metadata ─────────────────────────────────
            ev_counts = [str(n_trials.get(k, '-')) for k in self._ev_trial_keys]
            texts = (
                [str(unit_idx),                        # col 1  #
                 self.data.cell_type_label[unit_idx],  # col 2  Type
                 str(depth),                           # col 3  Depth
                 pair_text]                            # col 4  Pair
                + ev_counts                            # col 5+
            )
            n_text_cols = len(texts)
            numeric_cols = {1, 3, 4} | set(range(5, n_text_cols + 1))

            for offset, text in enumerate(texts):
                col  = offset + 1
                item = (_NumericTableItem(text) if col in numeric_cols
                        else QTableWidgetItem(text))
                item.setTextAlignment(Qt.AlignCenter)

                if is_beh:
                    item.setBackground(QColor('#0d2b25'))
                    item.setForeground(QColor('#80cbc4'))
                elif col == _PAIR_COL and pair_text:
                    item.setForeground(QColor('#FFD54F'))
                    if partner is not None:
                        item.setData(Qt.UserRole + 1, partner)
                        f = item.font(); f.setUnderline(True); item.setFont(f)
                        item.setToolTip(
                            f"Click to compare unit {unit_idx} \u2194 unit {partner}")
                elif col == 2 and text == 'PC':
                    item.setForeground(QColor('#1976D2'))

                self._table.setItem(row, col, item)

        self._table.setSortingEnabled(True)
        self._table.itemChanged.connect(self._on_item_changed)

        cmp_n = len(self._compare_indices)
        cmp_str = f"  |  ★ {cmp_n}" if cmp_n else ""
        self._count_label.setText(
            f"{len(indices)} / {self.data.n_units} units{cmp_str}")

    # ------------------------------------------------------------------
    def _apply_filter(self, text):
        text = text.strip().lower()
        if not text:
            self._populate()
            return
        matched = [
            i for i in range(self.data.n_units)
            if text in self.data.cell_type_label[i].lower()
            or text in str(self.data.probe_pos[i])
            or text in str(i)
        ]
        self._populate(matched)

    # ------------------------------------------------------------------
    def _sync_order(self):
        """Rebuild _filtered_indices to match the current visual row order."""
        self._filtered_indices = [
            self._table.item(r, _CHECK_COL).data(Qt.UserRole)
            for r in range(self._table.rowCount())
            if self._table.item(r, _CHECK_COL) is not None
        ]

    def _unit_idx_for_row(self, row: int) -> int | None:
        item = self._table.item(row, _CHECK_COL)
        if item is None:
            return None
        return item.data(Qt.UserRole)

    # ------------------------------------------------------------------
    def _on_item_changed(self, item):
        """Checkbox in the compare column was toggled."""
        if item.column() != _CHECK_COL:
            return
        uid = item.data(Qt.UserRole)
        if uid is None:
            return
        if item.checkState() == Qt.Checked:
            self._compare_indices.add(uid)
        else:
            self._compare_indices.discard(uid)
        self.units_selected.emit(sorted(self._compare_indices))
        # Update count label
        cmp_n = len(self._compare_indices)
        visible = self._table.rowCount()
        cmp_str = f"  |  ★ {cmp_n}" if cmp_n else ""
        self._count_label.setText(
            f"{visible} / {self.data.n_units} units{cmp_str}")

    def _select_all_filtered(self):
        """Check ★ for every unit currently visible in the table."""
        self._table.itemChanged.disconnect(self._on_item_changed)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _CHECK_COL)
            if item is not None:
                uid = item.data(Qt.UserRole)
                if uid is not None:
                    self._compare_indices.add(uid)
                    item.setCheckState(Qt.Checked)
        self._table.itemChanged.connect(self._on_item_changed)
        self.units_selected.emit(sorted(self._compare_indices))
        cmp_n   = len(self._compare_indices)
        visible = self._table.rowCount()
        self._count_label.setText(
            f"{visible} / {self.data.n_units} units  |  ★ {cmp_n}")

    def _clear_compare(self):
        """Uncheck all compare checkboxes."""
        self._compare_indices.clear()
        self._table.itemChanged.disconnect(self._on_item_changed)
        for row in range(self._table.rowCount()):
            item = self._table.item(row, _CHECK_COL)
            if item is not None:
                item.setCheckState(Qt.Unchecked)
        self._table.itemChanged.connect(self._on_item_changed)
        self.units_selected.emit([])
        visible = self._table.rowCount()
        self._count_label.setText(f"{visible} / {self.data.n_units} units")

    # ------------------------------------------------------------------
    def _on_selection(self):
        """Row clicked → emit unit_selected for PSTH view only."""
        if getattr(self, '_in_selection', False):
            return
        self._in_selection = True
        try:
            row = self._table.currentRow()
            uid = self._unit_idx_for_row(row)
            if uid is not None:
                self.unit_selected.emit(uid)
        finally:
            self._in_selection = False

    # ------------------------------------------------------------------
    def _on_cell_clicked(self, row: int, col: int):
        """Open pair comparison when the user clicks a clickable Pair cell."""
        if col != _PAIR_COL:
            return
        item = self._table.item(row, col)
        if item is None:
            return
        partner = item.data(Qt.UserRole + 1)
        if partner is None:
            return
        unit_idx = self._unit_idx_for_row(row)
        if unit_idx is None:
            return
        self.pair_compare_requested.emit([unit_idx, partner])

    # ------------------------------------------------------------------
    def select_unit(self, unit_idx: int):
        """Programmatically select a unit by its 0-based index.
        Uses blockSignals so selectRow doesn't re-fire itemSelectionChanged.
        Checkbox states are unaffected (they're independent of row selection)."""
        for row in range(self._table.rowCount()):
            if self._unit_idx_for_row(row) == unit_idx:
                self._table.blockSignals(True)
                self._table.selectRow(row)
                self._table.blockSignals(False)
                break
