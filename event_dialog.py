"""
event_dialog.py  –  Dialog for loading new event files.

Workflow:
  1. User picks a .mat file
  2. App finds all 1-D numeric arrays in the file
  3. User checks the ones to add, gives each a display name
  4. User chooses time units  (neural samples @ 30 kHz  OR  seconds)
  5. On OK, returns {event_key -> (times_array, units, display_name)}
"""

import numpy as np
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel,
    QLineEdit, QPushButton, QTableWidget, QTableWidgetItem,
    QCheckBox, QComboBox, QDialogButtonBox,
    QFileDialog, QMessageBox, QWidget, QHeaderView,
    QAbstractItemView
)
from PyQt5.QtCore import Qt

DARK = (
    "QDialog, QWidget { background: #0d0d2a; color: #e0e0e0; }"
    "QLabel { color: #e0e0e0; }"
    "QLineEdit { background: #1a1a3e; color: #e0e0e0; "
    "           border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px; }"
    "QPushButton { background: #1a1a3e; color: #e0e0e0; "
    "              border: 1px solid #3a3a6a; border-radius: 4px; "
    "              padding: 4px 10px; }"
    "QPushButton:hover { background: #2a2a6e; }"
    "QComboBox { background: #1a1a3e; color: #e0e0e0; "
    "            border: 1px solid #3a3a6a; padding: 2px; }"
    "QTableWidget { background: #0f0f2a; color: #e0e0e0; "
    "               gridline-color: #2a2a5a; }"
    "QHeaderView::section { background: #1a1a4a; color: #aaaacc; "
    "                       border: 1px solid #2a2a5a; padding: 4px; }"
    "QCheckBox { color: #e0e0e0; }"
    "QDialogButtonBox QPushButton { min-width: 70px; }"
)


class AddEventDialog(QDialog):
    """
    Returns via `get_result()`:
        { event_key: {'times': np.ndarray, 'units': str, 'label': str} }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add events from file")
        self.setMinimumSize(560, 400)
        self.setStyleSheet(DARK)
        self._candidates: dict[str, np.ndarray] = {}
        self._result: dict = {}
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # File picker row
        file_row = QHBoxLayout()
        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select a .mat file…")
        self._path_edit.setReadOnly(True)
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self._path_edit)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Instructions
        hint = QLabel(
            "Detected 1-D numeric arrays (event timing vectors).\n"
            "Check the ones you want to add and give each a display name."
        )
        hint.setStyleSheet("color: #7777aa; font-size: 8pt;")
        layout.addWidget(hint)

        # Variable table: [✓] | Variable name | Size | Display name
        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(['Add', 'Variable', 'N events', 'Display name'])
        self._table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.NoSelection)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self._table.setMinimumHeight(180)
        layout.addWidget(self._table)

        # Time units row
        units_row = QHBoxLayout()
        units_row.addWidget(QLabel("Time units in file:"))
        self._units_combo = QComboBox()
        self._units_combo.addItems([
            "Neural samples  (30 000 Hz)",
            "Frame index",
        ])
        units_row.addWidget(self._units_combo)
        units_row.addStretch()
        layout.addLayout(units_row)

        # OK / Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
            Qt.Horizontal
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── File loading ───────────────────────────────────────────────────────────
    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select event file", "",
            "MAT files (*.mat);;All files (*)"
        )
        if not path:
            return
        self._path_edit.setText(path)
        self._load_candidates(path)

    def _load_candidates(self, path: str):
        from data_loader import SessionData
        candidates = SessionData.detect_event_vectors(path)

        if not candidates:
            QMessageBox.warning(
                self, "No events found",
                "No 1-D numeric arrays were found in this file.\n"
                "Make sure the file contains timing vectors."
            )
            return

        self._candidates = candidates
        self._table.setRowCount(len(candidates))

        for row, (var_name, arr) in enumerate(candidates.items()):
            # Checkbox
            cb = QCheckBox()
            cb.setChecked(True)
            cb_cell = QWidget()
            cb_lay  = QHBoxLayout(cb_cell)
            cb_lay.addWidget(cb)
            cb_lay.setAlignment(Qt.AlignCenter)
            cb_lay.setContentsMargins(0, 0, 0, 0)
            self._table.setCellWidget(row, 0, cb_cell)

            # Variable name (read-only)
            vname_item = QTableWidgetItem(var_name)
            vname_item.setFlags(Qt.ItemIsEnabled)
            self._table.setItem(row, 1, vname_item)

            # Size
            size_item = QTableWidgetItem(str(len(arr)))
            size_item.setFlags(Qt.ItemIsEnabled)
            size_item.setTextAlignment(Qt.AlignCenter)
            self._table.setItem(row, 2, size_item)

            # Editable display name (defaults to variable name)
            name_edit = QLineEdit(var_name)
            self._table.setCellWidget(row, 3, name_edit)

        self._table.resizeRowsToContents()

    # ── Accept ─────────────────────────────────────────────────────────────────
    def _on_accept(self):
        units_str = ('samples'
                     if self._units_combo.currentIndex() == 0
                     else 'index')
        var_names = list(self._candidates.keys())
        self._result = {}

        for row in range(self._table.rowCount()):
            cb_cell = self._table.cellWidget(row, 0)
            cb      = cb_cell.findChild(QCheckBox)
            if cb is None or not cb.isChecked():
                continue

            name_edit = self._table.cellWidget(row, 3)
            label     = name_edit.text().strip() or var_names[row]
            # Use the label as the event key (strip spaces → underscore)
            key       = label.replace(' ', '_')
            times     = self._candidates[var_names[row]]

            self._result[key] = {
                'times': times,
                'units': units_str,
                'label': label,
            }

        self.accept()

    # ── Public ─────────────────────────────────────────────────────────────────
    def get_result(self) -> dict:
        """
        Returns dict of events to add:
            { event_key: {'times': np.ndarray, 'units': str, 'label': str} }
        """
        return self._result
