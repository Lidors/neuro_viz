"""
control_panel.py  –  Right-side parameter panel.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QGroupBox, QFormLayout,
    QDoubleSpinBox, QSpinBox, QLabel,
    QComboBox, QPushButton, QHBoxLayout
)
from PyQt5.QtCore import pyqtSignal, Qt

DARK_GROUP = (
    "QGroupBox { color: #aaaacc; font-weight: bold; font-size: 9pt; "
    "border: 1px solid #3a3a6a; border-radius: 5px; margin-top: 8px; "
    "padding-top: 6px; } "
    "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
)
DARK_INPUT = (
    "QDoubleSpinBox, QSpinBox, QComboBox { "
    "background: #0f0f2a; color: #e0e0e0; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px; }"
)
DARK_BTN = (
    "QPushButton { background: #1a1a3e; color: #e0e0e0; "
    "border: 1px solid #3a3a6a; border-radius: 4px; padding: 4px 8px; }"
    "QPushButton:hover  { background: #2a2a6e; }"
    "QPushButton:pressed { background: #3a5a9a; }"
)


class ControlPanel(QWidget):
    # ── Signals ───────────────────────────────────────────────────────────────
    xlim_changed       = pyqtSignal(float, float)   # (lo, hi) seconds
    clim_changed       = pyqtSignal(float, float)   # (lo, hi) Hz
    smooth_changed     = pyqtSignal(int)             # sigma in samples
    block_size_changed = pyqtSignal(int)             # trials per block
    colormap_changed        = pyqtSignal(str)    # heatmap cmap name
    block_line_cmap_changed = pyqtSignal(str)    # block-line cmap name
    add_event_clicked  = pyqtSignal()               # "Add event file" button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(230)
        self.setStyleSheet("QWidget { background: #0d0d2a; }")
        self._build_ui()

    # ── Build ──────────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ---- Time axis ----
        xgrp = QGroupBox("Time axis (s)")
        xgrp.setStyleSheet(DARK_GROUP)
        xform = QFormLayout(xgrp)
        xform.setSpacing(5)

        self._x_lo = self._make_dspin(-30.0, 0.0, -3.0, step=0.5)
        self._x_hi = self._make_dspin(0.0, 30.0,  3.0, step=0.5)
        self._x_lo.valueChanged.connect(self._emit_xlim)
        self._x_hi.valueChanged.connect(self._emit_xlim)
        xform.addRow("From:", self._x_lo)
        xform.addRow("To:",   self._x_hi)
        layout.addWidget(xgrp)

        # ---- Colour limits ----
        cgrp = QGroupBox("Colour limits (Hz)")
        cgrp.setStyleSheet(DARK_GROUP)
        cform = QFormLayout(cgrp)
        cform.setSpacing(5)

        self._c_lo = self._make_dspin(0.0, 500.0, 0.0, step=1.0, decimals=1)
        self._c_hi = self._make_dspin(0.0, 500.0, 20.0, step=1.0, decimals=1)
        self._c_lo.valueChanged.connect(self._emit_clim)
        self._c_hi.valueChanged.connect(self._emit_clim)
        cform.addRow("Min:", self._c_lo)
        cform.addRow("Max:", self._c_hi)
        layout.addWidget(cgrp)

        # ---- Smoothing ----
        sgrp = QGroupBox("Smoothing  (Gaussian σ)")
        sgrp.setStyleSheet(DARK_GROUP)
        sform = QFormLayout(sgrp)
        sform.setSpacing(5)

        self._smooth = QSpinBox()
        self._smooth.setRange(0, 200)
        self._smooth.setValue(10)           # default: 10 samples
        self._smooth.setSuffix(" samples")
        self._smooth.setStyleSheet(DARK_INPUT)
        self._smooth.valueChanged.connect(lambda v: self.smooth_changed.emit(v))

        self._smooth_ms_label = QLabel()
        self._smooth_ms_label.setStyleSheet("color: #7777aa; font-size: 8pt;")
        self._smooth.valueChanged.connect(self._update_smooth_label)
        self._update_smooth_label(10)

        sform.addRow("σ:", self._smooth)
        sform.addRow("", self._smooth_ms_label)
        layout.addWidget(sgrp)

        # ---- Block size ----
        bgrp = QGroupBox("Block size  (trials/block)")
        bgrp.setStyleSheet(DARK_GROUP)
        bform = QFormLayout(bgrp)
        bform.setSpacing(5)

        self._block_size = QSpinBox()
        self._block_size.setRange(1, 500)
        self._block_size.setValue(10)       # default: 10 trials
        self._block_size.setSuffix(" trials")
        self._block_size.setStyleSheet(DARK_INPUT)
        self._block_size.valueChanged.connect(
            lambda v: self.block_size_changed.emit(v)
        )
        bform.addRow("Size:", self._block_size)
        layout.addWidget(bgrp)

        # ---- Colormap ----
        cmgrp = QGroupBox("Colormap")
        cmgrp.setStyleSheet(DARK_GROUP)
        cmform = QFormLayout(cmgrp)
        cmform.setSpacing(5)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems([
            'viridis', 'plasma', 'inferno', 'magma',
            'hot', 'coolwarm', 'RdBu_r', 'gray'
        ])
        self._cmap_combo.setStyleSheet(DARK_INPUT)
        self._cmap_combo.currentTextChanged.connect(
            lambda t: self.colormap_changed.emit(t)
        )
        cmform.addRow("Heatmap:", self._cmap_combo)

        self._line_cmap_combo = QComboBox()
        self._line_cmap_combo.addItems([
            'plasma', 'viridis', 'inferno', 'magma',
            'turbo', 'coolwarm', 'RdYlGn', 'rainbow', 'hot'
        ])
        self._line_cmap_combo.setStyleSheet(DARK_INPUT)
        self._line_cmap_combo.currentTextChanged.connect(
            lambda t: self.block_line_cmap_changed.emit(t)
        )
        cmform.addRow("Block lines:", self._line_cmap_combo)

        layout.addWidget(cmgrp)

        # ---- Events ----
        evgrp = QGroupBox("Events")
        evgrp.setStyleSheet(DARK_GROUP)
        evlay = QVBoxLayout(evgrp)
        evlay.setSpacing(5)

        add_ev_btn = QPushButton("+ Add event file…")
        add_ev_btn.setStyleSheet(DARK_BTN)
        add_ev_btn.clicked.connect(self.add_event_clicked)
        evlay.addWidget(add_ev_btn)
        layout.addWidget(evgrp)

        # ---- Navigation ----
        navgrp = QGroupBox("Unit navigation")
        navgrp.setStyleSheet(DARK_GROUP)
        navlay = QVBoxLayout(navgrp)
        navlay.setSpacing(5)

        nav_row = QHBoxLayout()
        self._prev_btn = QPushButton("◀ Prev")
        self._next_btn = QPushButton("Next ▶")
        self._prev_btn.setStyleSheet(DARK_BTN)
        self._next_btn.setStyleSheet(DARK_BTN)
        nav_row.addWidget(self._prev_btn)
        nav_row.addWidget(self._next_btn)
        navlay.addLayout(nav_row)

        self._unit_label = QLabel("Unit: –")
        self._unit_label.setAlignment(Qt.AlignCenter)
        self._unit_label.setStyleSheet("color: #aaaacc; font-size: 8pt;")
        navlay.addWidget(self._unit_label)
        layout.addWidget(navgrp)

        layout.addStretch()

    # ── Helpers ────────────────────────────────────────────────────────────────
    def _make_dspin(self, lo, hi, val, step=0.5, decimals=1) -> QDoubleSpinBox:
        sb = QDoubleSpinBox()
        sb.setRange(lo, hi)
        sb.setSingleStep(step)
        sb.setValue(val)
        sb.setDecimals(decimals)
        sb.setStyleSheet(DARK_INPUT)
        return sb

    def _update_smooth_label(self, n_samples: int):
        # Show equivalent in ms assuming ~60 fps camera (16.7 ms/sample)
        ms = n_samples * (1000 / 60)
        self._smooth_ms_label.setText(f"≈ {ms:.0f} ms at 60 fps")

    def _emit_xlim(self):
        lo, hi = self._x_lo.value(), self._x_hi.value()
        if lo < hi:
            self.xlim_changed.emit(lo, hi)

    def _emit_clim(self):
        lo, hi = self._c_lo.value(), self._c_hi.value()
        if lo < hi:
            self.clim_changed.emit(lo, hi)

    # ── Public helpers ─────────────────────────────────────────────────────────
    def set_unit_label(self, unit_idx: int, cell_type: str, depth: int):
        self._unit_label.setText(
            f"Unit {unit_idx}  ·  {cell_type}  ·  {depth} µm"
        )

    def connect_nav(self, prev_fn, next_fn):
        self._prev_btn.clicked.connect(prev_fn)
        self._next_btn.clicked.connect(next_fn)
