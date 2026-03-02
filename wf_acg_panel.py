"""
wf_acg_panel.py  –  Waveform (8-channel) + Autocorrelogram panel.

Layout (horizontal split):
  Left   : Multi-channel waveform.  Channels are spatially offset by their
            real Y position on the probe.  Main channel is highlighted.
  Right  : Autocorrelogram bar chart (±50 ms).

Updates whenever a new unit is selected.  Shows a message when SST is not
loaded.
"""

import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QFileDialog
)
from PyQt5.QtCore import Qt

BG_COLOR   = '#16213e'
FG_COLOR   = '#e0e0e0'
ZERO_COLOR = '#555577'
WF_COLOR   = '#4a90d9'
ACG_COLOR  = '#2196F3'

WF_SPACING = 1.5   # inter-channel spacing in normalised units (matches MATLAB)


def _make_plot(gw, title=''):
    """Add a styled PlotItem to an existing GraphicsLayoutWidget."""
    p = gw.addPlot(title=title)
    p.getViewBox().setBackgroundColor(BG_COLOR)
    p.showGrid(x=False, y=False)
    p.getAxis('bottom').setPen(FG_COLOR)
    p.getAxis('left').setPen(FG_COLOR)
    p.getAxis('bottom').setTextPen(FG_COLOR)
    p.getAxis('left').setTextPen(FG_COLOR)
    if title:
        p.setTitle(title, color=FG_COLOR, size='9pt')
    return p


class WFACGPanel(QWidget):
    """
    Shows waveform + ACG for the currently selected unit.
    Requires data.get_wf() and data.get_acg() to be available
    (i.e. SST file was loaded).
    """

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data = data
        self.setMinimumHeight(220)
        self.setMaximumHeight(320)
        # Persistent graphics items — reused across unit changes
        self._wf_fills: list = []
        self._wf_lines: list = []
        self._wf_n_ch: int   = -1
        self._acg_bars       = None
        self._build_ui()

    # ── UI ─────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- "Load SST" banner shown when SST is missing ----
        self._no_sst_banner = QWidget()
        banner_layout = QHBoxLayout(self._no_sst_banner)
        banner_layout.setContentsMargins(8, 4, 8, 4)
        lbl = QLabel("SST file not loaded — WF and ACG unavailable.")
        lbl.setStyleSheet("color: #7777aa; font-size: 9pt;")
        load_btn = QPushButton("Load sst.mat…")
        load_btn.setFixedHeight(24)
        load_btn.setStyleSheet(
            "QPushButton { background: #1a1a3e; color: #e0e0e0; "
            "border: 1px solid #3a3a6a; border-radius: 3px; padding: 2px 8px; }"
            "QPushButton:hover { background: #2a2a6e; }"
        )
        load_btn.clicked.connect(self._load_sst_dialog)
        banner_layout.addWidget(lbl)
        banner_layout.addStretch()
        banner_layout.addWidget(load_btn)
        outer.addWidget(self._no_sst_banner)

        # ---- Plots ----
        plots_widget = QWidget()
        plots_layout = QHBoxLayout(plots_widget)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(4)

        # WF — wrap GraphicsLayoutWidget in a QWidget so addWidget accepts it
        wf_container = QWidget()
        wf_inner = QVBoxLayout(wf_container)
        wf_inner.setContentsMargins(0, 0, 0, 0)
        self._wf_gw = pg.GraphicsLayoutWidget()
        self._wf_gw.setBackground(BG_COLOR)
        self._wf_plot = _make_plot(self._wf_gw, title='Waveform')
        self._wf_plot.hideAxis('left')
        self._wf_plot.hideAxis('bottom')
        wf_inner.addWidget(self._wf_gw)

        # ACG — same pattern
        acg_container = QWidget()
        acg_inner = QVBoxLayout(acg_container)
        acg_inner.setContentsMargins(0, 0, 0, 0)
        self._acg_gw = pg.GraphicsLayoutWidget()
        self._acg_gw.setBackground(BG_COLOR)
        self._acg_plot = _make_plot(self._acg_gw, title='ACG')
        self._acg_plot.setLabel('bottom', 'Lag (ms)', color=FG_COLOR)
        self._acg_plot.hideAxis('left')
        acg_inner.addWidget(self._acg_gw)

        plots_layout.addWidget(wf_container,  3)
        plots_layout.addWidget(acg_container, 2)
        outer.addWidget(plots_widget)

        # Start with correct visibility
        self._refresh_banner()

    # ── Loading ────────────────────────────────────────────────────────────────
    def _load_sst_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open SST file", "",
            "MAT files (*.mat);;All files (*)"
        )
        if path:
            try:
                self.data.load_sst(path)
                self._refresh_banner()
            except Exception as e:
                from PyQt5.QtWidgets import QMessageBox
                QMessageBox.critical(self, "Load error", str(e))

    def _refresh_banner(self):
        self._no_sst_banner.setVisible(not self.data._sst_loaded)

    # ── Update ─────────────────────────────────────────────────────────────────
    def update_unit(self, unit_idx: int):
        self._refresh_banner()
        self._draw_wf(unit_idx)
        self._draw_acg(unit_idx)

    # ── Waveform ───────────────────────────────────────────────────────────────
    def _draw_wf(self, unit_idx: int):
        self._wf_plot.setTitle(
            f'Waveform  (unit {unit_idx})', color=FG_COLOR, size='9pt'
        )

        result = self.data.get_wf(unit_idx)
        if result is None:
            if self._wf_n_ch != -1:
                self._wf_plot.clear()
                self._wf_fills = []
                self._wf_lines = []
                self._wf_n_ch  = -1
            return

        wf, wf_std, main_ch = result   # (8,81), (8,81), int
        n_ch, n_t = wf.shape
        t = np.arange(n_t, dtype=np.float64)

        # Normalise by max absolute amplitude across all channels (MATLAB style)
        max_amp = np.abs(wf).max()
        if max_amp == 0:
            max_amp = 1.0
        wf_n  = wf     / max_amp
        std_n = wf_std / max_amp

        # Re-allocate persistent items only when channel count changes
        if n_ch != self._wf_n_ch:
            self._wf_plot.clear()
            self._wf_fills = []
            self._wf_lines = []
            fill_color = pg.mkColor(WF_COLOR)
            fill_color.setAlphaF(0.20)
            line_pen = pg.mkPen(color=WF_COLOR, width=1.5)
            for _ in range(n_ch):
                fill = pg.PlotCurveItem(pen=pg.mkPen(None),
                                        brush=pg.mkBrush(fill_color),
                                        antialias=True)
                self._wf_plot.addItem(fill)
                self._wf_fills.append(fill)
                line = pg.PlotCurveItem(pen=line_pen, antialias=True)
                self._wf_plot.addItem(line)
                self._wf_lines.append(line)
            self._wf_n_ch = n_ch
            self._wf_plot.setXRange(t[0], t[-1], padding=0.02)
            self._wf_plot.setYRange(-WF_SPACING, n_ch * WF_SPACING, padding=0.05)

        # Update curve data in-place (no scene rebuild)
        for ch in range(n_ch):
            y_off  = ch * WF_SPACING
            m      = wf_n[ch]
            s      = std_n[ch]
            t_fill = np.concatenate([t, t[::-1]])
            y_fill = np.concatenate([m + s + y_off, (m - s + y_off)[::-1]])
            self._wf_fills[ch].setData(t_fill, y_fill)
            self._wf_lines[ch].setData(t, m + y_off)

    # ── ACG ────────────────────────────────────────────────────────────────────
    def _draw_acg(self, unit_idx: int):
        self._acg_plot.setTitle(
            f'ACG  (unit {unit_idx})', color=FG_COLOR, size='9pt'
        )

        result = self.data.get_acg(unit_idx)
        if result is None:
            if self._acg_bars is not None:
                self._acg_plot.removeItem(self._acg_bars)
                self._acg_bars = None
            return

        counts, bins_ms = result
        bin_w = bins_ms[1] - bins_ms[0]   # full bin width — bars touch

        if self._acg_bars is None:
            self._acg_bars = pg.BarGraphItem(
                x=bins_ms, height=counts, width=bin_w,
                brush=pg.mkBrush(ACG_COLOR), pen=pg.mkPen(None)
            )
            self._acg_plot.addItem(self._acg_bars)
        else:
            # Update in-place — no scene rebuild
            self._acg_bars.setOpts(x=bins_ms, height=counts, width=bin_w)

        self._acg_plot.setXRange(bins_ms[0] - bin_w / 2,
                                 bins_ms[-1] + bin_w / 2, padding=0)
        self._acg_plot.setYRange(0, counts.max() * 1.1 + 1, padding=0)
