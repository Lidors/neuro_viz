"""
psth_panel.py  –  Main PSTH visualization panel.

Top toolbar  : [Heatmap] [Blocks] [Mean PSTH] [Multi-cell]  view-type buttons
Event bar    : one checkbox per event (multi-select = compare mode)
Main area    : stacked view widgets
"""

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QPushButton,
    QButtonGroup, QStackedWidget, QLabel, QCheckBox,
    QFrame, QScrollArea, QSizePolicy, QSplitter, QFileDialog, QMessageBox
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter
from PyQt5.QtPrintSupport import QPrinter


def _render_widget_to_pdf(widget, path: str, parent=None):
    """Render any QWidget to a vector PDF at *path* (landscape A4), white background."""
    from PyQt5.QtWidgets import QMessageBox

    # ── 1. Switch all pyqtgraph plots to light theme ──────────────────────
    pws = widget.findChildren(pg.PlotWidget)
    saved = []
    for pw in pws:
        saved.append({
            'pw': pw,
            'bg': pw.backgroundBrush().color(),
            'ax': {ax: (pw.getAxis(ax).pen(), pw.getAxis(ax).textPen())
                   for ax in ('bottom', 'left')},
        })
        pw.setBackground('w')
        for ax in ('bottom', 'left'):
            pw.getAxis(ax).setPen(pg.mkPen('k'))
            pw.getAxis(ax).setTextPen(pg.mkPen('k'))

    def _restore():
        for s in saved:
            s['pw'].setBackground(s['bg'])
            for ax, (pen, tpen) in s['ax'].items():
                s['pw'].getAxis(ax).setPen(pen)
                s['pw'].getAxis(ax).setTextPen(tpen)

    # ── 2. Set up vector PDF printer ──────────────────────────────────────
    printer = QPrinter(QPrinter.HighResolution)
    printer.setOutputFormat(QPrinter.PdfFormat)
    printer.setOutputFileName(path)
    printer.setPageSize(QPrinter.A4)
    printer.setOrientation(QPrinter.Landscape)

    painter = QPainter()
    if not painter.begin(printer):
        _restore()
        QMessageBox.critical(parent, "PDF error",
                             f"Could not open {path} for writing.")
        return

    # ── 3. Render widget directly to printer (vector output) ──────────────
    page_rect   = printer.pageRect()
    widget_size = widget.size()
    painter.fillRect(page_rect, Qt.white)
    scale_x = page_rect.width()  / max(widget_size.width(),  1)
    scale_y = page_rect.height() / max(widget_size.height(), 1)
    scale   = min(scale_x, scale_y)
    painter.translate(page_rect.left(), page_rect.top())
    painter.scale(scale, scale)
    widget.render(painter)
    painter.end()

    # ── 4. Restore dark theme ─────────────────────────────────────────────
    _restore()


# Dark-mode colours (must match main.py setup)
BG_COLOR   = '#16213e'
FG_COLOR   = '#e0e0e0'
GRID_COLOR = '#2a2a4a'
ZERO_COLOR = '#555577'

# Colour palette for multi-cell view (one colour per selected cell)
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


def _make_colormap(name: str = 'viridis'):
    import matplotlib.pyplot as _plt
    cmap   = _plt.get_cmap(name)
    n      = 256
    colors = (cmap(np.linspace(0, 1, n))[:, :3] * 255).astype(np.uint8)
    pos    = np.linspace(0, 1, n)
    return pg.ColorMap(pos, colors)


def _styled_plot(gw, row=0, col=0, title=''):
    p = gw.addPlot(row=row, col=col, title=title)
    p.getViewBox().setBackgroundColor(BG_COLOR)
    p.showGrid(x=True, y=True, alpha=0.15)
    for ax in ('bottom', 'left'):
        p.getAxis(ax).setPen(FG_COLOR)
        p.getAxis(ax).setTextPen(FG_COLOR)
    if title:
        p.setTitle(title, color=FG_COLOR, size='10pt')
    return p


def _mini_plot(gw, row=0, col=0, title=''):
    """Compact plot for metric mini-panels."""
    p = gw.addPlot(row=row, col=col)
    p.getViewBox().setBackgroundColor(BG_COLOR)
    p.showGrid(x=True, y=True, alpha=0.15)
    for ax in ('bottom', 'left'):
        p.getAxis(ax).setPen(FG_COLOR)
        p.getAxis(ax).setTextPen(FG_COLOR)
        p.getAxis(ax).setStyle(tickFont=QFont('Arial', 7))
    p.setTitle(title, color=FG_COLOR, size='8pt')
    p.setLabel('bottom', 'Block #', color=FG_COLOR)
    return p


def _smooth_rows(matrix: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing along axis-1 (time) for each row."""
    if sigma <= 0:
        return matrix
    out = matrix.copy()
    for i in range(out.shape[0]):
        row = out[i]
        mask = ~np.isnan(row)
        if mask.any():
            tmp = row.copy()
            tmp[~mask] = 0.0
            out[i] = gaussian_filter1d(tmp, sigma)
    return out


# ─── Metric mini-panel (reused in BlocksView and MultiCellView) ──────────────
class _MetricPanel(QWidget):
    """Mini-plot: dot-product amplitude vs block number."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        layout.addWidget(self._gw)
        self._plots = []
        self._ann_lines: list[list[str]] = [[], []]
        self._build()

    def _build(self):
        self._gw.clear()
        self._plots = []
        self._ann_lines = [[], []]
        for i, lbl in enumerate(METRIC_LABELS):
            p = _mini_plot(self._gw, row=0, col=i, title=lbl)
            self._plots.append(p)

    def clear_data(self):
        for p in self._plots:
            p.clear()
        self._ann_lines = [[], []]
        for i, lbl in enumerate(METRIC_LABELS):
            self._plots[i].setTitle(lbl, color=FG_COLOR, size='8pt')

    def plot_series(self, block_nums, metrics_dict, color, name=None):
        """Add one coloured series to each metric plot."""
        from scipy.stats import pearsonr as _pr
        r, g, b = _hex_to_rgb(color)
        pen    = pg.mkPen(color=(r, g, b), width=2)
        brush  = pg.mkBrush(r, g, b)
        for i, key in enumerate(METRIC_KEYS):
            kw = dict(pen=pen, symbol='o', symbolSize=5,
                      symbolBrush=brush, symbolPen=pg.mkPen(None))
            if name:
                kw['name'] = name
            bn   = np.asarray(block_nums, float)
            vals = np.asarray(metrics_dict[key], float)
            self._plots[i].plot(bn, vals, **kw)
            # Pearson r annotation
            try:
                mask = np.isfinite(bn) & np.isfinite(vals)
                if mask.sum() >= 3:
                    rval, pval = _pr(bn[mask], vals[mask])
                    p_str = 'p<0.001' if pval < 0.001 else f'p={pval:.3f}'
                    short = (name[:12] + '…') if name and len(name) > 12 else (name or '')
                    self._ann_lines[i].append(f'{short}  r={rval:.2f} {p_str}')
                    self._plots[i].setTitle(
                        METRIC_LABELS[i] + '  |  ' + '  |  '.join(self._ann_lines[i]),
                        color=FG_COLOR, size='7pt')
            except Exception:
                pass


# ─── Heatmap ─────────────────────────────────────────────────────────────────
class HeatmapView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._gw   = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        layout.addWidget(self._gw)
        self._plots = []
        self._imgs  = []
        self._cmap  = _make_colormap('viridis')
        self._clim  = (0.0, 20.0)
        self._xlim  = (-3.0, 3.0)

    def _rebuild(self, n: int):
        self._gw.clear()
        self._plots, self._imgs = [], []
        cols = min(n, 2)
        for i in range(n):
            p   = _styled_plot(self._gw, row=i // cols, col=i % cols)
            p.setLabel('bottom', 'Time (s)', color=FG_COLOR)
            p.setLabel('left',   'Trial',    color=FG_COLOR)
            p.setXRange(*self._xlim, padding=0)
            img = pg.ImageItem()
            img.setLookupTable(self._cmap.getLookupTable(nPts=256))
            p.addItem(img)
            self._plots.append(p)
            self._imgs.append(img)

    def update_plots(self, data_list, titles, clim, xlim, smooth_samples=0):
        self._clim, self._xlim = clim, xlim
        # Only rebuild axes/items when the number of events changes
        if len(data_list) != len(self._imgs):
            self._rebuild(len(data_list))
        for i, ((aligned, time_ax), title) in enumerate(zip(data_list, titles)):
            self._plots[i].setTitle(title, color=FG_COLOR, size='10pt')
            self._plots[i].setXRange(*xlim, padding=0)
            if aligned is None or aligned.size == 0:
                self._imgs[i].clear()
                continue
            disp = np.nan_to_num(aligned, nan=0.0)          # (n_trials, n_time)
            if smooth_samples > 0:
                disp = _smooth_rows(disp, smooth_samples)
            disp = np.ascontiguousarray(disp.T, dtype=np.float32)  # (n_time, n_trials)
            self._imgs[i].setImage(disp, levels=clim)
            t0, t1   = time_ax[0], time_ax[-1]
            n_trials = aligned.shape[0]
            self._imgs[i].setRect(pg.QtCore.QRectF(t0, 0, t1 - t0, n_trials))
            self._plots[i].setYRange(0, n_trials, padding=0.02)

    def set_clim(self, lo, hi):
        self._clim = (lo, hi)
        for img in self._imgs:
            img.setLevels((lo, hi))

    def set_xlim(self, lo, hi):
        self._xlim = (lo, hi)
        for p in self._plots:
            p.setXRange(lo, hi, padding=0)

    def set_colormap(self, name: str):
        self._cmap = _make_colormap(name)
        lut = self._cmap.getLookupTable(nPts=256)
        for img in self._imgs:
            img.setLookupTable(lut)


# ─── Block PSTH ───────────────────────────────────────────────────────────────
class BlocksView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        splitter = QSplitter(Qt.Vertical)

        # Top: block PSTH line plots
        top = QWidget()
        top_l = QVBoxLayout(top)
        top_l.setContentsMargins(0, 0, 0, 0)
        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        top_l.addWidget(self._gw)
        splitter.addWidget(top)

        # Bottom: metric mini-plots
        self._metric_panel = _MetricPanel()
        splitter.addWidget(self._metric_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

        self._plots         = []
        self._xlim          = (-3.0, 3.0)
        self._line_cmap_name = 'plasma'

    def set_line_colormap(self, name: str):
        self._line_cmap_name = name

    def _rebuild_lines(self, n: int, y_label: str = 'FR (Hz)'):
        self._gw.clear()
        self._plots = []
        cols = min(n, 2)
        for i in range(n):
            p = _styled_plot(self._gw, row=i // cols, col=i % cols)
            p.setLabel('bottom', 'Time (s)', color=FG_COLOR)
            p.setLabel('left',   y_label,    color=FG_COLOR)
            leg = p.addLegend(offset=(-10, 10))
            leg.setLabelTextColor(FG_COLOR)
            p.setXRange(*self._xlim, padding=0)
            p.addItem(pg.InfiniteLine(pos=0, angle=90,
                                      pen=pg.mkPen(ZERO_COLOR, width=1,
                                                   style=Qt.DashLine)))
            self._plots.append(p)

    def update_plots(self, data_list, metrics_list, titles, colors, xlim,
                     y_label: str = 'FR (Hz)'):
        """
        data_list    : [(blk_matrix, time_ax), ...]  one per event
        metrics_list : [metrics_dict, ...]           one per event
        titles       : event labels
        colors       : event hex colours
        """
        self._xlim = xlim
        self._rebuild_lines(len(data_list), y_label)
        self._metric_panel.clear_data()

        import matplotlib.pyplot as _plt
        for i, ((blk, time_ax), title) in enumerate(zip(data_list, titles)):
            self._plots[i].setTitle(title, color=FG_COLOR, size='10pt')
            self._plots[i].setXRange(*xlim, padding=0)
            if blk is None or blk.size == 0:
                continue
            n_blocks = blk.shape[1]
            cmap = _plt.get_cmap(self._line_cmap_name)
            for b in range(n_blocks):
                c   = cmap(b / max(n_blocks - 1, 1))
                rgb = (int(c[0]*255), int(c[1]*255), int(c[2]*255))
                self._plots[i].plot(
                    time_ax, blk[:, b],
                    pen=pg.mkPen(color=rgb, width=1.8),
                    name=f'Blk {b+1}'
                )

        for metrics, color, title in zip(metrics_list, colors, titles):
            if metrics is not None:
                self._metric_panel.plot_series(
                    metrics['block_nums'], metrics, color, name=title)

    def set_xlim(self, lo, hi):
        self._xlim = (lo, hi)
        for p in self._plots:
            p.setXRange(lo, hi, padding=0)


# Colours for pairwise difference lines
_DIFF_COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
    '#DDA0DD', '#FFEAA7', '#82E0AA', '#F0B27A',
]


# ─── Mean PSTH ───────────────────────────────────────────────────────────────
class MeanView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 4)
        layout.setSpacing(2)

        # ── Diff-mode toggle button ───────────────────────────────────────
        hdr = QHBoxLayout()
        hdr.setContentsMargins(4, 2, 4, 0)
        hdr.addStretch()
        self._diff_btn = QPushButton("Δ  Diff mode")
        self._diff_btn.setCheckable(True)
        self._diff_btn.setFixedHeight(22)
        self._diff_btn.setToolTip("Show pairwise differences  (shortcut: D)")
        self._diff_btn.setStyleSheet(
            "QPushButton { background: #1a1a3e; color: #aaaacc; "
            "border: 1px solid #3a3a6a; border-radius: 3px; "
            "font-size: 8pt; padding: 0 8px; }"
            "QPushButton:checked { background: #3a5a9a; color: #ffffff; "
            "border: 1px solid #5a8aca; }"
            "QPushButton:hover { background: #2a2a5e; }"
        )
        hdr.addWidget(self._diff_btn)
        layout.addLayout(hdr)

        self._gw = pg.GraphicsLayoutWidget()
        self._gw.setBackground(BG_COLOR)
        layout.addWidget(self._gw)

        self._plots:    list  = []
        self._xlim            = (-3.0, 3.0)
        self._last_args       = None

        self._diff_btn.toggled.connect(self._on_diff_toggled)

    # ── private ───────────────────────────────────────────────────────────────
    def _on_diff_toggled(self, _):
        if self._last_args:
            self.update_plots(*self._last_args)

    def _new_plot(self, row: int, col: int, title: str,
                  y_label: str, horiz_zero: bool = False):
        p = _styled_plot(self._gw, row=row, col=col, title=title)
        p.setLabel('bottom', 'Time (s)', color=FG_COLOR)
        p.setLabel('left',   y_label,    color=FG_COLOR)
        leg = p.addLegend(offset=(-10, 10))
        leg.setLabelTextColor(FG_COLOR)
        p.setXRange(*self._xlim, padding=0)
        p.addItem(pg.InfiniteLine(pos=0, angle=90,
                                  pen=pg.mkPen(ZERO_COLOR, width=1,
                                               style=Qt.DashLine)))
        if horiz_zero:
            p.addItem(pg.InfiniteLine(pos=0, angle=0,
                                      pen=pg.mkPen(ZERO_COLOR, width=1,
                                                   style=Qt.DashLine)))
        return p

    def _update_regular(self, data_list, titles, colors, xlim, y_label):
        self._gw.clear()
        p = self._new_plot(0, 0, '', y_label)
        self._plots = [p]
        for (mean, sem, time_ax), title, color in zip(data_list, titles, colors):
            if mean is None:
                continue
            r, g, b = _hex_to_rgb(color)
            if sem is not None and np.any(sem > 0):
                upper = pg.PlotDataItem(time_ax, mean + sem)
                lower = pg.PlotDataItem(time_ax, mean - sem)
                p.addItem(pg.FillBetweenItem(upper, lower,
                                             brush=pg.mkBrush(r, g, b, 55)))
            p.plot(time_ax, mean,
                   pen=pg.mkPen(color=(r, g, b), width=2.2),
                   name=title)
        p.setXRange(*xlim, padding=0)

    def _update_diff(self, data_list, titles, colors, xlim, y_label):
        n     = len(data_list)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        self._gw.clear()
        self._plots = []
        cols = min(len(pairs), 2)

        for idx, (i, j) in enumerate(pairs):
            mean_i, sem_i, tax = data_list[i]
            mean_j, sem_j, _   = data_list[j]
            if mean_i is None or mean_j is None:
                continue

            diff     = mean_i - mean_j
            sem_diff = (np.sqrt(sem_i**2 + sem_j**2)
                        if sem_i is not None and sem_j is not None
                        else None)

            p = self._new_plot(idx // cols, idx % cols,
                               f"{titles[i]}  −  {titles[j]}",
                               f'Δ {y_label}', horiz_zero=True)
            self._plots.append(p)

            dc = _DIFF_COLORS[idx % len(_DIFF_COLORS)]
            r, g, b = _hex_to_rgb(dc)
            if sem_diff is not None and np.any(sem_diff > 0):
                upper = pg.PlotDataItem(tax, diff + sem_diff)
                lower = pg.PlotDataItem(tax, diff - sem_diff)
                p.addItem(pg.FillBetweenItem(upper, lower,
                                             brush=pg.mkBrush(r, g, b, 55)))
            p.plot(tax, diff,
                   pen=pg.mkPen(color=(r, g, b), width=2.2))
            p.setXRange(*xlim, padding=0)

    # ── public ────────────────────────────────────────────────────────────────
    def update_plots(self, data_list, titles, colors, xlim,
                     y_label: str = 'FR (Hz)'):
        self._last_args = (data_list, titles, colors, xlim, y_label)
        self._xlim      = xlim
        if self._diff_btn.isChecked() and len(data_list) >= 2:
            self._update_diff(data_list, titles, colors, xlim, y_label)
        else:
            self._update_regular(data_list, titles, colors, xlim, y_label)

    def toggle_diff(self):
        """Toggle diff mode on/off (keyboard shortcut entry point)."""
        self._diff_btn.setChecked(not self._diff_btn.isChecked())

    def set_xlim(self, lo, hi):
        self._xlim = (lo, hi)
        for p in self._plots:
            p.setXRange(lo, hi, padding=0)
        if self._last_args:
            args    = list(self._last_args)
            args[3] = (lo, hi)
            self._last_args = tuple(args)


# ─── Multi-cell view ──────────────────────────────────────────────────────────
class MultiCellView(QWidget):
    """
    Overlaid mean PSTHs for multiple selected units + block metrics.
    One colour per unit.  Uses individual PlotWidgets in a QGridLayout to
    avoid pyqtgraph GraphicsLayoutWidget recursion issues.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4)
        outer.setSpacing(4)

        # Control row
        ctrl = QHBoxLayout()
        self._norm_cb = QCheckBox("Normalize (0–1 per cell)")
        self._norm_cb.setStyleSheet(
            f"QCheckBox {{ color: {FG_COLOR}; font-size: 9pt; }}")
        self._norm_cb.stateChanged.connect(self._on_normalize)
        ctrl.addWidget(self._norm_cb)
        ctrl.addStretch()
        outer.addLayout(ctrl)

        # Splitter: top = mean PSTH plots, bottom = block metrics
        splitter = QSplitter(Qt.Vertical)

        # Top: plain widget with grid layout — PlotWidgets go here
        self._plots_container = QWidget()
        self._plots_container.setStyleSheet(
            f"background: {BG_COLOR};")
        self._plots_grid = QGridLayout(self._plots_container)
        self._plots_grid.setContentsMargins(0, 0, 0, 0)
        self._plots_grid.setSpacing(4)
        splitter.addWidget(self._plots_container)

        self._metric_panel = _MetricPanel()
        splitter.addWidget(self._metric_panel)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        outer.addWidget(splitter)

        self._pw_list  = []   # active PlotWidgets
        self._last_args = None
        self._xlim = (-3.0, 3.0)

    # ── helpers ───────────────────────────────────────────────────────────────
    def _clear_plots(self):
        for pw in self._pw_list:
            self._plots_grid.removeWidget(pw)
            pw.hide()
            pw.deleteLater()
        self._pw_list = []

    def _make_pw(self, title: str, y_label: str) -> pg.PlotWidget:
        pw = pg.PlotWidget()
        pw.setBackground(BG_COLOR)
        pw.showGrid(x=True, y=True, alpha=0.15)
        for ax in ('bottom', 'left'):
            pw.getAxis(ax).setPen(FG_COLOR)
            pw.getAxis(ax).setTextPen(FG_COLOR)
        pw.setLabel('bottom', 'Time (s)', color=FG_COLOR)
        pw.setLabel('left', y_label, color=FG_COLOR)
        leg = pw.addLegend(offset=(-10, 10))
        leg.setLabelTextColor(FG_COLOR)
        pw.addItem(pg.InfiniteLine(pos=0, angle=90,
                                   pen=pg.mkPen(ZERO_COLOR, width=1,
                                                style=Qt.DashLine)))
        pw.setXRange(*self._xlim, padding=0)
        pw.setTitle(title, color=FG_COLOR, size='10pt')
        return pw

    # ── slots ─────────────────────────────────────────────────────────────────
    def _on_normalize(self, _):
        if self._last_args:
            self.update_plots(*self._last_args)

    # ── public ────────────────────────────────────────────────────────────────
    def update_plots(self, unit_indices, event_keys, data,
                     xlim, block_size, smooth_samples):
        self._last_args = (unit_indices, event_keys, data,
                           xlim, block_size, smooth_samples)
        normalize = self._norm_cb.isChecked()
        self._xlim = xlim

        self._clear_plots()
        self._metric_panel.clear_data()

        n_ev  = len(event_keys)
        cols  = min(n_ev, 2)
        y_lbl = 'Norm FR' if normalize else 'FR (Hz)'

        # Create one PlotWidget per event BEFORE adding to layout
        pws = []
        for j, key in enumerate(event_keys):
            lbl = data.get_event_label(key)
            pw  = self._make_pw(title=lbl, y_label=y_lbl)
            pws.append(pw)
            self._pw_list.append(pw)

        # Add to grid after creation (ViewBoxMenu is built before insertion)
        for j, pw in enumerate(pws):
            self._plots_grid.addWidget(pw, j // cols, j % cols)

        # Draw each selected cell
        for ci, unit_idx in enumerate(unit_indices):
            color      = CELL_COLORS[ci % len(CELL_COLORS)]
            r, g, b    = _hex_to_rgb(color)
            cell_label = f"U{unit_idx} {data.cell_type_label[unit_idx]}"

            for ji, key in enumerate(event_keys):
                mean, sem, tax = data.get_mean_psth(
                    unit_idx, key, smooth_samples)
                if normalize:
                    mn, mx = mean.min(), mean.max()
                    rng = mx - mn
                    if rng > 1e-9:
                        mean = (mean - mn) / rng
                        sem  = sem  / rng

                if sem is not None and np.any(sem > 0):
                    upper = pg.PlotDataItem(tax, mean + sem)
                    lower = pg.PlotDataItem(tax, mean - sem)
                    pws[ji].addItem(
                        pg.FillBetweenItem(upper, lower,
                                           brush=pg.mkBrush(r, g, b, 40)))
                pws[ji].plot(
                    tax, mean,
                    pen=pg.mkPen(color=(r, g, b), width=2),
                    name=cell_label if ji == 0 else None
                )

                if ji == 0:
                    try:
                        metrics = data.compute_block_metrics(
                            unit_idx, key, block_size, smooth_samples)
                        self._metric_panel.plot_series(
                            metrics['block_nums'], metrics, color,
                            name=cell_label)
                    except Exception:
                        pass

    def set_xlim(self, lo, hi):
        self._xlim = (lo, hi)
        for pw in self._pw_list:
            pw.setXRange(lo, hi, padding=0)
        if self._last_args:
            args        = list(self._last_args)
            args[3]     = (lo, hi)
            self._last_args = tuple(args)


# ─── PSTHPanel ────────────────────────────────────────────────────────────────
class PSTHPanel(QWidget):
    """
    Top toolbar  : view-type toggle buttons
    Event bar    : one coloured checkbox per event  (multi-select → compare)
    Main area    : stacked view widgets
    """
    plot_updated = pyqtSignal()

    def __init__(self, data, parent=None):
        super().__init__(parent)
        self.data            = data
        self._unit_idx       = 0
        self._multi_units    = [0]
        self._clim           = (0.0, 20.0)
        self._xlim           = (-3.0, 3.0)
        self._smooth_samples = 10
        self._block_size     = 10
        self._event_checks: dict[str, QCheckBox] = {}
        self._build_ui()
        self._populate_event_bar()

    # ── UI build ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # ---- View-type toolbar ----
        vbar = QHBoxLayout()
        vbar.addWidget(self._make_section_label("View:"))
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        for idx, lbl in enumerate(['Heatmap', 'Blocks', 'Mean PSTH']):
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setChecked(idx == 0)
            btn.setFixedHeight(28)
            btn.setStyleSheet(self._btn_style())
            self._view_group.addButton(btn, idx)
            vbar.addWidget(btn)
        vbar.addStretch()
        layout.addLayout(vbar)
        self._view_group.buttonClicked.connect(self._on_view_changed)

        # ---- Event checkbox bar ----
        ebar_outer = QHBoxLayout()
        ebar_outer.addWidget(self._make_section_label("Events:"))

        self._event_bar = QHBoxLayout()
        self._event_bar.setSpacing(8)
        event_widget = QWidget()
        event_widget.setLayout(self._event_bar)

        scroll = QScrollArea()
        scroll.setWidget(event_widget)
        scroll.setWidgetResizable(True)
        scroll.setFixedHeight(36)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        ebar_outer.addWidget(scroll)
        ebar_outer.addStretch()
        layout.addLayout(ebar_outer)

        # ---- Status label ----
        self._status = QLabel()
        self._status.setStyleSheet(f"color: {FG_COLOR}; font-size: 9pt;")
        layout.addWidget(self._status)

        # ---- Stacked views ----
        self._stack     = QStackedWidget()
        self._heatmap   = HeatmapView()
        self._blocks    = BlocksView()
        self._mean_view = MeanView()
        self._stack.addWidget(self._heatmap)    # 0
        self._stack.addWidget(self._blocks)     # 1
        self._stack.addWidget(self._mean_view)  # 2
        layout.addWidget(self._stack)

    def _make_section_label(self, text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet(
            f"color: {FG_COLOR}; font-weight: bold; font-size: 9pt;")
        return lbl

    def _btn_style(self) -> str:
        return (
            f"QPushButton {{ background: #1a1a3e; color: {FG_COLOR}; "
            f"border: 1px solid #3a3a6a; border-radius: 4px; padding: 2px 10px; }}"
            f"QPushButton:checked {{ background: #3a5a9a; color: white; "
            f"border: 1px solid #5a8aCA; }}"
            f"QPushButton:hover {{ background: #2a2a5e; }}"
        )

    # ── Event checkboxes ──────────────────────────────────────────────────────
    def _populate_event_bar(self):
        while self._event_bar.count():
            item = self._event_bar.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._event_checks.clear()

        for key in self.data.event_keys:
            self._add_event_checkbox(key)

        checks = list(self._event_checks.values())
        if checks:
            checks[0].setChecked(True)

    def _add_event_checkbox(self, key: str):
        label = self.data.get_event_label(key)
        color = self.data.get_event_color(key)
        r, g, b = _hex_to_rgb(color)

        cb = QCheckBox(label)
        cb.setChecked(False)
        cb.setStyleSheet(
            f"QCheckBox {{ color: {color}; font-weight: bold; font-size: 9pt; }}"
            f"QCheckBox::indicator:checked {{ background-color: {color}; "
            f"border: 2px solid {color}; border-radius: 3px; }}"
            f"QCheckBox::indicator:unchecked {{ background-color: #1a1a3e; "
            f"border: 2px solid {color}; border-radius: 3px; }}"
        )
        cb.stateChanged.connect(self._on_event_changed)
        self._event_bar.addWidget(cb)
        self._event_checks[key] = cb

    def add_event_to_bar(self, key: str):
        if key not in self._event_checks:
            self._add_event_checkbox(key)

    # ── Slots ─────────────────────────────────────────────────────────────────
    def _on_view_changed(self, btn):
        self._stack.setCurrentIndex(self._view_group.id(btn))
        self.refresh()

    def _on_event_changed(self, _state):
        checked = [k for k, cb in self._event_checks.items() if cb.isChecked()]
        if not checked:
            sender = self.sender()
            if isinstance(sender, QCheckBox):
                sender.blockSignals(True)
                sender.setChecked(True)
                sender.blockSignals(False)
        self.refresh()

    # ── Public API ────────────────────────────────────────────────────────────
    def set_unit(self, unit_idx: int):
        self._unit_idx = unit_idx
        self.refresh()

    def set_units(self, unit_indices: list):
        """Store multi-cell selection (used by Compare dialog). No refresh."""
        if unit_indices:
            self._multi_units = unit_indices

    def get_active_event_keys(self) -> list:
        return [k for k in self.data.event_keys
                if self._event_checks.get(k, QCheckBox()).isChecked()]

    def set_clim(self, lo: float, hi: float):
        self._clim = (lo, hi)
        self._heatmap.set_clim(lo, hi)

    def set_xlim(self, lo: float, hi: float):
        self._xlim = (lo, hi)
        view_idx = self._stack.currentIndex()
        if view_idx == 1:
            # Blocks view: full refresh so metrics are recomputed in new window
            self.refresh()
        else:
            # Other views: just pan the axes
            self._heatmap.set_xlim(lo, hi)
            self._blocks.set_xlim(lo, hi)
            self._mean_view.set_xlim(lo, hi)

    def set_smooth(self, smooth_samples: int):
        self._smooth_samples = smooth_samples
        self.refresh()

    def set_block_size(self, block_size: int):
        self._block_size = block_size
        if self._stack.currentIndex() == 1:
            self.refresh()

    def set_colormap(self, name: str):
        self._heatmap.set_colormap(name)

    def set_block_line_cmap(self, name: str):
        self._blocks.set_line_colormap(name)
        if self._stack.currentIndex() == 1:
            self.refresh()

    # ── Refresh ───────────────────────────────────────────────────────────────
    def refresh(self):
        u        = self._unit_idx
        view_idx = self._stack.currentIndex()

        keys = self.get_active_event_keys()
        if not keys:
            return

        y_label = self.data.get_unit_ylabel(u)

        n_info = '  '.join(f"{k}: {self.data.n_trials[k]}" for k in keys)
        self._status.setText(
            f"Unit {u}  |  {self.data.cell_type_label[u]}"
            f"  |  depth {self.data.probe_pos[u]}"
            f"  |  trials: {n_info}"
            f"  |  smooth={self._smooth_samples} samp  block={self._block_size}"
        )

        if view_idx == 0:   # ── Heatmap ──
            data_list = [self.data.get_heatmap(u, k) for k in keys]
            self._heatmap.update_plots(
                data_list,
                [self.data.get_event_label(k) for k in keys],
                clim=self._clim, xlim=self._xlim,
                smooth_samples=self._smooth_samples
            )

        elif view_idx == 1:  # ── Blocks ──
            data_list, metrics_list, colors = [], [], []
            for key in keys:
                blk, tax = self.data.compute_block_psth(
                    u, key, self._block_size, self._smooth_samples)
                data_list.append((blk, tax))
                try:
                    m = self.data.compute_block_metrics(
                        u, key, self._block_size, self._smooth_samples,
                        xlim=self._xlim)
                except Exception:
                    m = None
                metrics_list.append(m)
                colors.append(self.data.get_event_color(key))
            self._blocks.update_plots(
                data_list, metrics_list,
                [self.data.get_event_label(k) for k in keys],
                colors, xlim=self._xlim, y_label=y_label
            )

        elif view_idx == 2:  # ── Mean PSTH ──
            data_list, colors = [], []
            for key in keys:
                mean, sem, tax = self.data.get_mean_psth(
                    u, key, self._smooth_samples)
                data_list.append((mean, sem, tax))
                colors.append(self.data.get_event_color(key))
            self._mean_view.update_plots(
                data_list,
                [self.data.get_event_label(k) for k in keys],
                colors, xlim=self._xlim, y_label=y_label
            )

        self.plot_updated.emit()

    # ── PDF export ────────────────────────────────────────────────────────────
    def save_pdf(self):
        """Render the current view tab to a PDF file chosen by the user."""
        view_names = {0: 'Heatmap', 1: 'Blocks', 2: 'MeanPSTH'}
        view_idx   = self._stack.currentIndex()
        default    = f"U{self._unit_idx}_{view_names.get(view_idx, 'view')}.pdf"

        path, _ = QFileDialog.getSaveFileName(
            self, "Save PDF", default, "PDF files (*.pdf)")
        if not path:
            return

        widget  = self._stack.currentWidget()
        _render_widget_to_pdf(widget, path, self)
