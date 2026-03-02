"""
video_viewer.py – Synchronized video + neural activity viewer.

Alignment: fr[:, i]  ==  video frame i  (1:1 index, no fps math).

Controls
--------
  Strip / FR win / Smooth  – top bar
  Browse + filter list     – left panel (text + PC/IN filter, double-click or Add)
  Tracked list             – left panel bottom (colored, removable)
  Offset (frames)          – fine-tune video↔neural alignment
"""
from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from PyQt5.QtCore    import Qt
from PyQt5.QtGui     import QImage, QPixmap, QFont, QColor, QPainter
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget, QLabel,
    QSlider, QSpinBox, QLineEdit, QPushButton, QComboBox,
    QSplitter, QFileDialog,
    QListWidget, QListWidgetItem, QAbstractItemView,
    QSizePolicy, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
)

try:
    import cv2
    _HAVE_CV2 = True
except ImportError:
    _HAVE_CV2 = False

from population_viewer import (
    BG_COLOR, FG_COLOR,
    _hex_to_rgb, _lbl, _btn, _COMBO_STYLE,
)

_SB_STYLE = (
    f"QSpinBox {{ background: #0d1b2a; color: {FG_COLOR}; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 1px 4px; }"
)
_LE_STYLE = (
    f"QLineEdit {{ background: #0d1b2a; color: {FG_COLOR}; "
    "border: 1px solid #3a3a6a; border-radius: 3px; padding: 1px 4px; }"
)
_TYPE_BTN_BASE = (
    "QPushButton { background: #1a1a3e; color: #8888aa; border: 1px solid #3a3a6a; "
    "border-radius: 3px; padding: 1px 4px; font-size: 8pt; }"
    "QPushButton:checked { background: #3a5a9a; color: #e0e0e0; }"
)
_SLIDER_STYLE = (
    "QSlider::groove:horizontal { background: #1a1a3e; height: 6px; border-radius: 3px; }"
    "QSlider::handle:horizontal  { background: #4a90d9; width: 14px; height: 14px; "
    "  margin: -4px 0; border-radius: 7px; }"
    "QSlider::sub-page:horizontal { background: #2a4a8a; border-radius: 3px; }"
)

MAX_STRIP   = 11
FRAME_H     = 200
_FR_WIN_DEF = 120
_NO_VID     = "No video"

_CELL_COLORS = [
    '#4dd0e1', '#ef5350', '#66bb6a', '#ffa726',
    '#ab47bc', '#26c6da', '#d4e157', '#ff7043',
    '#42a5f5', '#ec407a',
]


class _ZoomView(QGraphicsView):
    """QGraphicsView with scroll-to-zoom centered on cursor and drag-to-pan."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setStyleSheet("background: #0a0a1e; border: none;")

    def wheelEvent(self, event):
        factor = 1.25 if event.angleDelta().y() > 0 else 1 / 1.25
        self.scale(factor, factor)

    def mouseDoubleClickEvent(self, event):
        """Double-click resets zoom to fit."""
        self.fitInView(self.scene().sceneRect(), Qt.KeepAspectRatio)


class FrameZoomDialog(QDialog):
    """Persistent window showing a single video frame at full resolution.

    Click any frame in the strip to send it here.
    Scroll to zoom, drag to pan, double-click to fit.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Frame zoom")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(900, 650)
        self.setStyleSheet("background: #0a0a1e;")

        vl = QVBoxLayout(self)
        vl.setContentsMargins(4, 2, 4, 4)
        vl.setSpacing(3)

        self._info = QLabel("Click a frame in the strip to zoom")
        self._info.setStyleSheet(f"color: #8899aa; font-size: 8pt;")
        vl.addWidget(self._info)

        self._view  = _ZoomView()
        self._scene = QGraphicsScene()
        self._item  = QGraphicsPixmapItem()
        self._scene.addItem(self._item)
        self._view.setScene(self._scene)
        vl.addWidget(self._view)

    def show_frame(self, frame_bgr: np.ndarray, frame_idx: int):
        import cv2
        h, w  = frame_bgr.shape[:2]
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
        qimg  = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
        px    = QPixmap.fromImage(qimg)
        self._item.setPixmap(px)
        self._scene.setSceneRect(0, 0, w, h)
        self._view.fitInView(self._scene.sceneRect(), Qt.KeepAspectRatio)
        self._info.setText(
            f"Frame {frame_idx}   {w}×{h}   "
            "— scroll to zoom · drag to pan · double-click to fit")
        self.show()
        self.raise_()


class VideoSyncWindow(QDialog):
    """Synchronized video frame strip + per-cell neural FR traces."""

    def __init__(self, session_data, current_unit: int = 0, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Video + Neural Viewer")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(1500, 750)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._sd = session_data
        self._n_neural = self._sd.fr.shape[1]

        # Video state
        self._cap:           object = None
        self._video_fps:     float  = 30.0
        self._total_vframes: int    = 0

        # Navigation
        self._center_nf:    int = 0
        self._n_strip:      int = 7
        self._fr_window:    int = _FR_WIN_DEF
        self._frame_offset: int = 0
        self._smooth_sigma: int = 0
        self._frame_h:      int = FRAME_H   # dynamic strip frame height

        # Zoom window (lazy-created)
        self._zoom_win: FrameZoomDialog | None = None

        # Tracked cells
        self._tracked_units: list[int]   = []
        self._unit_colors:   dict[int, str] = {}
        self._color_idx:     int          = 0

        # FR plot handles
        self._fr_plots:     list = []
        self._fr_curves:    list = []
        self._center_lines: list = []

        # Filter state
        self._type_filter: str = 'All'

        # Event lookup (frame index = neural = video)
        self._event_vframes: dict[str, np.ndarray] = {}

        self._build_ui()
        self._slider.setRange(0, self._n_neural - 1)
        self._frame_sb.setRange(0, self._n_neural - 1)

        self._build_browse_list()
        self._populate_event_combo()
        self._build_event_video_frames()

        # Pre-track the initially selected unit
        self._add_units([current_unit])
        self._go_to_frame(0)

    # ─────────────────────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # ── Top bar ───────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        self._load_btn = _btn("Load video…")
        self._load_btn.clicked.connect(self._load_video_dialog)
        top.addWidget(self._load_btn)

        self._vid_lbl = QLabel(_NO_VID)
        self._vid_lbl.setStyleSheet(
            "color: #6688aa; font-size: 8pt; font-style: italic;")
        top.addWidget(self._vid_lbl, stretch=1)

        def _sb(lo, hi, val, w, step=1):
            sb = QSpinBox()
            sb.setRange(lo, hi)
            sb.setValue(val)
            sb.setSingleStep(step)
            sb.setFixedSize(w, 24)
            sb.setStyleSheet(_SB_STYLE)
            return sb

        top.addWidget(_lbl("Offset:"))
        self._offset_sb = _sb(-9999, 9999, 0, 64)
        self._offset_sb.valueChanged.connect(self._on_offset_changed)
        top.addWidget(self._offset_sb)
        top.addSpacing(8)

        top.addWidget(_lbl("Strip:"))
        self._nstrip_sb = _sb(1, MAX_STRIP, self._n_strip, 40)
        self._nstrip_sb.valueChanged.connect(self._on_nstrip_changed)
        top.addWidget(self._nstrip_sb)
        top.addSpacing(8)

        top.addWidget(_lbl("FR win:"))
        self._frwin_sb = _sb(10, 5000, self._fr_window, 56, step=10)
        self._frwin_sb.valueChanged.connect(self._on_frwin_changed)
        top.addWidget(self._frwin_sb)
        top.addSpacing(8)

        top.addWidget(_lbl("Smooth σ:"))
        self._smooth_sb = _sb(0, 100, 0, 48)
        self._smooth_sb.valueChanged.connect(self._on_smooth_changed)
        top.addWidget(self._smooth_sb)
        top.addSpacing(8)

        top.addWidget(_lbl("Frame H:"))
        self._frameh_sb = _sb(80, 600, FRAME_H, 56, step=20)
        self._frameh_sb.valueChanged.connect(self._on_frame_h_changed)
        top.addWidget(self._frameh_sb)

        outer.addLayout(top)

        # ── Main area ─────────────────────────────────────────────────────────
        h_split = QSplitter(Qt.Horizontal)
        h_split.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; width: 3px; }")

        # ── Left panel ────────────────────────────────────────────────────────
        left_split = QSplitter(Qt.Vertical)
        left_split.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")
        left_split.setFixedWidth(190)

        # Browse section
        browse_w = QWidget()
        browse_w.setStyleSheet(f"background: {BG_COLOR};")
        browse_vl = QVBoxLayout(browse_w)
        browse_vl.setContentsMargins(4, 4, 4, 4)
        browse_vl.setSpacing(3)

        browse_vl.addWidget(_lbl("Browse units"))

        # Text filter
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText("search…")
        self._filter_edit.setStyleSheet(_LE_STYLE)
        self._filter_edit.setFixedHeight(22)
        self._filter_edit.textChanged.connect(self._apply_filter)
        browse_vl.addWidget(self._filter_edit)

        # Type filter buttons (mutually exclusive)
        type_row = QHBoxLayout()
        type_row.setSpacing(2)
        self._type_btns: dict[str, QPushButton] = {}
        for label in ('All', 'PC', 'IN'):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setChecked(label == 'All')
            btn.setFixedHeight(20)
            btn.setStyleSheet(_TYPE_BTN_BASE)
            btn.clicked.connect(lambda _checked, l=label: self._set_type_filter(l))
            type_row.addWidget(btn)
            self._type_btns[label] = btn
        browse_vl.addLayout(type_row)

        # Browse list (multi-select; double-click or Add to track)
        self._browse_list = QListWidget()
        self._browse_list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; "
            "border: 1px solid #3a3a6a; font-size: 8pt;")
        self._browse_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self._browse_list.itemDoubleClicked.connect(
            lambda item: self._add_units([item.data(Qt.UserRole)]))
        browse_vl.addWidget(self._browse_list, stretch=1)

        add_btn = _btn("Add ↓  (or double-click)")
        add_btn.clicked.connect(self._add_selected_units)
        browse_vl.addWidget(add_btn)
        left_split.addWidget(browse_w)

        # Tracked section
        tracked_w = QWidget()
        tracked_w.setStyleSheet(f"background: {BG_COLOR};")
        tracked_vl = QVBoxLayout(tracked_w)
        tracked_vl.setContentsMargins(4, 4, 4, 4)
        tracked_vl.setSpacing(3)

        tracked_vl.addWidget(_lbl("Tracked cells"))
        self._tracked_list = QListWidget()
        self._tracked_list.setStyleSheet(
            f"background: #0d1b2a; color: {FG_COLOR}; "
            "border: 1px solid #3a3a6a; font-size: 8pt;")
        self._tracked_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        tracked_vl.addWidget(self._tracked_list, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        rm_btn = _btn("Remove")
        rm_btn.clicked.connect(self._remove_tracked)
        btn_row.addWidget(rm_btn)
        clr_btn = _btn("Clear all")
        clr_btn.clicked.connect(self._clear_tracked)
        btn_row.addWidget(clr_btn)
        tracked_vl.addLayout(btn_row)
        left_split.addWidget(tracked_w)

        left_split.setSizes([300, 200])
        h_split.addWidget(left_split)

        # ── Right panel ───────────────────────────────────────────────────────
        right_split = QSplitter(Qt.Vertical)
        right_split.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")

        # Frame strip
        self._strip_w = QWidget()
        self._strip_w.setStyleSheet("background: #0a0a1e;")
        self._strip_w.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._strip_w.setFixedHeight(self._frame_h + 4)
        strip_lay = QHBoxLayout(self._strip_w)
        strip_lay.setContentsMargins(2, 2, 2, 2)
        strip_lay.setSpacing(2)
        strip_lay.addStretch()
        self._frame_labels: list[QLabel] = []
        for strip_idx in range(MAX_STRIP):
            lbl = QLabel()
            lbl.setFixedHeight(self._frame_h)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet("background: #111122; color: #445566; font-size: 8pt;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            lbl.setCursor(Qt.PointingHandCursor)
            # clicking opens the zoom window for this strip position
            lbl.mousePressEvent = (
                lambda _ev, idx=strip_idx: self._on_strip_click(idx))
            strip_lay.addWidget(lbl)
            self._frame_labels.append(lbl)
        strip_lay.addStretch()
        self._strip_layout = strip_lay
        right_split.addWidget(self._strip_w)

        # FR traces
        self._fr_gw = pg.GraphicsLayoutWidget()
        self._fr_gw.setBackground(BG_COLOR)
        right_split.addWidget(self._fr_gw)
        right_split.setStretchFactor(0, 0)
        right_split.setStretchFactor(1, 1)

        h_split.addWidget(right_split)
        h_split.setStretchFactor(0, 0)
        h_split.setStretchFactor(1, 1)
        outer.addWidget(h_split, stretch=1)

        # ── Navigation bar ────────────────────────────────────────────────────
        nav = QHBoxLayout()
        nav.setSpacing(6)

        prev_btn = _btn("◀")
        prev_btn.setFixedWidth(28)
        prev_btn.clicked.connect(self._prev_frame)
        nav.addWidget(prev_btn)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 0)
        self._slider.setStyleSheet(_SLIDER_STYLE)
        self._slider.valueChanged.connect(self._on_slider)
        nav.addWidget(self._slider, stretch=1)

        next_btn = _btn("▶")
        next_btn.setFixedWidth(28)
        next_btn.clicked.connect(self._next_frame)
        nav.addWidget(next_btn)

        nav.addSpacing(8)
        nav.addWidget(_lbl("Frame:"))
        self._frame_sb = QSpinBox()
        self._frame_sb.setRange(0, 0)
        self._frame_sb.setFixedSize(80, 24)
        self._frame_sb.setStyleSheet(_SB_STYLE)
        self._frame_sb.editingFinished.connect(
            lambda: self._go_to_frame(self._frame_sb.value()))
        nav.addWidget(self._frame_sb)

        nav.addSpacing(16)
        nav.addWidget(_lbl("Event:"))
        self._ev_combo = QComboBox()
        self._ev_combo.setFixedHeight(24)
        self._ev_combo.setFixedWidth(120)
        self._ev_combo.setStyleSheet(_COMBO_STYLE)
        nav.addWidget(self._ev_combo)

        prev_ev = _btn("◀ Prev")
        prev_ev.clicked.connect(lambda: self._jump_event(-1))
        nav.addWidget(prev_ev)
        next_ev = _btn("Next ▶")
        next_ev.clicked.connect(lambda: self._jump_event(+1))
        nav.addWidget(next_ev)

        outer.addLayout(nav)

    # ─────────────────────────────────────────────────────────────────────────
    # Browse / filter
    # ─────────────────────────────────────────────────────────────────────────

    def _get_cell_type(self, uid: int) -> str:
        try:
            return str(self._sd.cell_type_label[uid])
        except Exception:
            return ''

    def _build_browse_list(self):
        self._browse_list.clear()
        n = getattr(self._sd, 'n_neural_units', getattr(self._sd, 'n_units', 0))
        for i in range(n):
            ct = self._get_cell_type(i)
            depth = ''
            try:
                depth = f"  {self._sd.probe_depth[i]}µm"
            except Exception:
                pass
            item = QListWidgetItem(f"U{i}  {ct}{depth}")
            item.setData(Qt.UserRole, i)
            self._browse_list.addItem(item)
        self._apply_filter()

    def _apply_filter(self):
        text = self._filter_edit.text().strip().lower()
        for row in range(self._browse_list.count()):
            item = self._browse_list.item(row)
            uid  = item.data(Qt.UserRole)
            ct   = self._get_cell_type(uid)
            text_ok = (not text) or (text in str(uid)) or (text in ct.lower())
            type_ok = (self._type_filter == 'All') or (ct == self._type_filter)
            item.setHidden(not (text_ok and type_ok))

    def _set_type_filter(self, label: str):
        self._type_filter = label
        for l, btn in self._type_btns.items():
            btn.setChecked(l == label)
        self._apply_filter()

    # ─────────────────────────────────────────────────────────────────────────
    # Tracked cells management
    # ─────────────────────────────────────────────────────────────────────────

    def _add_units(self, uids: list[int]):
        changed = False
        for uid in uids:
            if uid not in self._tracked_units:
                self._tracked_units.append(uid)
                color = _CELL_COLORS[self._color_idx % len(_CELL_COLORS)]
                self._unit_colors[uid] = color
                self._color_idx += 1
                changed = True
        if changed:
            self._rebuild_tracked_list()
            self._rebuild_fr_plots()
            self._load_fr_data()
            self._update_fr_at_frame(self._center_nf)

    def _add_selected_units(self):
        uids = [item.data(Qt.UserRole)
                for item in self._browse_list.selectedItems()]
        self._add_units(uids)

    def _remove_tracked(self):
        selected_uids = {item.data(Qt.UserRole)
                         for item in self._tracked_list.selectedItems()}
        self._tracked_units = [u for u in self._tracked_units
                                if u not in selected_uids]
        self._rebuild_tracked_list()
        self._rebuild_fr_plots()
        self._load_fr_data()
        self._update_fr_at_frame(self._center_nf)

    def _clear_tracked(self):
        self._tracked_units.clear()
        self._rebuild_tracked_list()
        self._rebuild_fr_plots()

    def _rebuild_tracked_list(self):
        self._tracked_list.clear()
        for uid in self._tracked_units:
            color = self._unit_colors.get(uid, '#ffffff')
            ct    = self._get_cell_type(uid)
            item  = QListWidgetItem(f"■  U{uid}  {ct}")
            item.setData(Qt.UserRole, uid)
            item.setForeground(QColor(color))
            self._tracked_list.addItem(item)

    # ─────────────────────────────────────────────────────────────────────────
    # Per-cell FR plots
    # ─────────────────────────────────────────────────────────────────────────

    def _rebuild_fr_plots(self):
        self._fr_gw.clear()
        self._fr_plots.clear()
        self._fr_curves.clear()
        self._center_lines.clear()

        n = len(self._tracked_units)
        for i, uid in enumerate(self._tracked_units):
            color = self._unit_colors.get(uid, _CELL_COLORS[i % len(_CELL_COLORS)])
            r, g, b = _hex_to_rgb(color)

            pi = self._fr_gw.addPlot(row=i, col=0)
            pi.getViewBox().setBackgroundColor(BG_COLOR)
            pi.showGrid(x=True, y=False, alpha=0.15)
            pi.getAxis('left').setPen(FG_COLOR)
            pi.getAxis('left').setTextPen(color)
            pi.getAxis('left').setStyle(tickFont=QFont('Arial', 8))
            pi.getAxis('bottom').setPen(FG_COLOR)
            pi.getAxis('bottom').setTextPen(FG_COLOR)
            pi.getAxis('bottom').setStyle(tickFont=QFont('Arial', 8))
            pi.setLabel('left', f'U{uid}', color=color, size='8pt')
            if i == n - 1:
                pi.setLabel('bottom', 'Frame / Time', color=FG_COLOR)
            else:
                pi.hideAxis('bottom')
            if i > 0:
                pi.setXLink(self._fr_plots[0])

            curve = pg.PlotCurveItem(
                pen=pg.mkPen(color, width=1.5),
                fillLevel=0,
                brush=pg.mkBrush(r, g, b, 70),
            )
            pi.addItem(curve)

            cl = pg.InfiniteLine(
                angle=90, movable=False,
                pen=pg.mkPen('#ffffff', width=1, style=Qt.DashLine))
            pi.addItem(cl)

            self._fr_plots.append(pi)
            self._fr_curves.append(curve)
            self._center_lines.append(cl)

    def _load_fr_data(self):
        """Load (optionally Gaussian-smoothed) full FR into each curve."""
        if not self._fr_curves:
            return

        if self._sd.T_bin is not None and len(self._sd.T_bin) >= self._n_neural:
            x_ax = self._sd.T_bin[:self._n_neural]
        else:
            x_ax = np.arange(self._n_neural, dtype=float)

        for uid, curve in zip(self._tracked_units, self._fr_curves):
            fr = self._sd.fr[uid, :self._n_neural].astype(float)
            if self._smooth_sigma > 0:
                fr = gaussian_filter1d(fr, self._smooth_sigma)
            curve.setData(x_ax, fr)

    # ─────────────────────────────────────────────────────────────────────────
    # Event combo
    # ─────────────────────────────────────────────────────────────────────────

    def _populate_event_combo(self):
        self._ev_combo.clear()
        for key in self._sd.event_keys:
            self._ev_combo.addItem(self._sd.get_event_label(key), userData=key)

    def _build_event_video_frames(self):
        """Direct copy: trigger frame index = video frame index."""
        self._event_vframes.clear()
        for key in self._sd.event_keys:
            frames = self._sd._event_frames.get(key)
            if frames is not None and len(frames) > 0:
                arr = np.clip(
                    np.asarray(frames, dtype=int) - self._frame_offset,
                    0, self._n_neural - 1)
                self._event_vframes[key] = arr

    # ─────────────────────────────────────────────────────────────────────────
    # Video loading
    # ─────────────────────────────────────────────────────────────────────────

    def _load_video_dialog(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open video file", "",
            "Video files (*.mp4 *.avi *.mkv *.mov *.wmv);;All files (*)")
        if path:
            self.load_video(path)

    def load_video(self, path: str) -> bool:
        if not _HAVE_CV2:
            self._vid_lbl.setText("opencv-python not installed")
            return False
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            self._vid_lbl.setText(f"Could not open: {path}")
            return False
        if self._cap is not None:
            self._cap.release()
        self._cap           = cap
        self._video_fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self._total_vframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        warn = ''
        if self._total_vframes != self._n_neural:
            warn = f"  ⚠ video={self._total_vframes} neural={self._n_neural}"
            print(f"[VideoViewer] frame count mismatch – use Offset to align")

        self._vid_lbl.setText(
            f"{path.split('/')[-1].split(chr(92))[-1]}  "
            f"({self._total_vframes} fr @ {self._video_fps:.1f} fps){warn}")
        self._go_to_frame(self._center_nf)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Navigation
    # ─────────────────────────────────────────────────────────────────────────

    def _go_to_frame(self, nf: int, force_recenter: bool = False):
        nf = max(0, min(self._n_neural - 1, nf))
        self._center_nf = nf

        self._slider.blockSignals(True)
        self._frame_sb.blockSignals(True)
        self._slider.setValue(nf)
        self._frame_sb.setValue(nf)
        self._slider.blockSignals(False)
        self._frame_sb.blockSignals(False)

        self._update_frame_strip()
        self._update_fr_at_frame(nf, force_recenter=force_recenter)

    def _prev_frame(self):    self._go_to_frame(self._center_nf - 1)
    def _next_frame(self):    self._go_to_frame(self._center_nf + 1)
    def _on_slider(self, v):  self._go_to_frame(v)

    def _on_offset_changed(self, value: int):
        self._frame_offset = value
        self._build_event_video_frames()
        self._go_to_frame(self._center_nf)

    def _on_nstrip_changed(self, value: int):
        self._n_strip = value
        self._update_frame_strip()

    def _on_frwin_changed(self, value: int):
        self._fr_window = value
        self._update_fr_at_frame(self._center_nf, force_recenter=True)

    def _on_smooth_changed(self, value: int):
        self._smooth_sigma = value
        self._load_fr_data()

    def _on_frame_h_changed(self, value: int):
        """Resize all strip frame labels and the strip container live."""
        self._frame_h = value
        self._strip_w.setFixedHeight(value + 4)
        for lbl in self._frame_labels:
            lbl.setFixedHeight(value)
        self._update_frame_strip()

    def _on_strip_click(self, strip_idx: int):
        """Click a frame label → show that frame full-res in the zoom window."""
        if self._cap is None or not _HAVE_CV2:
            return
        half = self._n_strip // 2
        vf   = self._center_nf - half + strip_idx
        if not (0 <= vf < self._total_vframes):
            return
        self._seek_to_frame(vf)
        ret, frame = self._cap.read()
        if not ret:
            return
        if self._zoom_win is None:
            self._zoom_win = FrameZoomDialog(self)
        self._zoom_win.show_frame(frame, vf)

    def _jump_event(self, direction: int):
        key = self._ev_combo.currentData()
        if key is None:
            return
        vframes = self._event_vframes.get(key)
        if vframes is None or len(vframes) == 0:
            return
        cv = self._center_nf
        if direction > 0:
            cands  = vframes[vframes > cv]
            target = int(cands[0]) if len(cands) else int(vframes[-1])
        else:
            cands  = vframes[vframes < cv]
            target = int(cands[-1]) if len(cands) else int(vframes[0])
        self._go_to_frame(target, force_recenter=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Frame strip
    # ─────────────────────────────────────────────────────────────────────────

    def _update_frame_strip(self):
        half     = self._n_strip // 2
        vf_start = self._center_nf - half
        raw      = self._read_strip_frames(vf_start, self._n_strip)

        for i, lbl in enumerate(self._frame_labels):
            if i < self._n_strip:
                lbl.setVisible(True)
                vf        = vf_start + i
                is_center = (vf == self._center_nf)
                frame     = raw[i]
                if frame is not None:
                    lbl.setPixmap(self._frame_to_pixmap(frame))
                    lbl.setText('')
                else:
                    lbl.setPixmap(QPixmap())
                    lbl.setText('–')
                bc = '#4dd0e1' if is_center else '#1a1a3e'
                bw = 3        if is_center else 1
                lbl.setStyleSheet(
                    f"background: #111122; color: #445566; font-size: 8pt; "
                    f"border: {bw}px solid {bc};")
            else:
                lbl.setVisible(False)

    def _seek_to_frame(self, target_vf: int):
        """MSEC seek + grab-forward for exact H.264 frame access."""
        import cv2
        lb       = int(self._video_fps)
        start_vf = max(0, target_vf - lb)
        self._cap.set(cv2.CAP_PROP_POS_MSEC, start_vf / self._video_fps * 1000.0)
        landed = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        if landed < 0:
            landed = int(self._cap.get(cv2.CAP_PROP_POS_MSEC)
                         / 1000.0 * self._video_fps)
        landed = max(0, min(landed, target_vf))
        for _ in range(target_vf - landed + lb):
            if landed >= target_vf:
                break
            if not self._cap.grab():
                break
            landed += 1

    def _read_strip_frames(self, vf_start: int, n: int) -> list:
        if self._cap is None or not _HAVE_CV2:
            return [None] * n
        first = next(
            (vf_start + i for i in range(n)
             if 0 <= vf_start + i < self._total_vframes), None)
        if first is None:
            return [None] * n
        self._seek_to_frame(first)
        frames, pos = [], first
        for i in range(n):
            vf = vf_start + i
            if vf < 0 or vf >= self._total_vframes:
                frames.append(None)
                continue
            if vf != pos:
                self._seek_to_frame(vf)
                pos = vf
            ret, frame = self._cap.read()
            frames.append(frame if ret else None)
            pos += 1
        return frames

    def _frame_to_pixmap(self, frame: np.ndarray) -> QPixmap:
        import cv2
        h, w  = frame.shape[:2]
        fh    = self._frame_h
        w_new = max(1, int(w * fh / h))
        rgb   = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (w_new, fh))
        qimg  = QImage(rgb.data, w_new, fh, w_new * 3, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    # ─────────────────────────────────────────────────────────────────────────
    # FR traces
    # ─────────────────────────────────────────────────────────────────────────

    def _update_fr_at_frame(self, center_nf: int, force_recenter: bool = False):
        """Move center lines; re-centre x-range only when needed or forced."""
        if not self._fr_plots:
            return

        # x coordinate of the center frame
        if self._sd.T_bin is not None and center_nf < len(self._sd.T_bin):
            x_c = float(self._sd.T_bin[center_nf])
        else:
            x_c = float(center_nf)

        for cl in self._center_lines:
            cl.setValue(x_c)

        # Decide whether to update x-range:
        # – always on force_recenter (jump to event, window resize)
        # – otherwise only if center is outside the current view
        half     = self._fr_window // 2
        nf_start = max(0, center_nf - half)
        nf_end   = min(self._n_neural - 1, center_nf + half)

        if self._sd.T_bin is not None:
            x_lo = float(self._sd.T_bin[nf_start])
            x_hi = float(self._sd.T_bin[nf_end])
        else:
            x_lo, x_hi = float(nf_start), float(nf_end)

        if self._fr_plots and x_hi > x_lo:
            vr     = self._fr_plots[0].viewRange()[0]
            in_view = vr[0] <= x_c <= vr[1]
            if force_recenter or not in_view:
                self._fr_plots[0].setXRange(x_lo, x_hi, padding=0.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Key navigation
    # ─────────────────────────────────────────────────────────────────────────

    def keyPressEvent(self, event):
        k = event.key()
        if k in (Qt.Key_Left,  Qt.Key_A):     self._prev_frame()
        elif k in (Qt.Key_Right, Qt.Key_D):   self._next_frame()
        elif k == Qt.Key_Comma:               self._jump_event(-1)
        elif k == Qt.Key_Period:              self._jump_event(+1)
        else:                                 super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    # Cleanup
    # ─────────────────────────────────────────────────────────────────────────

    def closeEvent(self, event):
        if self._cap is not None:
            self._cap.release()
        super().closeEvent(event)
