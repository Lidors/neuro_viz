"""
app_window.py  –  Main application window (dark mode).

Keyboard shortcuts
------------------
←  / →    : previous / next unit
1 / 2 / 3 : Heatmap / Blocks / Mean PSTH view
X         : Export selected cell(s) — one window per cell
P         : Open CS/SS pair of current unit in a new window
V         : Video + Neural viewer
C         : Python console
Ctrl+L    : Claude AI chat
"""

import os

from PyQt5.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
    QAction, QFileDialog, QMessageBox, QStatusBar, QLabel, QTabBar,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence, QColor

from unit_table    import UnitTableWidget
from psth_panel    import PSTHPanel
from control_panel import ControlPanel
from wf_acg_panel  import WFACGPanel

DARK_MAIN = (
    "QMainWindow, QSplitter { background: #0d0d2a; }"
    "QMenuBar { background: #111128; color: #ccccee; }"
    "QMenuBar::item:selected { background: #2a2a6a; }"
    "QMenu { background: #111128; color: #ccccee; border: 1px solid #3a3a6a; }"
    "QMenu::item:selected { background: #2a2a6a; }"
    "QStatusBar { background: #111128; color: #8888aa; font-size: 8pt; }"
    "QSplitter::handle { background: #2a2a5a; width: 3px; }"
)

_TABBAR_STYLE = (
    "QTabBar { background: #0d0d2a; }"
    "QTabBar::tab {"
    "  background: #111128; color: #6666aa;"
    "  padding: 3px 16px 3px 12px;"
    "  border: 1px solid #2a2a5a; border-bottom: none;"
    "  border-radius: 3px 3px 0 0; margin-right: 2px; font-size: 9pt; }"
    "QTabBar::tab:selected { background: #1a1a4a; color: #ccccff; }"
    "QTabBar::tab:hover    { background: #161640; color: #aaaaee; }"
)

# Colors cycled for exports from the main window
_EXPORT_COLORS = [
    '#2196F3', '#F44336', '#4CAF50', '#FF9800',
    '#9C27B0', '#00BCD4', '#FF5722', '#8BC34A',
]


class MainWindow(QMainWindow):
    def __init__(self, data):
        super().__init__()
        self.data          = data
        self._unit_idx     = 0
        self.setStyleSheet(DARK_MAIN)

        # Session management
        name = os.path.splitext(os.path.basename(
            getattr(data, '_res_path', 'Session')))[0]
        self._sessions: list[tuple[str, object]] = [(name, data)]
        self._cur_sess_idx: int = 0

        # Dialog tracking (closed on session switch)
        self._open_dialogs: list         = []
        self._cell_compare_wins: list    = []   # all open cell compare windows
        self._active_cell_win            = None  # most recently used window
        self._video_win                  = None
        self._console_win                = None
        self._claude_win                 = None
        self._export_color_idx: int      = 0

        self._build_ui()
        self._connect_signals()

        # Debounce rapid unit changes (arrow keys held down)
        self._pending_unit = 0
        self._load_timer = QTimer(self)
        self._load_timer.setSingleShot(True)
        self._load_timer.setInterval(80)   # ms — coalesces rapid key presses
        self._load_timer.timeout.connect(self._do_load_unit)

        self._load_unit(0)

    # ── Layout ────────────────────────────────────────────────────────────────
    def _build_ui(self):
        self.setWindowTitle("NeuroPy Viewer")
        self.resize(1440, 880)

        # ── Central widget: tab bar (top) + content (below) ───────────────────
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Session tab bar row
        tab_row = QHBoxLayout()
        tab_row.setContentsMargins(4, 2, 4, 0)
        tab_row.setSpacing(4)

        self._sess_tabs = QTabBar()
        self._sess_tabs.setStyleSheet(_TABBAR_STYLE)
        self._sess_tabs.setTabsClosable(True)
        self._sess_tabs.setMovable(False)
        self._sess_tabs.setExpanding(False)
        self._sess_tabs.addTab(self._sessions[0][0])   # first session tab
        self._sess_tabs.currentChanged.connect(self._switch_session)
        self._sess_tabs.tabCloseRequested.connect(self._close_session_tab)
        tab_row.addWidget(self._sess_tabs)

        add_tab_btn = QPushButton("+")
        add_tab_btn.setFixedSize(24, 22)
        add_tab_btn.setToolTip("Add session")
        add_tab_btn.setStyleSheet(
            "QPushButton { background: #1a1a3e; color: #8888cc; "
            "border: 1px solid #3a3a6a; border-radius: 3px; font-size: 12pt; "
            "padding: 0; line-height: 20px; }"
            "QPushButton:hover { background: #2a2a6a; color: #ccccff; }"
        )
        add_tab_btn.clicked.connect(self._add_session_tab)
        tab_row.addWidget(add_tab_btn)
        tab_row.addStretch()
        root.addLayout(tab_row)

        # Thin separator line below tabs
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: #2a2a5a;")
        root.addWidget(sep)

        # ── Main content: left | center | right ───────────────────────────────
        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Left column: unit table + action buttons + WF/ACG
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setMinimumWidth(200)
        left_splitter.setMaximumWidth(340)

        self._unit_table = UnitTableWidget(self.data)

        # Action buttons
        btn_widget = QWidget()
        btn_layout = QVBoxLayout(btn_widget)
        btn_layout.setContentsMargins(4, 2, 4, 2)
        btn_layout.setSpacing(2)

        _cell_btn_style = (
            "QPushButton { background: #1a3a3a; color: #4dd0e1; "
            "border: 1px solid #2a7a8a; border-radius: 3px; "
            "font-size: 8pt; padding: 2px 6px; }"
            "QPushButton:hover { background: #1e4a4a; }"
        )

        self._export_cell_btn = QPushButton("Export cell(s) →")
        self._export_cell_btn.setFixedHeight(26)
        self._export_cell_btn.setToolTip(
            "Add selected unit(s) to the latest Cell Compare window  (X)")
        self._export_cell_btn.setStyleSheet(_cell_btn_style)
        self._export_cell_btn.clicked.connect(lambda: self._export_cell(new_window=False))

        self._new_win_btn = QPushButton("New window ↗")
        self._new_win_btn.setFixedHeight(26)
        self._new_win_btn.setToolTip(
            "Open a fresh Cell Compare window and export to it  (N)")
        self._new_win_btn.setStyleSheet(_cell_btn_style)
        self._new_win_btn.clicked.connect(lambda: self._export_cell(new_window=True))

        cell_row = QHBoxLayout()
        cell_row.setSpacing(3)
        cell_row.addWidget(self._export_cell_btn, 3)
        cell_row.addWidget(self._new_win_btn, 2)
        btn_layout.addLayout(cell_row)

        self._wf_acg = WFACGPanel(self.data)

        left_splitter.addWidget(self._unit_table)
        left_splitter.addWidget(btn_widget)
        left_splitter.addWidget(self._wf_acg)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 0)
        left_splitter.setStretchFactor(2, 0)
        splitter.addWidget(left_splitter)

        # Center: PSTH panel
        self._psth_panel = PSTHPanel(self.data)
        splitter.addWidget(self._psth_panel)

        # Right: controls
        self._controls = ControlPanel()
        splitter.addWidget(self._controls)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        self._build_menu()
        self.setStatusBar(QStatusBar())

    def _build_menu(self):
        mb = self.menuBar()

        # File
        fm = mb.addMenu("&File")
        add_sess_act = QAction("&Add session…", self)
        add_sess_act.setShortcut(QKeySequence("Ctrl+O"))
        add_sess_act.triggered.connect(self._add_session_tab)
        fm.addAction(add_sess_act)

        add_ev_act = QAction("&Add event file…", self)
        add_ev_act.triggered.connect(self._add_event_file)
        fm.addAction(add_ev_act)

        fm.addSeparator()
        pdf_act = QAction("Save current tab as &PDF…", self)
        pdf_act.setShortcut(QKeySequence("Ctrl+P"))
        pdf_act.triggered.connect(lambda: self._psth_panel.save_pdf())
        fm.addAction(pdf_act)

        fm.addSeparator()
        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence("Ctrl+Q"))
        quit_act.triggered.connect(self.close)
        fm.addAction(quit_act)

        # View
        vm = mb.addMenu("&View")
        for i, lbl in enumerate(['Heatmap', 'Block PSTH', 'Mean PSTH']):
            act = QAction(f"&{i+1}  {lbl}", self)
            act.setShortcut(str(i + 1))
            act.triggered.connect(lambda _, x=i: self._set_view(x))
            vm.addAction(act)
        vm.addSeparator()

        exp_cell_act = QAction("Export cell(s) → active window", self)
        exp_cell_act.setShortcut("X")
        exp_cell_act.triggered.connect(lambda: self._export_cell(new_window=False))
        vm.addAction(exp_cell_act)

        exp_new_act = QAction("Export cell(s) → new window", self)
        exp_new_act.setShortcut("N")
        exp_new_act.triggered.connect(lambda: self._export_cell(new_window=True))
        vm.addAction(exp_new_act)

        pair_act = QAction("Open CS/SS pair → new window", self)
        pair_act.setShortcut("P")
        pair_act.triggered.connect(self._open_current_pair)
        vm.addAction(pair_act)

        vid_act = QAction("Video + Neural viewer…", self)
        vid_act.setShortcut("V")
        vid_act.triggered.connect(self._open_video_viewer)
        vm.addAction(vid_act)

        vm.addSeparator()
        console_act = QAction("Python Console…", self)
        console_act.setShortcut("C")
        console_act.triggered.connect(self._open_console)
        vm.addAction(console_act)

        claude_act = QAction("Claude AI Chat…", self)
        claude_act.setShortcut(QKeySequence("Ctrl+L"))
        claude_act.triggered.connect(self._open_claude_chat)
        vm.addAction(claude_act)

        # Export-to-console submenu
        exp_menu = vm.addMenu("Export to Console")
        exp_sd = QAction("SessionData → sd", self)
        exp_sd.triggered.connect(lambda: self._export_to_console('sd', self.data))
        exp_menu.addAction(exp_sd)
        exp_fr = QAction("Firing rates → fr", self)
        exp_fr.triggered.connect(
            lambda: self._export_to_console('fr', self.data.fr))
        exp_menu.addAction(exp_fr)
        exp_psths = QAction("Selected PSTHs → psths", self)
        exp_psths.triggered.connect(self._export_psths_to_console)
        exp_menu.addAction(exp_psths)

        # Help
        hm = mb.addMenu("&Help")
        kb_act = QAction("Keyboard shortcuts", self)
        kb_act.triggered.connect(self._show_shortcuts)
        hm.addAction(kb_act)

    # ── Signal wiring ────────────────────────────────────────────────────────
    def _connect_signals(self):
        self._unit_table.unit_selected.connect(self._load_unit)
        self._unit_table.units_selected.connect(self._psth_panel.set_units)
        self._unit_table.pair_compare_requested.connect(self._open_pair)

        c = self._controls
        c.xlim_changed.connect(self._psth_panel.set_xlim)
        c.clim_changed.connect(self._psth_panel.set_clim)
        c.smooth_changed.connect(self._psth_panel.set_smooth)
        c.block_size_changed.connect(self._psth_panel.set_block_size)
        c.colormap_changed.connect(self._psth_panel.set_colormap)
        c.block_line_cmap_changed.connect(self._psth_panel.set_block_line_cmap)
        c.add_event_clicked.connect(self._add_event_file)
        c.connect_nav(self._prev_unit, self._next_unit)

    # ── Session switching ─────────────────────────────────────────────────────

    def _switch_session(self, idx: int):
        """Switch to session at index idx, closing any tracked dialogs."""
        if idx < 0 or idx >= len(self._sessions):
            return
        self._cur_sess_idx = idx
        self.data          = self._sessions[idx][1]

        # Close all tracked (non-comparison) dialogs
        for dlg in list(self._open_dialogs):
            try:
                dlg.close()
            except RuntimeError:
                pass   # C++ object already deleted
        self._open_dialogs.clear()

        # Re-inject SessionData into console (if open)
        if self._console_win is not None and self._console_win.isVisible():
            self._console_win.inject('sd', self.data, quiet=True)

        # Reload panels
        self._unit_table.data = self.data
        self._unit_table._populate()
        self._wf_acg.data = self.data
        self._psth_panel.data = self.data
        self._psth_panel._populate_event_bar()
        self._load_unit(0)

    def _add_session_tab(self, path: str | None = None):
        """Load a new session and add it as a tab.  Opens file dialog if path not given."""
        if not path:
            path, _ = QFileDialog.getOpenFileName(
                self, "Add session (res.mat)", "",
                "MAT files (*.mat);;All files (*)"
            )
        if not path:
            return

        t_bin = os.path.join(os.path.dirname(path), 'T_bin.mat')
        try:
            from data_loader import SessionData
            new_data = SessionData(path, t_bin if os.path.exists(t_bin) else None)
        except Exception as e:
            QMessageBox.critical(self, "Load error", str(e))
            return

        name = os.path.splitext(os.path.basename(path))[0]
        self._sessions.append((name, new_data))

        # Block signals while adding tab to avoid premature _switch_session
        self._sess_tabs.blockSignals(True)
        new_idx = self._sess_tabs.addTab(name)
        self._sess_tabs.blockSignals(False)

        # Switch to the newly added tab
        self._sess_tabs.setCurrentIndex(new_idx)
        # setCurrentIndex fires currentChanged → _switch_session

    def _close_session_tab(self, idx: int):
        """Remove a session tab. At least one session must remain."""
        if len(self._sessions) <= 1:
            return

        self._sessions.pop(idx)

        # Remove tab without triggering an intermediate _switch_session
        self._sess_tabs.blockSignals(True)
        self._sess_tabs.removeTab(idx)
        # Clamp current index to valid range
        new_idx = min(self._cur_sess_idx, len(self._sessions) - 1)
        self._sess_tabs.setCurrentIndex(new_idx)
        self._sess_tabs.blockSignals(False)

        self._switch_session(new_idx)

    def _track_dialog(self, dlg):
        """Register a non-modal dialog to be closed on session switch."""
        self._open_dialogs.append(dlg)
        dlg.finished.connect(lambda _=None: self._untrack_dialog(dlg))

    def _untrack_dialog(self, dlg):
        try:
            self._open_dialogs.remove(dlg)
        except ValueError:
            pass

    # ── Unit loading ──────────────────────────────────────────────────────────
    def _load_unit(self, unit_idx: int):
        """Schedule a unit load; coalesces rapid calls (arrow key held down)."""
        self._pending_unit = unit_idx
        self._load_timer.start()   # resets the 80 ms countdown on each call

    def _do_load_unit(self):
        """Actually render the pending unit — called by the debounce timer."""
        unit_idx = self._pending_unit
        self._unit_idx = unit_idx
        self._psth_panel.set_unit(unit_idx)
        self._unit_table.select_unit(unit_idx)
        self._wf_acg.update_unit(unit_idx)
        depth = self.data.probe_depth[unit_idx] or self.data.probe_pos[unit_idx]
        self._controls.set_unit_label(
            unit_idx,
            self.data.cell_type_label[unit_idx],
            depth
        )
        sess_name = self._sessions[self._cur_sess_idx][0]
        self.statusBar().showMessage(
            f"[{sess_name}]  Unit {unit_idx}"
            f"  |  {self.data.cell_type_label[unit_idx]}"
            f"  |  depth {depth} µm"
            f"  |  {self.data.n_units} total units"
        )

    # ── Cell export ───────────────────────────────────────────────────────────

    def _export_cell(self, new_window: bool = False):
        """Export selected unit(s) to a Cell Compare window.

        new_window=False  → add to the latest active window (create one if needed)
        new_window=True   → always open a fresh window (it becomes the new active)
        """
        from cell_compare_window import CellCompareWindow, CellEntry

        sess_name = self._sessions[self._cur_sess_idx][0]

        # Gather units: multi-selection takes priority, else current unit
        sel = sorted(set(self._psth_panel._multi_units))
        units = sel if len(sel) > 1 else [self._unit_idx]

        # Determine target window
        if new_window or self._active_cell_win is None \
                or not self._active_cell_win.isVisible():
            win = CellCompareWindow(parent=None)
            win.show()
            self._cell_compare_wins.append(win)
            self._active_cell_win = win
        else:
            win = self._active_cell_win
            win.raise_()
            win.activateWindow()

        for unit_idx in units:
            color = _EXPORT_COLORS[self._export_color_idx % len(_EXPORT_COLORS)]
            self._export_color_idx += 1
            entry = CellEntry(
                label        = f"U{unit_idx}  [{sess_name}]",
                color        = color,
                unit_indices = [unit_idx],
                session_data = self.data,
                session_name = sess_name,
            )
            win.add_cell(entry)

    def _open_current_pair(self):
        """Open the CS/SS pair of the currently displayed unit (P key)."""
        pair_map = getattr(self.data, 'pair_map', {})
        partner = pair_map.get(self._unit_idx)
        if partner is None:
            self.statusBar().showMessage("Current unit has no CS/SS pair.", 3000)
            return
        self._open_pair([self._unit_idx, partner])

    def _open_pair(self, unit_pair: list):
        """Open both units of a CS/SS pair in a fresh Cell Compare window."""
        from cell_compare_window import CellCompareWindow, CellEntry
        sess_name = self._sessions[self._cur_sess_idx][0]
        win = CellCompareWindow(parent=None)
        win.show()
        self._cell_compare_wins.append(win)
        self._active_cell_win = win
        for unit_idx in unit_pair:
            color = _EXPORT_COLORS[self._export_color_idx % len(_EXPORT_COLORS)]
            self._export_color_idx += 1
            entry = CellEntry(
                label        = f"U{unit_idx}  [{sess_name}]",
                color        = color,
                unit_indices = [unit_idx],
                session_data = self.data,
                session_name = sess_name,
            )
            win.add_cell(entry)

    def _open_video_viewer(self):
        """Open (or raise) the Video + Neural viewer for the current session."""
        from video_viewer import VideoSyncWindow
        if self._video_win is None or not self._video_win.isVisible():
            self._video_win = VideoSyncWindow(
                session_data=self.data,
                current_unit=self._unit_idx,
                parent=None)
            self._video_win.show()
        else:
            self._video_win.raise_()
            self._video_win.activateWindow()

    # ── Console / Claude ─────────────────────────────────────────────────────

    def _open_console(self):
        """Open (or raise) the interactive Python console."""
        from console_widget import ConsoleWindow
        if self._console_win is None or not self._console_win.isVisible():
            self._console_win = ConsoleWindow(parent=None)
            self._console_win.inject('sd', self.data)
            self._console_win.show()
        else:
            self._console_win.raise_()
            self._console_win.activateWindow()

    def _open_claude_chat(self):
        """Open the Claude AI chat (creates console first if needed)."""
        self._open_console()
        from claude_chat import ClaudeChatWindow
        if self._claude_win is None or not self._claude_win.isVisible():
            self._claude_win = ClaudeChatWindow(self._console_win, parent=None)
            self._claude_win.show()
        else:
            self._claude_win.raise_()
            self._claude_win.activateWindow()

    def _export_to_console(self, name: str, value):
        """Inject a named variable into the console namespace."""
        self._open_console()
        self._console_win.inject(name, value)

    def _export_psths_to_console(self):
        """Export mean PSTHs for selected/current units as a dict."""
        self._open_console()
        sel = sorted(set(self._psth_panel._multi_units))
        units = sel if len(sel) > 1 else [self._unit_idx]
        psths = {}
        for u in units:
            for key in self.data.event_keys:
                mean, sem, tax = self.data.get_mean_psth(
                    u, key, self._controls._smooth.value())
                psths[(u, key)] = {'mean': mean, 'sem': sem, 'time': tax}
        self._console_win.inject('psths', psths)
        self._console_win.inject('psth_units', units, quiet=True)

    # ── Other actions ─────────────────────────────────────────────────────────

    def _prev_unit(self):
        if self._unit_idx > 0:
            self._load_unit(self._unit_idx - 1)

    def _next_unit(self):
        if self._unit_idx < self.data.n_units - 1:
            self._load_unit(self._unit_idx + 1)

    def _set_view(self, idx: int):
        btn = self._psth_panel._view_group.button(idx)
        if btn:
            btn.setChecked(True)
            self._psth_panel._stack.setCurrentIndex(idx)
            self._psth_panel.refresh()

    def _add_event_file(self):
        from event_dialog import AddEventDialog
        dlg = AddEventDialog(self)
        if dlg.exec_() != AddEventDialog.Accepted:
            return

        result = dlg.get_result()
        if not result:
            return

        added = []
        for key, info in result.items():
            self.data.add_event(
                key   = key,
                times = info['times'],
                units = info['units'],
                label = info['label'],
            )
            self._psth_panel.add_event_to_bar(key)
            added.append(info['label'])

        self.statusBar().showMessage(
            f"Added events: {', '.join(added)}", 4000
        )
        self._psth_panel.refresh()

    # ── Shortcuts dialog ──────────────────────────────────────────────────────
    def _show_shortcuts(self):
        QMessageBox.information(self, "Keyboard shortcuts",
            "<b>Navigation</b><br>"
            "&nbsp; ←  /  →   &nbsp; previous / next unit<br><br>"
            "<b>View</b><br>"
            "&nbsp; <b>1</b>  &nbsp; Heatmap<br>"
            "&nbsp; <b>2</b>  &nbsp; Block PSTH<br>"
            "&nbsp; <b>3</b>  &nbsp; Mean PSTH<br>"
            "&nbsp; <b>X</b>  &nbsp; Export selected cell(s) → active window<br>"
            "&nbsp; <b>N</b>  &nbsp; Export selected cell(s) → new window<br>"
            "&nbsp; <b>P</b>  &nbsp; Open CS/SS pair of current unit → new window<br>"
            "&nbsp; <b>V</b>  &nbsp; Video + Neural viewer<br>"
            "&nbsp; <b>C</b>  &nbsp; Python console<br>"
            "&nbsp; <b>Ctrl+L</b>  &nbsp; Claude AI chat<br>"
            "&nbsp; <b>D</b>  &nbsp; Toggle diff mode (Mean PSTH)<br><br>"
            "<b>Sessions</b><br>"
            "&nbsp; Click tab to switch session<br>"
            "&nbsp; Click + to add a session<br>"
            "&nbsp; Click × on a tab to close a session<br><br>"
            "<b>File</b><br>"
            "&nbsp; Ctrl+O &nbsp; Add session<br>"
            "&nbsp; Ctrl+P &nbsp; Save current tab as PDF<br>"
            "&nbsp; Ctrl+Q &nbsp; Quit<br>"
        )

    # ── Key events ────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        if   key == Qt.Key_Left:  self._prev_unit()
        elif key == Qt.Key_Right: self._next_unit()
        elif key == Qt.Key_1:     self._set_view(0)
        elif key == Qt.Key_2:     self._set_view(1)
        elif key == Qt.Key_3:     self._set_view(2)
        elif key == Qt.Key_X:     self._export_cell(new_window=False)
        elif key == Qt.Key_N:     self._export_cell(new_window=True)
        elif key == Qt.Key_V:     self._open_video_viewer()
        elif key == Qt.Key_P:     self._open_current_pair()
        elif key == Qt.Key_C:     self._open_console()
        elif key == Qt.Key_D:
            if self._psth_panel._stack.currentIndex() != 2:
                self._set_view(2)
            self._psth_panel._mean_view.toggle_diff()
        else:                     super().keyPressEvent(event)
