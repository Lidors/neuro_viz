"""
console_widget.py  --  Interactive Python console with shared namespace.

Opens as a floating window.  Variables can be injected from the GUI via
``inject(name, value)`` and results pushed back via helper functions
available in the namespace (``create_group``, ``add_event``, ``show_plot``).
"""
from __future__ import annotations

import io
import re
import sys
import traceback

import numpy as np
import pyqtgraph as pg
from PyQt5.QtCore import Qt, pyqtSignal, QRegExp
from PyQt5.QtGui import (
    QColor, QFont, QTextCharFormat, QSyntaxHighlighter,
    QPixmap, QTextCursor,
)
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QPlainTextEdit, QTextEdit, QPushButton, QLabel,
    QApplication, QFileDialog,
)

from population_viewer import BG_COLOR, FG_COLOR

# ── Dark-theme styles ─────────────────────────────────────────────────────────

_MONO_FONT = QFont('Consolas', 10)

_EDITOR_STYLE = (
    f"QPlainTextEdit {{ background: #0a0e1a; color: #dcdcdc; "
    f"border: 1px solid #3a3a6a; selection-background-color: #2a4a7a; }}"
)
_OUTPUT_STYLE = (
    f"QTextEdit {{ background: #0d0d1e; color: {FG_COLOR}; "
    f"border: 1px solid #3a3a6a; selection-background-color: #2a4a7a; }}"
)
_BTN_STYLE = (
    "QPushButton { background: #1a1a3e; color: #e0e0e0; "
    "border: 1px solid #3a3a6a; border-radius: 3px; "
    "font-size: 9pt; padding: 3px 10px; }"
    "QPushButton:hover { background: #2a2a6a; }"
    "QPushButton:pressed { background: #3a5a9a; }"
)


# ── Syntax highlighter ────────────────────────────────────────────────────────

def _fmt(color: str, bold: bool = False, italic: bool = False) -> QTextCharFormat:
    f = QTextCharFormat()
    f.setForeground(QColor(color))
    if bold:
        f.setFontWeight(QFont.Bold)
    if italic:
        f.setFontItalic(True)
    return f


class PythonHighlighter(QSyntaxHighlighter):
    """Minimal Python syntax highlighter for dark themes."""

    _KEYWORDS = (
        'False None True and as assert async await break class continue '
        'def del elif else except finally for from global if import in is '
        'lambda nonlocal not or pass raise return try while with yield'
    ).split()

    _BUILTINS = (
        'abs all any bin bool bytes callable chr dict dir divmod enumerate '
        'filter float format frozenset getattr globals hasattr hash hex id '
        'input int isinstance issubclass iter len list map max min next '
        'object oct open ord pow print range repr reversed round set '
        'setattr slice sorted staticmethod str sum super tuple type vars zip'
    ).split()

    def __init__(self, parent=None):
        super().__init__(parent)
        kw_fmt  = _fmt('#cc99ff', bold=True)
        bi_fmt  = _fmt('#66ccff')
        num_fmt = _fmt('#ffcc66')
        str_fmt = _fmt('#99cc99')
        cmt_fmt = _fmt('#666688', italic=True)
        self_fmt = _fmt('#ef5350')

        self._rules: list[tuple[re.Pattern, QTextCharFormat]] = []
        # Keywords
        for w in self._KEYWORDS:
            self._rules.append((re.compile(rf'\b{w}\b'), kw_fmt))
        # Builtins
        for w in self._BUILTINS:
            self._rules.append((re.compile(rf'\b{w}\b'), bi_fmt))
        # self
        self._rules.append((re.compile(r'\bself\b'), self_fmt))
        # Numbers
        self._rules.append((re.compile(r'\b[0-9]+\.?[0-9]*([eE][+-]?[0-9]+)?\b'), num_fmt))
        # Single-line strings (double then single quotes)
        self._rules.append((re.compile(r'"[^"\\]*(\\.[^"\\]*)*"'), str_fmt))
        self._rules.append((re.compile(r"'[^'\\]*(\\.[^'\\]*)*'"), str_fmt))
        # Comments
        self._rules.append((re.compile(r'#[^\n]*'), cmt_fmt))

    def highlightBlock(self, text: str):
        for pattern, fmt in self._rules:
            for m in pattern.finditer(text):
                self.setFormat(m.start(), m.end() - m.start(), fmt)


# ── Code editor ───────────────────────────────────────────────────────────────

class CodeEditor(QPlainTextEdit):
    """Multi-line code input with Ctrl+Enter execution and history."""

    execute_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(_MONO_FONT)
        self.setStyleSheet(_EDITOR_STYLE)
        self.setTabStopDistance(32)
        self.setPlaceholderText(
            "# Type Python code here.  Ctrl+Enter to run.\n"
            "# Up/Down arrows to browse command history.")
        PythonHighlighter(self.document())

        self._history: list[str] = []
        self._hist_idx: int = -1       # -1 = not browsing
        self._draft: str = ''          # stash current text while browsing

    # ── Public: called by ConsoleWindow after each run ────────────────────
    def push_history(self, code: str):
        """Append executed code to history."""
        if code and (not self._history or self._history[-1] != code):
            self._history.append(code)
        self._hist_idx = -1

    def keyPressEvent(self, event):
        key = event.key()
        mod = event.modifiers()
        # Ctrl+Enter / Ctrl+Return → execute
        if mod & Qt.ControlModifier and key in (Qt.Key_Return, Qt.Key_Enter):
            self.execute_requested.emit()
            return
        # Up / Down arrow → history navigation (only when no modifier)
        if key == Qt.Key_Up and mod == Qt.NoModifier:
            self._history_up()
            return
        if key == Qt.Key_Down and mod == Qt.NoModifier:
            self._history_down()
            return
        # Tab → 4 spaces
        if key == Qt.Key_Tab and not (mod & Qt.ShiftModifier):
            self.insertPlainText('    ')
            return
        # Enter → auto-indent
        if key in (Qt.Key_Return, Qt.Key_Enter) and not (mod & Qt.ControlModifier):
            cursor = self.textCursor()
            line = cursor.block().text()
            indent = len(line) - len(line.lstrip())
            # Add extra indent after colon
            if line.rstrip().endswith(':'):
                indent += 4
            super().keyPressEvent(event)
            if indent > 0:
                self.insertPlainText(' ' * indent)
            return
        super().keyPressEvent(event)

    # ── History helpers ───────────────────────────────────────────────────
    def _history_up(self):
        if not self._history:
            return
        if self._hist_idx == -1:
            self._draft = self.toPlainText()
            self._hist_idx = len(self._history) - 1
        elif self._hist_idx > 0:
            self._hist_idx -= 1
        else:
            return
        self.setPlainText(self._history[self._hist_idx])
        self._move_cursor_to_end()

    def _history_down(self):
        if self._hist_idx == -1:
            return
        if self._hist_idx < len(self._history) - 1:
            self._hist_idx += 1
            self.setPlainText(self._history[self._hist_idx])
        else:
            self._hist_idx = -1
            self.setPlainText(self._draft)
        self._move_cursor_to_end()

    def _move_cursor_to_end(self):
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.setTextCursor(cursor)


# ── Output area ───────────────────────────────────────────────────────────────

class OutputArea(QTextEdit):
    """Read-only scrollable output with inline image support."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(_MONO_FONT)
        self.setStyleSheet(_OUTPUT_STYLE)
        self.setAcceptRichText(True)

    def _scroll_to_bottom(self):
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())

    def append_text(self, text: str, color: str = FG_COLOR):
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        escaped = escaped.replace('\n', '<br>')
        self.append(f"<span style='color:{color}; white-space:pre;'>{escaped}</span>")
        self._scroll_to_bottom()

    def append_error(self, text: str):
        self.append_text(text, color='#ff6666')

    def append_image(self, pixmap: QPixmap):
        """Insert a QPixmap inline in the output."""
        cursor = self.textCursor()
        cursor.movePosition(QTextCursor.End)
        # Scale to fit (max 700 px wide)
        if pixmap.width() > 700:
            pixmap = pixmap.scaledToWidth(700, Qt.SmoothTransformation)
        doc = self.document()
        name = f'fig_{id(pixmap)}'
        doc.addResource(doc.ImageResource, pg.QtCore.QUrl(name), pixmap)
        cursor.insertImage(name)
        cursor.insertText('\n')
        self._scroll_to_bottom()

    def append_html(self, html: str):
        self.append(html)
        self._scroll_to_bottom()


# ── Console window ────────────────────────────────────────────────────────────

class ConsoleWindow(QDialog):
    """Interactive Python console with a shared namespace.

    The namespace persists across runs.  Variables injected via ``inject()``
    and helper functions (``show_plot``, ``create_group``, ``add_event``)
    are available in every execution.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Python Console")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(860, 640)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._namespace: dict = {}

        self._setup_namespace()
        self._build_ui()

        self._output.append_text(
            "Python console ready.  Type code below, Ctrl+Enter to run.\n"
            "Up/Down arrows to browse command history.\n"
            "Injected: np, plt, stats.\n"
            "Helpers: show_plot(), save_fig([fig, path, dpi]), create_group(), add_event()\n",
            color='#66aa88')

    # ── Namespace setup ───────────────────────────────────────────────────────

    def _setup_namespace(self):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from scipy import stats

        self._namespace.update({
            'np': np,
            'plt': plt,
            'stats': stats,
            'show_plot': self._show_plot,
            'save_fig': self._save_fig,
            'create_group': self._create_group,
            'add_event': self._add_event,
        })

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        splitter = QSplitter(Qt.Vertical)
        splitter.setStyleSheet(
            "QSplitter::handle { background: #2a2a5a; height: 3px; }")

        self._output = OutputArea()
        self._editor = CodeEditor()
        self._editor.execute_requested.connect(self._run_code)

        splitter.addWidget(self._output)
        splitter.addWidget(self._editor)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)
        outer.addWidget(splitter, stretch=1)

        # Button row
        btn_row = QHBoxLayout()
        btn_row.setSpacing(6)

        run_btn = QPushButton("Run  (Ctrl+Enter)")
        run_btn.setStyleSheet(_BTN_STYLE)
        run_btn.clicked.connect(self._run_code)
        btn_row.addWidget(run_btn)

        clear_btn = QPushButton("Clear output")
        clear_btn.setStyleSheet(_BTN_STYLE)
        clear_btn.clicked.connect(self._output.clear)
        btn_row.addWidget(clear_btn)

        save_fig_btn = QPushButton("Save figure…")
        save_fig_btn.setStyleSheet(_BTN_STYLE)
        save_fig_btn.setToolTip("Save the last matplotlib figure to a file")
        save_fig_btn.clicked.connect(self._save_fig_dialog)
        btn_row.addWidget(save_fig_btn)

        btn_row.addStretch()

        ns_btn = QPushButton("Show namespace")
        ns_btn.setStyleSheet(_BTN_STYLE)
        ns_btn.setToolTip("List all variables in the console namespace")
        ns_btn.clicked.connect(self._show_namespace)
        btn_row.addWidget(ns_btn)

        outer.addLayout(btn_row)

    # ── Execution ─────────────────────────────────────────────────────────────

    def _run_code(self):
        """Execute the editor contents in the shared namespace."""
        code = self._editor.toPlainText().strip()
        if not code:
            return

        self._editor.push_history(code)
        # Echo the code
        for line in code.splitlines():
            self._output.append_text(f">>> {line}", color='#8888cc')

        import matplotlib.pyplot as plt
        plt.close('all')

        old_stdout, old_stderr = sys.stdout, sys.stderr
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            # Try eval first (single expression → show return value)
            try:
                result = eval(compile(code, '<console>', 'eval'),
                              self._namespace)
                captured = buf.getvalue()
                if captured:
                    self._output.append_text(captured)
                if result is not None:
                    self._output.append_text(repr(result), color='#aaddaa')
            except SyntaxError:
                # Fallback: exec (statements)
                exec(compile(code, '<console>', 'exec'), self._namespace)
                captured = buf.getvalue()
                if captured:
                    self._output.append_text(captured)
        except Exception:
            self._output.append_error(traceback.format_exc())
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            QApplication.restoreOverrideCursor()

        # Auto-capture matplotlib figures
        if plt.get_fignums():
            self._capture_figures()

        self._editor.clear()

    def _capture_figures(self):
        """Capture all open matplotlib figures as inline images."""
        import matplotlib.pyplot as plt
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=120,
                        facecolor='#0d0d2a', edgecolor='none',
                        bbox_inches='tight')
            buf.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(buf.read())
            self._output.append_image(pixmap)
        plt.close('all')

    # ── Public: inject variables from the GUI ─────────────────────────────────

    def inject(self, name: str, value, quiet: bool = False):
        """Inject a variable into the console namespace."""
        self._namespace[name] = value
        if not quiet:
            desc = type(value).__name__
            if hasattr(value, 'shape'):
                desc = f"{desc} shape={value.shape}"
            elif isinstance(value, dict):
                desc = f"dict with {len(value)} keys"
            elif isinstance(value, (list, tuple)):
                desc = f"{desc} of {len(value)} items"
            self._output.append_text(
                f"[exported]  {name} = <{desc}>", color='#66ccaa')

    # ── Helper functions available in the namespace ───────────────────────────

    def _show_plot(self, fig=None):
        """show_plot([fig]) -- display a matplotlib figure inline."""
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=120,
                    facecolor='#0d0d2a', edgecolor='none',
                    bbox_inches='tight')
        buf.seek(0)
        pixmap = QPixmap()
        pixmap.loadFromData(buf.read())
        self._output.append_image(pixmap)

    def _save_fig(self, fig=None, path: str | None = None, dpi: int = 200):
        """save_fig([fig, path, dpi]) -- save a matplotlib figure to file.

        If *path* is None, opens a file-save dialog.  Supported formats:
        png, svg, pdf, eps.
        """
        import matplotlib.pyplot as plt
        if fig is None:
            fig = plt.gcf()
        if path is None:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save figure", "",
                "PNG (*.png);;SVG (*.svg);;PDF (*.pdf);;EPS (*.eps)")
            if not path:
                return
        fig.savefig(path, dpi=dpi, facecolor='#0d0d2a', edgecolor='none',
                    bbox_inches='tight')
        self._output.append_text(f"[saved]  {path}", color='#66ccaa')

    def _save_fig_dialog(self):
        """Button handler: save last figure via dialog."""
        import matplotlib.pyplot as plt
        if not plt.get_fignums():
            self._output.append_text(
                "No open figures. Run a plot first, then save before "
                "it auto-closes, or call save_fig(fig) in code.",
                color='#ff9966')
            return
        self._save_fig()

    def _create_group(self, name: str, unit_indices: list,
                      color: str = '#2196F3'):
        """create_group(name, unit_indices[, color])
        Create a group in the active Cell Compare window.
        """
        main_win = self._find_main_window()
        if main_win is None:
            self._output.append_error("Cannot find main window.")
            return
        if main_win._active_cell_win is None \
                or not main_win._active_cell_win.isVisible():
            self._output.append_error(
                "No active Cell Compare window.  "
                "Export a cell first (X key in the main window).")
            return

        from cell_compare_window import CellEntry
        win = main_win._active_cell_win
        entry = CellEntry(
            label=name,
            color=color,
            unit_indices=list(unit_indices),
            session_data=main_win.data,
            session_name='console',
            is_group=True,
            members=[],
        )
        win.add_cell(entry)
        self._output.append_text(
            f"[import]  Created group '{name}' with {len(unit_indices)} units",
            color='#66ccaa')

    def _add_event(self, key: str, times, label: str | None = None):
        """add_event(key, frame_indices[, label])
        Add an event to the current SessionData.
        """
        main_win = self._find_main_window()
        if main_win is None:
            self._output.append_error("Cannot find main window.")
            return
        main_win.data.add_event(key, np.asarray(times), units='index',
                                label=label)
        main_win._psth_panel.add_event_to_bar(key)
        main_win._psth_panel.refresh()
        self._output.append_text(
            f"[import]  Added event '{key}' ({len(times)} trials)",
            color='#66ccaa')

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _find_main_window(self):
        from app_window import MainWindow
        for w in QApplication.topLevelWidgets():
            if isinstance(w, MainWindow):
                return w
        return None

    def _show_namespace(self):
        """Print the user-facing variables in the namespace."""
        skip = {'__builtins__', 'show_plot', 'save_fig', 'create_group',
                'add_event', 'np', 'plt', 'stats'}
        items = sorted(
            (k, v) for k, v in self._namespace.items()
            if not k.startswith('_') and k not in skip
        )
        if not items:
            self._output.append_text("(namespace is empty)", color='#888888')
            return
        lines = []
        for k, v in items:
            desc = type(v).__name__
            if hasattr(v, 'shape'):
                desc += f'  shape={v.shape}'
            elif isinstance(v, dict):
                desc += f'  ({len(v)} keys)'
            elif isinstance(v, (list, tuple)):
                desc += f'  ({len(v)} items)'
            lines.append(f"  {k:20s}  {desc}")
        self._output.append_text("Namespace:\n" + '\n'.join(lines),
                                 color='#aabbcc')
