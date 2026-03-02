"""
claude_chat.py  --  Claude AI analysis assistant.

Shares the same Python namespace as ConsoleWindow.  Claude can execute
code via the ``execute_python`` tool and see results (stdout + plots).

Requires:  pip install anthropic
"""
from __future__ import annotations

import io
import os
import sys
import traceback

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QColor
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QLineEdit, QPushButton, QLabel, QScrollArea,
    QTextEdit, QPlainTextEdit, QFrame,
    QMessageBox, QInputDialog, QApplication,
)

from population_viewer import BG_COLOR, FG_COLOR

_MONO = QFont('Consolas', 10)

_BTN = (
    "QPushButton { background: #1a1a3e; color: #e0e0e0; "
    "border: 1px solid #3a3a6a; border-radius: 3px; "
    "font-size: 9pt; padding: 3px 10px; }"
    "QPushButton:hover { background: #2a2a6a; }"
    "QPushButton:disabled { background: #111128; color: #555; }"
)
_INPUT = (
    f"QPlainTextEdit {{ background: #0a0e1a; color: #dcdcdc; "
    f"border: 1px solid #3a3a6a; border-radius: 3px; padding: 4px; "
    f"font-size: 10pt; }}"
)
_DESC_STYLE = (
    f"QPlainTextEdit {{ background: #0a0e1a; color: #aabbcc; "
    f"border: 1px solid #3a3a6a; font-size: 9pt; }}"
)

MAX_TOOL_CALLS_PER_TURN = 5
MAX_HISTORY_MESSAGES = 20          # keep last N messages to limit token costs


# ── Chat input (multi-line, Enter sends, Shift+Enter new line) ────────────────

class ChatInput(QPlainTextEdit):
    """Multi-line input: Enter sends, Shift+Enter adds a new line."""
    submit_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont('Consolas', 10))
        self.setMaximumHeight(120)
        self.setPlaceholderText("Ask Claude about your data...  (Shift+Enter for new line)")

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)          # new line
            else:
                self.submit_requested.emit()          # send
            return
        super().keyPressEvent(event)


# ── API worker (runs off the main thread) ─────────────────────────────────────

class ClaudeWorker(QThread):
    """Call the Anthropic messages API on a background thread."""
    finished = pyqtSignal(object)   # full response object
    error    = pyqtSignal(str)

    def __init__(self, client, messages: list, system_prompt: str,
                 tools: list, parent=None):
        super().__init__(parent)
        self._client   = client
        self._messages = messages
        self._system   = system_prompt
        self._tools    = tools

    def run(self):
        try:
            # Send system prompt as a cacheable block to reduce token costs.
            # After the first call the cached prompt is ~90% cheaper.
            system_blocks = [{
                'type': 'text',
                'text': self._system,
                'cache_control': {'type': 'ephemeral'},
            }]
            resp = self._client.messages.create(
                model='claude-sonnet-4-6',
                max_tokens=4096,
                system=system_blocks,
                messages=self._messages,
                tools=self._tools,
            )
            self.finished.emit(resp)
        except Exception as e:
            self.error.emit(str(e))


# ── Chat display ──────────────────────────────────────────────────────────────

class ChatDisplay(QScrollArea):
    """Scrollable message list with user/assistant/system bubbles."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setStyleSheet("QScrollArea { background: transparent; }")

        self._inner = QWidget()
        self._inner.setStyleSheet(f"background: {BG_COLOR};")
        self._layout = QVBoxLayout(self._inner)
        self._layout.setContentsMargins(6, 6, 6, 6)
        self._layout.setSpacing(8)
        self._layout.addStretch()   # push messages to top
        self.setWidget(self._inner)

    def _add_bubble(self, html: str, bg: str, align: str = 'left'):
        lbl = QLabel(html)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lbl.setTextInteractionFlags(
            Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        lbl.setCursor(Qt.IBeamCursor)
        lbl.setStyleSheet(
            f"background: {bg}; color: {FG_COLOR}; "
            f"border-radius: 6px; padding: 8px; font-size: 10pt;")
        wrapper = QHBoxLayout()
        wrapper.setContentsMargins(0, 0, 0, 0)
        if align == 'right':
            wrapper.addStretch()
        wrapper.addWidget(lbl, stretch=0)
        if align == 'left':
            wrapper.addStretch()
        w = QWidget()
        w.setLayout(wrapper)
        # Insert before the stretch at the end
        self._layout.insertWidget(self._layout.count() - 1, w)
        QApplication.processEvents()
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def add_user_message(self, text: str):
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        escaped = escaped.replace('\n', '<br>')
        self._add_bubble(escaped, bg='#1a2a4a', align='right')

    def add_assistant_message(self, text: str, code: str = '',
                              image: QPixmap | None = None):
        parts = []
        if text:
            escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            escaped = escaped.replace('\n', '<br>')
            parts.append(escaped)
        if code:
            code_esc = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            parts.append(
                f"<pre style='background:#0a0e1a; padding:6px; "
                f"border-radius:4px; font-family:Consolas; font-size:9pt;'>"
                f"{code_esc}</pre>")
        html = '<br>'.join(parts) if parts else '(empty)'
        self._add_bubble(html, bg='#1a1a3a', align='left')

        if image is not None:
            img_lbl = QLabel()
            if image.width() > 600:
                image = image.scaledToWidth(600, Qt.SmoothTransformation)
            img_lbl.setPixmap(image)
            img_lbl.setStyleSheet("padding: 4px;")
            self._layout.insertWidget(self._layout.count() - 1, img_lbl)
            QApplication.processEvents()
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().maximum())

    def add_system_message(self, text: str):
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        escaped = escaped.replace('\n', '<br>')
        self._add_bubble(
            f"<i style='color:#888'>{escaped}</i>",
            bg='#111128', align='left')

    def clear(self):
        """Remove all message bubbles."""
        while self._layout.count() > 1:      # keep the trailing stretch
            item = self._layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()


# ── Main chat window ──────────────────────────────────────────────────────────

class ClaudeChatWindow(QDialog):
    """Claude AI analysis chat.  Shares the console namespace for code exec."""

    def __init__(self, console_window, parent=None):
        super().__init__(parent)
        self._console = console_window
        self._messages: list[dict] = []
        self._client = None
        self._worker = None
        self._tool_calls_this_turn = 0

        self.setWindowTitle("Claude AI Assistant")
        self.setWindowFlags(self.windowFlags() | Qt.Window)
        self.resize(720, 820)
        self.setStyleSheet(f"background: {BG_COLOR};")

        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        # Chat display
        self._chat = ChatDisplay()
        outer.addWidget(self._chat, stretch=1)

        # Experiment description (collapsible)
        self._desc_toggle = QPushButton("+ Experiment description (optional)")
        self._desc_toggle.setCheckable(True)
        self._desc_toggle.setStyleSheet(_BTN)
        self._desc_toggle.toggled.connect(self._toggle_desc)
        outer.addWidget(self._desc_toggle)

        self._desc_edit = QPlainTextEdit()
        self._desc_edit.setPlaceholderText(
            "Describe your experiment here (e.g. 'Mice running a "
            "T-maze with reward ports, Purkinje cell recordings...').\n"
            "Claude will use this context when analyzing your data.")
        self._desc_edit.setStyleSheet(_DESC_STYLE)
        self._desc_edit.setMaximumHeight(100)
        self._desc_edit.setVisible(False)
        outer.addWidget(self._desc_edit)

        # Input row
        row = QHBoxLayout()
        row.setSpacing(4)
        self._input = ChatInput()
        self._input.setStyleSheet(_INPUT)
        self._input.submit_requested.connect(self._send)
        row.addWidget(self._input, stretch=1)

        self._send_btn = QPushButton("Send")
        self._send_btn.setStyleSheet(_BTN)
        self._send_btn.clicked.connect(self._send)
        row.addWidget(self._send_btn)

        self._new_topic_btn = QPushButton("New topic")
        self._new_topic_btn.setStyleSheet(_BTN)
        self._new_topic_btn.setToolTip(
            "Clear conversation history and start a new analysis direction.\n"
            "The experiment description and console variables are kept.")
        self._new_topic_btn.clicked.connect(self._new_topic)
        row.addWidget(self._new_topic_btn)

        outer.addLayout(row)

    def _new_topic(self):
        """Clear chat history but keep system prompt & experiment description."""
        self._messages.clear()
        self._chat.clear()
        self._chat.add_system_message(
            "Conversation cleared. Starting a new topic.\n"
            "Your experiment description and console variables are still available.")

    def _toggle_desc(self, checked: bool):
        self._desc_edit.setVisible(checked)
        self._desc_toggle.setText(
            "- Experiment description" if checked
            else "+ Experiment description (optional)")

    # ── API key + client ──────────────────────────────────────────────────────

    def _ensure_client(self) -> bool:
        if self._client is not None:
            return True
        try:
            import anthropic
        except ImportError:
            QMessageBox.warning(self, "Missing package",
                "The 'anthropic' package is required.\n\n"
                "Install it with:\n"
                "  pip install anthropic")
            return False

        api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        if not api_key:
            api_key, ok = QInputDialog.getText(
                self, "Anthropic API Key",
                "Enter your Anthropic API key:",
                QLineEdit.Password)
            if not ok or not api_key.strip():
                return False
            api_key = api_key.strip()

        self._client = anthropic.Anthropic(api_key=api_key)
        return True

    # ── System prompt ─────────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        main_win = self._console._find_main_window()
        sd = main_win.data if main_win else None

        prompt = (
            "You are a neuroscience data analysis assistant embedded in "
            "NeuroPy Viewer, a PyQt5 application for visualizing neural data.\n\n"
            "## Available API\n"
            "The user's session data is in variable `sd` (SessionData).  "
            "Key attributes and methods:\n"
            "```\n"
            "sd.fr                         # (n_units, n_frames) firing rate\n"
            "sd.T_bin                      # (n_frames,) timestamps in seconds\n"
            "sd.fps                        # frame rate (Hz)\n"
            "sd.n_units                    # total units\n"
            "sd.n_neural_units             # neural units only (excl. behavior)\n"
            "sd.event_keys                 # list of event names\n"
            "sd.n_trials                   # dict {event_key: n_trials}\n"
            "sd.cell_type_label            # list of cell types per unit\n"
            "sd.probe_depth                # depth per unit (microns)\n"
            "sd.pair_map                   # {unit: paired_unit} for CS-SS\n"
            "sd.beh_unit_indices           # indices of behavioral channels\n"
            "\n"
            "sd.get_mean_psth(unit, event, smooth=0, window=3.0)\n"
            "  -> (mean, sem, time_ax)  # all np.ndarray\n"
            "sd.get_heatmap(unit, event, window=3.0)\n"
            "  -> (trial_matrix, time_ax)\n"
            "sd.get_group_mean_psth(unit_list, event, smooth=0, window=3.0, "
            "normalize='none')\n"
            "  -> (mean, sem, time_ax)\n"
            "sd.compute_block_psth(unit, event, block_size=10, smooth=0)\n"
            "  -> (block_matrix, time_ax)\n"
            "sd.compute_block_metrics(unit, event, block_size=10, smooth=0)\n"
            "  -> dict: block_nums, amplitude, best_lag_ms\n"
            "```\n\n"
            "## Pre-imported modules\n"
            "`np` (numpy), `plt` (matplotlib.pyplot), `stats` (scipy.stats)\n\n"
            "## Helper functions\n"
            "- `show_plot(fig)` -- display matplotlib figure inline\n"
            "- `save_fig([fig, path, dpi])` -- save figure to file (dialog if no path)\n"
            "- `create_group(name, unit_indices)` -- create group in Cell "
            "Compare window\n"
            "- `add_event(key, frame_indices)` -- add event to session\n\n"
        )

        if sd is not None:
            n_types = {}
            for t in sd.cell_type_label[:sd.n_neural_units]:
                n_types[t] = n_types.get(t, 0) + 1
            type_str = ', '.join(f'{k}={v}' for k, v in sorted(n_types.items()))
            prompt += (
                f"## Current session\n"
                f"- Path: {getattr(sd, '_res_path', 'unknown')}\n"
                f"- Units: {sd.n_neural_units} neural + "
                f"{sd.n_units - sd.n_neural_units} behavioral\n"
                f"- Cell types: {type_str}\n"
                f"- Events: {sd.event_keys}\n"
                f"- Trials: {sd.n_trials}\n"
            )
            if sd.T_bin is not None:
                prompt += (
                    f"- Duration: {sd.T_bin[-1]:.0f} s  |  FPS: {sd.fps:.1f}\n"
                )
            prompt += '\n'

        # Namespace variables
        ns_vars = sorted(
            k for k in self._console._namespace
            if not k.startswith('_') and k not in
            {'np', 'plt', 'stats', 'show_plot', 'create_group', 'add_event'}
        )
        if ns_vars:
            prompt += "## Variables currently in the console namespace\n"
            for k in ns_vars:
                v = self._console._namespace[k]
                desc = type(v).__name__
                if hasattr(v, 'shape'):
                    desc += f' shape={v.shape}'
                prompt += f"- `{k}` : {desc}\n"
            prompt += '\n'

        desc = self._desc_edit.toPlainText().strip()
        if desc:
            prompt += f"## Experiment description\n{desc}\n\n"

        prompt += (
            "## Instructions\n"
            "When the user asks for analysis, use the `execute_python` tool "
            "to run code.  Always call show_plot() to display figures.  "
            "Keep code concise.  Explain results in plain language.  "
            "Use dark matplotlib styles (facecolor='#0d0d2a', text white)."
        )
        return prompt

    def _get_tools(self) -> list:
        return [{
            'name': 'execute_python',
            'description': (
                'Execute Python code in the shared console namespace.  '
                'Variables persist across calls.  Use show_plot() for figures.  '
                'Has numpy (np), matplotlib.pyplot (plt), scipy.stats (stats).'
            ),
            'input_schema': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python code to execute',
                    }
                },
                'required': ['code'],
            },
        }]

    # ── Send / receive ────────────────────────────────────────────────────────

    def _send(self):
        text = self._input.toPlainText().strip()
        if not text:
            return
        if not self._ensure_client():
            return

        self._input.clear()
        self._chat.add_user_message(text)
        self._messages.append({'role': 'user', 'content': text})

        self._tool_calls_this_turn = 0
        self._call_api()

    def _call_api(self):
        self._input.setEnabled(False)
        self._send_btn.setEnabled(False)
        # Trim history to last N messages to control token costs.
        # Always keep pairs intact (don't split assistant/tool_result pairs).
        if len(self._messages) > MAX_HISTORY_MESSAGES:
            self._messages = self._messages[-MAX_HISTORY_MESSAGES:]
            # Ensure first message is from 'user' (API requirement)
            while self._messages and self._messages[0].get('role') != 'user':
                self._messages.pop(0)
        self._worker = ClaudeWorker(
            self._client, list(self._messages),
            self._build_system_prompt(), self._get_tools())
        self._worker.finished.connect(self._on_response)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _on_response(self, response):
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)

        text_parts = []
        tool_results = []

        for block in response.content:
            if block.type == 'text':
                text_parts.append(block.text)
            elif block.type == 'tool_use' and block.name == 'execute_python':
                self._tool_calls_this_turn += 1
                code = block.input.get('code', '')
                output, error, pixmap = self._execute_code(code)

                self._chat.add_assistant_message('', code=code, image=pixmap)
                if output:
                    self._chat.add_system_message(output[:2000])
                if error:
                    self._chat.add_system_message(f"Error:\n{error[:2000]}")

                tool_results.append({
                    'id': block.id,
                    'output': (output or '') + (error or '') or '(no output)',
                })

        # Show text response
        if text_parts:
            self._chat.add_assistant_message('\n'.join(text_parts))

        # If there were tool calls, send results back and continue
        if tool_results:
            # Append assistant message with full content blocks
            self._messages.append({
                'role': 'assistant',
                'content': response.content,
            })
            # Append tool results
            result_blocks = []
            for tr in tool_results:
                result_blocks.append({
                    'type': 'tool_result',
                    'tool_use_id': tr['id'],
                    'content': tr['output'],
                })
            self._messages.append({
                'role': 'user',
                'content': result_blocks,
            })

            # Continue conversation (Claude may explain results)
            if self._tool_calls_this_turn < MAX_TOOL_CALLS_PER_TURN:
                self._call_api()
                return
            else:
                self._chat.add_system_message(
                    "(Reached max tool calls per turn.)")
        elif text_parts:
            # Pure text response — store in message history
            self._messages.append({
                'role': 'assistant',
                'content': '\n'.join(text_parts),
            })

    def _on_error(self, msg: str):
        self._input.setEnabled(True)
        self._send_btn.setEnabled(True)
        self._chat.add_system_message(f"API error: {msg}")

    # ── Code execution (shared with console) ──────────────────────────────────

    def _execute_code(self, code: str) -> tuple:
        """Execute code in the console namespace.
        Returns (stdout_str, error_str, pixmap_or_None).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        plt.close('all')
        old_stdout, old_stderr = sys.stdout, sys.stderr
        buf = io.StringIO()
        sys.stdout = buf
        sys.stderr = buf

        error = ''
        try:
            exec(compile(code, '<claude>', 'exec'), self._console._namespace)
        except Exception:
            error = traceback.format_exc()
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

        output = buf.getvalue()

        pixmap = None
        if plt.get_fignums():
            fig = plt.gcf()
            b = io.BytesIO()
            fig.savefig(b, format='png', dpi=120,
                        facecolor='#0d0d2a', edgecolor='none',
                        bbox_inches='tight')
            b.seek(0)
            pixmap = QPixmap()
            pixmap.loadFromData(b.read())
            plt.close('all')

        return output, error, pixmap
