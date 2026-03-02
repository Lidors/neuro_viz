"""
main.py  –  NeuroPy Viewer entry point.

Usage
-----
    python main.py                       # opens file dialog
    python main.py path/to/res.mat       # loads directly
"""

import sys
import os

sys.setrecursionlimit(5000)   # pyqtgraph WidgetGroup walks widget tree recursively
sys.path.insert(0, os.path.dirname(__file__))

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui     import QPalette, QColor
from PyQt5.QtCore    import Qt


def _apply_dark_palette(app: QApplication):
    app.setStyle("Fusion")
    p = QPalette()
    # Window / background tones
    p.setColor(QPalette.Window,          QColor(13,  13,  42))
    p.setColor(QPalette.WindowText,      QColor(224, 224, 224))
    p.setColor(QPalette.Base,            QColor( 9,   9,  30))
    p.setColor(QPalette.AlternateBase,   QColor(22,  22,  62))
    # Inputs / widgets
    p.setColor(QPalette.Text,            QColor(224, 224, 224))
    p.setColor(QPalette.Button,          QColor(26,  26,  62))
    p.setColor(QPalette.ButtonText,      QColor(224, 224, 224))
    # Accent
    p.setColor(QPalette.Highlight,       QColor(58,  90, 154))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    # Tooltips
    p.setColor(QPalette.ToolTipBase,     QColor(13,  13,  42))
    p.setColor(QPalette.ToolTipText,     QColor(200, 200, 220))
    # Disabled
    p.setColor(QPalette.Disabled, QPalette.Text,       QColor(100, 100, 140))
    p.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(100, 100, 140))
    app.setPalette(p)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("NeuroPy Viewer")
    _apply_dark_palette(app)

    # ── pyqtgraph config (MUST be after QApplication, before any pg widgets) ──
    import pyqtgraph as pg
    pg.setConfigOption('background', '#16213e')
    pg.setConfigOption('foreground', '#e0e0e0')
    pg.setConfigOption('antialias',  True)

    # ── Determine res.mat path ──────────────────────────────────────────────
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        res_path = sys.argv[1]
    else:
        res_path, _ = QFileDialog.getOpenFileName(
            None,
            "Open session file (res.mat)",
            os.path.expanduser("~"),
            "MAT files (*.mat);;All files (*)"
        )
        if not res_path:
            return

    # ── Detect format before deciding what companion files to look for ────────
    folder = os.path.dirname(res_path)
    import scipy.io as _sio
    try:
        _keys = _sio.loadmat(res_path, variable_names=[]).keys()
        _is_new = 'new_res' in _keys
    except Exception:
        _is_new = False   # will be determined properly inside SessionData

    if _is_new:
        # New format: T and SST are embedded — no companion files needed
        t_bin    = None
        sst_path = None
        print("  Format: new_res (T and SST embedded)", flush=True)
    else:
        # Old format: look for T_bin.mat and sst.mat
        t_bin = os.path.join(folder, 'T_bin.mat')
        t_bin = t_bin if os.path.exists(t_bin) else None

        sst_path = os.path.join(folder, 'sst.mat')
        if not os.path.exists(sst_path):
            try:
                tmp = _sio.loadmat(res_path, squeeze_me=True,
                                   struct_as_record=False)
                roi = str(tmp['res'].roi_file)
                data_folder = os.path.dirname(os.path.dirname(roi))
                candidate   = os.path.join(data_folder, 'sst.mat')
                sst_path = candidate if os.path.exists(candidate) else None
            except Exception:
                sst_path = None
        if sst_path:
            print(f"  SST:    {sst_path}", flush=True)
        else:
            print("  SST:    not found (WF / ACG unavailable)", flush=True)

    # ── Load data ────────────────────────────────────────────────────────────
    try:
        from data_loader import SessionData
        print(f"Loading {res_path} …", flush=True)
        data = SessionData(res_path, t_bin, sst_path)
        trial_str = "  ".join(
            f"{k}={data.n_trials[k]}" for k in data.event_keys)
        print(f"  {data.n_units} units  |  {trial_str}", flush=True)
        if data.T_bin is not None:
            print(f"  Time:   {len(data.T_bin)} bins, "
                  f"{data.fps:.1f} Hz, "
                  f"{data.T_bin[-1]:.0f} s total", flush=True)
        else:
            print("  Time:   no timing available – using pre-computed PSTH",
                  flush=True)
        if data._sst_loaded:
            print("  SST:    loaded (WF / ACG available)", flush=True)
    except Exception as e:
        QMessageBox.critical(None, "Failed to load", str(e))
        raise

    # ── Launch window ────────────────────────────────────────────────────────
    from app_window import MainWindow
    window = MainWindow(data)
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
