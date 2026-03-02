"""
data_loader.py  –  Session data container.

Supports two file formats:

  New format  (new_res.mat)
  -------------------------
  new_res.fr               – (n_units, n_frames) firing-rate matrix
  new_res.session.T        – (n_frames+1,) timestamps in seconds
  new_res.session.fs       – neural sample rate (int, e.g. 30000)
  new_res.Triggers         – struct with all event onsets:
      .port1R / .port2R    – direct frame indices into fr  (R1 / R2)
      .AllR                – direct frame indices (all reward)
      .stNR1 / .stNR2      – sample numbers at fs  (NR1 / NR2)
      .allNR               – sample numbers (all no-reward)
  new_res.sst              – spike-sorting table, 1:1 with fr rows

  Old format  (res.mat)
  ---------------------
  res.fr / res.t_bins / res.psth_p1r … (unchanged from previous version)
  res.stR1/stR2/stNR1/stNR2  – event onset samples at 30 kHz
  res.map                     – cluster-ID ↔ row mapping
  sst.mat                     – separate file with waveforms / ACG
"""

import os
import numpy as np
import scipy.io
import h5py
from scipy.ndimage import gaussian_filter1d
from analysis import block_amplitude, block_best_lag

FS = 30_000   # neural sample rate (Hz) – old format fallback

# ── auto colour palette for dynamically added events ──────────────────────────
_AUTO_COLORS = [
    '#E91E63', '#9C27B0', '#00BCD4', '#8BC34A',
    '#FF5722', '#607D8B', '#CDDC39', '#795548',
]

# ── Trigger colours – assigned sequentially to each discovered field ───────────
_TRIGGER_COLORS = [
    '#2196F3', '#4CAF50', '#FF9800', '#F44336',
    '#00BCD4', '#FF5722', '#9C27B0', '#E91E63',
    '#8BC34A', '#607D8B', '#CDDC39', '#795548',
]


def _load_mat(path: str) -> dict:
    """Load .mat supporting MATLAB < v7.3 and v7.3 (HDF5)."""
    try:
        return scipy.io.loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError:
        out = {}
        with h5py.File(path, 'r') as f:
            for k in f.keys():
                out[k] = f[k][()]
        return out


# ──────────────────────────────────────────────────────────────────────────────
class SessionData:
    """
    Container for one recording session.

    Public attributes
    -----------------
    n_units          : int
    fr               : (n_units, n_frames) float64  – full-session FR
    T_bin            : (n_frames,) float64           – fr timestamps in seconds
    fps              : float
    is_new_format    : bool
    event_keys       : list[str]
    n_trials         : dict[str -> int]
    probe_pos        : (n_units,) int   – y-coordinate / depth µm
    probe_depth      : (n_units,) int   – same (from SST)
    cell_type_label  : list[str]
    """

    def __init__(self, res_path: str, t_bin_path: str | None = None,
                 sst_path: str | None = None):
        self._res_path       = res_path
        self._t_bin_path     = t_bin_path
        self._sst_path       = sst_path
        self._auto_color_idx = 0
        self._sst_loaded     = False
        self._load()

    # ── Top-level loader ──────────────────────────────────────────────────────
    def _load(self):
        mat = _load_mat(self._res_path)
        self._heatmap_cache: dict[tuple, tuple] = {}

        if 'new_res' in mat:
            self.is_new_format = True
            self._load_new_format(mat['new_res'])
        else:
            self.is_new_format = False
            # Skip scipy metadata keys (__header__, __version__, __globals__)
            data_keys = [k for k in mat if not k.startswith('__')]
            if 'res' in mat:
                res = mat['res']
            elif data_keys:
                res = mat[data_keys[0]]
            else:
                raise ValueError("No data struct found in the .mat file.")
            self._load_old_format(res)

    # ── New format ────────────────────────────────────────────────────────────
    def _load_new_format(self, r):
        """Load from new_res struct."""
        self.fr      = np.asarray(r.fr, dtype=np.float64)   # (n_units, n_frames)
        self.n_units = self.fr.shape[0]
        n_frames     = self.fr.shape[1]

        # Time axis: session.T has n_frames or n_frames+1 entries
        T_full    = np.asarray(r.session.T, dtype=np.float64).ravel()
        T         = T_full[:n_frames]          # one timestamp per fr column
        self.T_bin = T
        self.fps  = 1.0 / float(T[1] - T[0]) if n_frames > 1 else 60.0
        fs        = int(r.session.fs)

        self._load_triggers(r.Triggers, T, fs, n_frames)

        # No pre-computed PSTHs in new format
        self._psth_precomp    = {}
        self._t_bins_precomp  = None

        # SST is embedded and 1:1 with fr rows
        self._load_sst_embedded(r.sst)

        # Build CS/SS pair map from Pairs field (cluster IDs → fr row indices)
        pairs_raw = (np.asarray(r.Pairs, dtype=int)
                     if hasattr(r, 'Pairs') and np.asarray(r.Pairs).size > 0
                     else np.empty((0, 2), dtype=int))
        self.pair_map = self._build_pair_map(pairs_raw, self._sst_map)

        # Append behavioral channels as pseudo-unit rows in fr
        self._append_behavioral(r.Behavior, n_frames)

    # ── Trigger loader (dynamic) ───────────────────────────────────────────────
    def _load_triggers(self, tr, T: np.ndarray, fs: int, n_frames: int):
        """
        Discover and load every field in the Triggers struct.

        Every field is treated as direct frame indices into fr.
        The MATLAB field name is used as-is for the event key.
        Empty or unreadable fields are skipped silently.
        """
        # Enumerate field names from whatever struct type scipy returns
        if hasattr(tr, '_fieldnames'):            # scipy mat_struct
            all_fields = list(tr._fieldnames)
        elif hasattr(tr, 'dtype') and tr.dtype.names:  # structured array
            all_fields = list(tr.dtype.names)
        else:                                           # generic object
            all_fields = [a for a in dir(tr) if not a.startswith('_')]

        self._event_frames:    dict[str, np.ndarray] = {}
        self._event_samples:   dict[str, np.ndarray] = {}
        self._event_key_order: list[str]             = []
        self._event_meta:      dict[str, dict]       = {}

        for ci, field in enumerate(all_fields):
            try:
                raw = getattr(tr, field)
                arr = np.atleast_1d(np.asarray(raw)).ravel()
                if arr.size == 0:
                    continue
            except Exception:
                continue

            # All fields → direct frame indices
            idx     = np.clip(arr.astype(np.int64), 0, n_frames - 1).astype(int)
            samples = T[idx] * fs   # back-convert for add_event compatibility
            color   = _TRIGGER_COLORS[ci % len(_TRIGGER_COLORS)]

            self._event_frames[field]  = idx
            self._event_samples[field] = samples
            self._event_key_order.append(field)
            self._event_meta[field]    = {'label': field, 'color': color}

        self.n_trials = {k: len(v) for k, v in self._event_frames.items()}

    def _load_sst_embedded(self, sst):
        """Load SST directly from the new_res struct (no map matching needed)."""
        sst_pos = np.asarray(sst.pos,       dtype=np.float64)   # (n_units, 2)
        sst_ct  = np.asarray(sst.cell_type, dtype=np.float64)   # (n_units, 3)

        self.probe_pos   = sst_pos[:, 1].astype(int)   # y-coord = depth µm
        self.probe_depth = sst_pos[:, 1].astype(int)
        self._cluster_ids = np.arange(self.n_units, dtype=int)
        # Store cluster-ID map so pair lookup can use it
        self._sst_map = (np.asarray(sst.map, dtype=int)
                         if hasattr(sst, 'map') else np.arange(self.n_units, dtype=int))

        self.cell_type_code  = sst_ct[:, 0]
        self.cell_type_label = [
            'PC' if ct == 1 else 'IN' if ct == 2 else f'CT{int(ct)}'
            for ct in self.cell_type_code
        ]

        # Direct 1:1 index mapping
        self._res_to_sst = {i: i for i in range(self.n_units)}

        self._sst_meanWF   = np.asarray(sst.meanWF,      dtype=np.float64)  # (8, 81, n)
        self._sst_stdWF    = np.asarray(sst.stdWF,       dtype=np.float64)
        self._sst_channels = np.asarray(sst.channels,    dtype=int)          # (8, n)
        self._sst_main_chan= np.asarray(sst.main_chan,   dtype=int)          # (n,)
        self._sst_chans_y  = np.asarray(sst.chans_ycoord,dtype=np.float64)  # (384,)
        self._sst_ACH      = np.asarray(sst.ACH,         dtype=np.float64)  # (201, n)
        self._sst_ccg_bins = np.asarray(sst.ccg_bins,    dtype=np.float64)  # (201,)
        self._sst_loaded   = True

    def _build_pair_map(self, pairs_raw: np.ndarray,
                        sst_map: np.ndarray) -> dict:
        """
        Build {unit_idx: paired_unit_idx | None} from cluster-ID pair table.

        pairs_raw : (n_pairs, 2) – each row is [cid_a, cid_b] in original IDs
        sst_map   : (n_units,)   – cluster ID for each fr row
        Returns a dict where both directions are stored; value is None when
        the partner cluster ID is not present in sst_map.
        """
        if pairs_raw.size == 0:
            return {}
        pairs_raw  = np.atleast_2d(pairs_raw)
        cid_to_idx = {int(cid): i for i, cid in enumerate(sst_map)}
        result: dict[int, int | None] = {}
        for a_cid, b_cid in pairs_raw:
            ia = cid_to_idx.get(int(a_cid))
            ib = cid_to_idx.get(int(b_cid))
            if ia is not None:
                result[ia] = ib   # ib may be None if partner not in dataset
            if ib is not None:
                result[ib] = ia
        return result

    def _append_behavioral(self, beh, n_frames: int):
        """Append X_pos, Y_pos, Speed as pseudo-unit rows at the end of fr."""
        def _pad(arr):
            a = np.asarray(arr, dtype=np.float64).ravel()
            if len(a) >= n_frames:
                return a[:n_frames]
            return np.concatenate([a, np.full(n_frames - len(a), np.nan)])

        def _get_beh(field):
            raw = getattr(beh, field, None)
            if raw is None:
                return None
            try:
                return _pad(raw)
            except Exception:
                return None

        _CANDIDATES = [
            ('X_pos',  'Xpos',  'X pos (cm)'),
            ('Y_pos',  'Ypos',  'Y pos (cm)'),
            ('Speed',  'Speed', 'Speed (cm/s)'),
        ]
        beh_signals = []
        for src_field, key, ylabel in _CANDIDATES:
            data = _get_beh(src_field)
            if data is not None:
                beh_signals.append((key, ylabel, data))

        if not beh_signals:
            return   # nothing to append

        self.n_neural_units = self.n_units   # save original neural count

        beh_matrix = np.stack([s for _, _, s in beh_signals], axis=0)
        self.fr     = np.vstack([self.fr, beh_matrix])
        n_beh       = len(beh_signals)
        self.n_units += n_beh

        # Extend unit metadata arrays
        self.probe_pos   = np.concatenate([self.probe_pos,   np.zeros(n_beh, dtype=int)])
        self.probe_depth = np.concatenate([self.probe_depth, np.zeros(n_beh, dtype=int)])
        self.cell_type_code = np.concatenate([self.cell_type_code, np.zeros(n_beh)])
        for lbl, _, _ in beh_signals:
            self.cell_type_label.append(lbl)

        # y-axis label and index list for UI use
        self._beh_unit_labels: dict[int, str] = {}
        for i, (_, ylabel, _) in enumerate(beh_signals):
            self._beh_unit_labels[self.n_neural_units + i] = ylabel
        self.beh_unit_indices = list(range(self.n_neural_units, self.n_units))
        # _res_to_sst is NOT extended — behavior units have no WF/ACG

    # ── Old format ────────────────────────────────────────────────────────────
    def _load_old_format(self, res):
        """
        Generic loader for any old-style res struct.

        Only `fr` is required.  Everything else is discovered automatically:
          • T_bin   – from separate file, or from res.T / res.t_bins / res.T_bin
          • Triggers – from res.Triggers sub-struct (preferred) or discovered
                       by scanning all 1-D integer-valued fields
          • Metadata – depth / cell-type from res.map, or zeroed if absent
          • Pre-computed PSTHs – loaded if present (psth_p1r etc.)
          • SST    – from separate sst.mat if provided
        """
        # ── fr matrix (required) ──────────────────────────────────────────
        self.fr      = np.asarray(res.fr, dtype=np.float64)
        self.n_units = self.fr.shape[0]
        n_frames     = self.fr.shape[1]
        self.n_neural_units   = self.n_units
        self.beh_unit_indices = []
        self._beh_unit_labels: dict[int, str] = {}
        self.pair_map: dict[int, int | None]  = {}
        self._sst_map = np.array([], dtype=int)

        all_fields = self._get_struct_fields(res)
        print(f"  [old format] {self.n_units} units × {n_frames} frames")
        print(f"  [old format] fields found: {all_fields}")

        # ── T_bin (timing) ────────────────────────────────────────────────
        self.T_bin = None
        self.fps   = None

        # 1) Separate T_bin.mat file
        if self._t_bin_path and os.path.exists(self._t_bin_path):
            try:
                tb = _load_mat(self._t_bin_path)
                t_keys = [k for k in tb if not k.startswith('__')]
                raw = tb[t_keys[0]] if t_keys else None
                if raw is not None:
                    self.T_bin = np.asarray(raw, dtype=np.float64).ravel()
                    self.fps   = (len(self.T_bin) /
                                  (self.T_bin[-1] - self.T_bin[0]))
                    print(f"  [old format] T_bin loaded from file "
                          f"({len(self.T_bin)} frames, {self.fps:.1f} Hz)")
            except Exception as e:
                print(f"  [old format] T_bin file error: {e}")

        # 2) Embedded in struct (try common field names)
        if self.T_bin is None:
            for tf in ('T_bin', 'T', 't_bins', 't'):
                raw = getattr(res, tf, None)
                if raw is None:
                    continue
                try:
                    arr = np.asarray(raw, dtype=np.float64).ravel()
                    # t_bins may be in ms
                    if tf == 't_bins':
                        arr = arr / 1000.0
                    if len(arr) > 1:
                        self.T_bin = arr[:n_frames]
                        self.fps   = (len(self.T_bin) /
                                      (self.T_bin[-1] - self.T_bin[0]))
                        print(f"  [old format] T_bin from res.{tf} "
                              f"({len(self.T_bin)} frames, {self.fps:.1f} Hz)")
                        break
                except Exception:
                    continue

        if self.T_bin is None:
            print("  [old format] no timing found – using pre-computed PSTH only")

        # ── Pre-computed PSTHs (optional, for files without T_bin) ────────
        def _opt(field, div=1.0):
            raw = getattr(res, field, None)
            if raw is None:
                return None
            try:
                return np.asarray(raw, dtype=np.float64) / div
            except Exception:
                return None

        self._t_bins_precomp = _opt('t_bins', div=1000.0)
        self._psth_precomp   = {}
        for key, field in [('R1', 'psth_p1r'), ('R2', 'psth_p2r'),
                            ('NR1', 'psth_p1nr'), ('NR2', 'psth_p2nr')]:
            arr = _opt(field)
            if arr is not None:
                self._psth_precomp[key] = arr
        if self._psth_precomp:
            print(f"  [old format] pre-computed PSTHs: {list(self._psth_precomp)}")

        # ── Event / trigger discovery ──────────────────────────────────────
        self._event_frames:    dict[str, np.ndarray] = {}
        self._event_samples:   dict[str, np.ndarray] = {}
        self._event_key_order: list[str]             = []
        self._event_meta:      dict[str, dict]       = {}

        triggers_found: dict[str, np.ndarray] = {}   # {key: frame_idx_array}

        # Strategy A – dedicated Triggers sub-struct
        tr = getattr(res, 'Triggers', None)
        if tr is not None:
            print("  [old format] found res.Triggers sub-struct")
            # Use _load_triggers logic but store into local dict
            tr_fields = self._get_struct_fields(tr)
            for ci, field in enumerate(tr_fields):
                try:
                    raw = getattr(tr, field)
                    arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
                    if arr.size == 0:
                        continue
                    arr_int = arr.astype(np.int64)
                    if arr_int.min() >= 0 and arr_int.max() < n_frames:
                        triggers_found[field] = np.clip(arr_int, 0,
                                                        n_frames - 1).astype(int)
                        print(f"    Triggers.{field}: {arr.size} events (frame idx)")
                    elif (self.T_bin is not None
                          and arr_int.min() >= 0
                          and float(arr_int.max()) / FS < float(self.T_bin[-1]) * 2):
                        triggers_found[field] = self._samples_to_frames(arr)
                        print(f"    Triggers.{field}: {arr.size} events (30kHz→frame)")
                    else:
                        print(f"    Triggers.{field}: skipped "
                              f"(range {arr.min():.0f}–{arr.max():.0f})")
                except Exception as e:
                    print(f"    Triggers.{field}: error ({e})")

        # Strategy B – scan top-level struct fields
        if not triggers_found:
            print("  [old format] scanning top-level fields for triggers…")
            triggers_found = self._discover_triggers_old(res, all_fields, n_frames)

        # Store discovered triggers as event_frames (direct frame indices)
        for ci, (key, idx_arr) in enumerate(triggers_found.items()):
            color = _TRIGGER_COLORS[ci % len(_TRIGGER_COLORS)]
            self._event_frames[key]  = idx_arr
            self._event_samples[key] = (self.T_bin[idx_arr] * FS
                                        if self.T_bin is not None
                                        else idx_arr.astype(float))
            self._event_key_order.append(key)
            self._event_meta[key] = {'label': key, 'color': color}

        self.n_trials = {k: len(v) for k, v in self._event_frames.items()}
        if self.n_trials:
            print(f"  [old format] events loaded: "
                  + "  ".join(f"{k}={v}" for k, v in self.n_trials.items()))
        else:
            print("  [old format] no events found")

        # ── Unit metadata + WF/ACG ────────────────────────────────────────
        self._cluster_ids   = np.arange(self.n_units, dtype=int)
        self.probe_pos      = np.zeros(self.n_units, dtype=int)
        self.probe_depth    = np.zeros(self.n_units, dtype=int)
        self.cell_type_code = np.zeros(self.n_units)

        # res.sst is the primary source (embedded in the mat file)
        sst_sub = getattr(res, 'sst', None)
        if sst_sub is not None:
            print("  [old format] found res.sst – loading metadata + WF/ACG (1:1)")
            self._load_sst_struct(sst_sub, direct=True)
        elif self._sst_path and os.path.exists(self._sst_path):
            self._load_sst_file(self._sst_path)
        else:
            self._sst_loaded = False

        self.cell_type_label = [
            'PC' if ct == 1 else 'IN' if ct == 2 else f'CT{int(ct)}'
            for ct in self.cell_type_code
        ]

        # ── Pairs ─────────────────────────────────────────────────────────
        pairs_field = getattr(res, 'Pairs', None)
        if pairs_field is not None:
            try:
                pairs_raw = np.atleast_2d(np.asarray(pairs_field, dtype=int))
                if pairs_raw.size > 0:
                    self.pair_map = self._build_pair_map(pairs_raw, self._sst_map)
                    print(f"  [old format] {len(self.pair_map)} paired units")
            except Exception as e:
                print(f"  [old format] Pairs error: {e}")

        # ── Behavioral channels ────────────────────────────────────────────
        beh = getattr(res, 'Behavior', None)
        if beh is not None:
            try:
                self._append_behavioral(beh, n_frames)
                print(f"  [old format] behavioral channels appended "
                      f"(total units now {self.n_units})")
            except Exception as e:
                print(f"  [old format] Behavior error: {e}")

    def _samples_to_frames(self, samples: np.ndarray) -> np.ndarray:
        event_sec = samples / FS
        idx = np.searchsorted(self.T_bin, event_sec)
        return np.clip(idx, 0, len(self.T_bin) - 1).astype(int)

    # ── Generic struct helpers ────────────────────────────────────────────────
    @staticmethod
    def _get_struct_fields(obj) -> list:
        """Return field names from a scipy mat_struct, structured ndarray, or plain object."""
        if hasattr(obj, '_fieldnames'):
            return list(obj._fieldnames)
        if hasattr(obj, 'dtype') and hasattr(obj.dtype, 'names') and obj.dtype.names:
            return list(obj.dtype.names)
        return [a for a in dir(obj) if not a.startswith('_')]

    def _discover_triggers_old(self, res, fields: list,
                               n_frames: int) -> dict:
        """
        Scan a struct for trigger-like 1-D arrays and return
        {field_name: frame_index_array}.

        Two heuristics:
          Frame-index  – values in [0, n_frames), treated directly.
          30 kHz sample – max_val / FS < session_length * 2 and T_bin available.

        Arrays that are too large (> n_frames * 2 elements) are skipped
        because they are likely the fr matrix or similar dense signals.
        """
        triggers: dict[str, np.ndarray] = {}
        session_dur_s = (float(self.T_bin[-1]) if self.T_bin is not None
                         else n_frames / 60.0)

        # Skip fields that are almost certainly not triggers
        _SKIP = {'fr', 'psth_p1r', 'psth_p2r', 'psth_p1nr', 'psth_p2nr',
                 't_bins', 'map', 'roi_file', 'name', 'path'}

        for field in fields:
            if field.lower() in _SKIP:
                continue
            raw = getattr(res, field, None)
            if raw is None:
                continue
            try:
                arr = np.atleast_1d(np.asarray(raw, dtype=np.float64)).ravel()
            except Exception:
                continue

            if arr.size < 1 or arr.ndim != 1:
                continue
            if arr.size > n_frames * 2:
                continue
            if not np.issubdtype(arr.dtype, np.floating):
                continue
            # Must look like integers (event indices / sample numbers)
            if np.any(arr != np.floor(arr)):
                continue

            arr_int = arr.astype(np.int64)

            # Heuristic 1 – values already in frame-index range
            if arr_int.min() >= 0 and arr_int.max() < n_frames:
                print(f"    [discover] '{field}': {arr.size} events "
                      f"→ frame indices (range {arr_int.min()}–{arr_int.max()})")
                triggers[field] = np.clip(arr_int, 0, n_frames - 1).astype(int)

            # Heuristic 2 – values look like 30 kHz samples
            elif (self.T_bin is not None
                  and arr_int.min() >= 0
                  and float(arr_int.max()) / FS < session_dur_s * 2):
                print(f"    [discover] '{field}': {arr.size} events "
                      f"→ converting 30 kHz samples")
                triggers[field] = self._samples_to_frames(arr)

        return triggers

    # ── SST loading ───────────────────────────────────────────────────────────
    def _load_sst_file(self, path: str):
        """Load SST from a separate .mat file and delegate to _load_sst_struct."""
        try:
            mat = _load_mat(path)
        except Exception as e:
            print(f"  [SST] could not load {path}: {e}")
            self._sst_loaded = False
            return

        data_keys = [k for k in mat if not k.startswith('__')]
        if 'sst' in mat:
            sst = mat['sst']
        elif data_keys:
            sst = mat[data_keys[0]]
            print(f"  [SST] using key '{data_keys[0]}' instead of 'sst'")
        else:
            print("  [SST] no data struct found in file")
            self._sst_loaded = False
            return

        self._load_sst_struct(sst)

    def _load_sst_struct(self, sst, direct: bool = False):
        """
        Populate unit metadata and WF/ACG from an SST struct.

        Parameters
        ----------
        direct : bool
            True  → sst rows are 1:1 with fr rows (no map matching needed).
            False → use sst.map to match cluster IDs to fr rows.

        Expected fields (all optional):
          map        – (n_sst,) cluster IDs (used only when direct=False)
          pos        – (n_sst, 2+) probe position; column 1 = depth µm
          cell_type  – (n_sst, …) cell-type codes; column 0 is used
          meanWF / stdWF / channels / main_chan / chans_ycoord / ACH / ccg_bins
        """
        sst_fields = self._get_struct_fields(sst)
        print(f"  [SST] fields: {sst_fields}")

        def _get(field, dtype, fallback=None):
            raw = getattr(sst, field, None)
            if raw is None:
                return fallback
            try:
                return np.asarray(raw, dtype=dtype)
            except Exception as e:
                print(f"  [SST] field '{field}' error: {e}")
                return fallback

        # ── Build res→sst index map ────────────────────────────────────────
        sst_map_raw = _get('map', int)
        if sst_map_raw is not None:
            # Store cluster IDs so _build_pair_map can use them
            self._sst_map = sst_map_raw.ravel()
            self._cluster_ids = self._sst_map[:self.n_units].copy()
        else:
            self._sst_map = np.arange(self.n_units, dtype=int)

        if direct:
            # sst is already aligned 1:1 with fr — no lookup needed
            self._res_to_sst = {i: i for i in range(self.n_units)}
            print(f"  [SST] direct 1:1 mapping ({self.n_units} units)")
        else:
            self._res_to_sst: dict[int, int] = {}
            if sst_map_raw is not None:
                for i, cid in enumerate(self._cluster_ids):
                    hits = np.where(self._sst_map == cid)[0]
                    if len(hits):
                        self._res_to_sst[i] = int(hits[0])
                print(f"  [SST] matched {len(self._res_to_sst)} / {self.n_units} units")
            else:
                print("  [SST] no map field — assuming 1:1 correspondence")
                self._res_to_sst = {i: i for i in range(self.n_units)}

        # ── Depth from pos[:,1] ────────────────────────────────────────────
        sst_pos = _get('pos', np.float64)
        if sst_pos is not None and sst_pos.ndim >= 2:
            for i, si in self._res_to_sst.items():
                try:
                    if si < sst_pos.shape[0]:
                        self.probe_depth[i] = int(sst_pos[si, 1])
                        self.probe_pos[i]   = int(sst_pos[si, 1])
                except Exception:
                    pass

        # ── Cell type from cell_type[:,0] ──────────────────────────────────
        sst_ct = _get('cell_type', np.float64)
        if sst_ct is not None:
            for i, si in self._res_to_sst.items():
                try:
                    ct_val  = sst_ct[si, 0] if sst_ct.ndim >= 2 else sst_ct[si]
                    ct_code = int(ct_val)
                    self.cell_type_code[i]  = ct_code
                    self.cell_type_label[i] = (
                        'PC' if ct_code == 1 else
                        'IN' if ct_code == 2 else
                        f'CT{ct_code}'
                    )
                except Exception:
                    pass

        # ── Waveforms / ACG ────────────────────────────────────────────────
        self._sst_meanWF   = _get('meanWF',      np.float64)
        self._sst_stdWF    = _get('stdWF',        np.float64)
        self._sst_channels = _get('channels',     int)
        self._sst_main_chan= _get('main_chan',    int)
        self._sst_chans_y  = _get('chans_ycoord', np.float64)
        self._sst_ACH      = _get('ACH',          np.float64)
        self._sst_ccg_bins = _get('ccg_bins',     np.float64)

        self._sst_loaded = True
        print(f"  [SST] loaded  WF={'yes' if self._sst_meanWF is not None else 'no'}  "
              f"ACG={'yes' if self._sst_ACH is not None else 'no'}")

    def load_sst(self, path: str):
        """Public method to load / reload SST file at runtime (old format)."""
        self._sst_path = path
        self._load_sst_file(path)

    # ── Unit metadata helpers ─────────────────────────────────────────────────
    def get_unit_ylabel(self, unit_idx: int) -> str:
        """Return the appropriate y-axis label for a unit (FR or behavioral)."""
        return self._beh_unit_labels.get(unit_idx, 'FR (Hz)')

    # ── SST data accessors ────────────────────────────────────────────────────
    def get_wf(self, unit_idx: int
               ) -> tuple[np.ndarray, np.ndarray, int] | None:
        if (not self._sst_loaded
                or unit_idx not in self._res_to_sst
                or self._sst_meanWF is None
                or self._sst_stdWF is None
                or self._sst_main_chan is None
                or self._sst_channels is None):
            return None
        si = self._res_to_sst[unit_idx]
        try:
            wf     = self._sst_meanWF[:, :, si]
            wf_std = self._sst_stdWF[:, :, si]
            mc_global     = self._sst_main_chan[si]
            chans         = self._sst_channels[:, si]
            local_matches = np.where(chans == mc_global)[0]
            main_ch_local = int(local_matches[0]) if len(local_matches) else 0
            return wf, wf_std, main_ch_local
        except Exception:
            return None

    def get_acg(self, unit_idx: int
                ) -> tuple[np.ndarray, np.ndarray] | None:
        if (not self._sst_loaded
                or unit_idx not in self._res_to_sst
                or self._sst_ACH is None
                or self._sst_ccg_bins is None):
            return None
        si = self._res_to_sst[unit_idx]
        try:
            return self._sst_ACH[:, si], self._sst_ccg_bins
        except Exception:
            return None

    # ── Event properties ──────────────────────────────────────────────────────
    @property
    def event_keys(self) -> list[str]:
        return list(self._event_key_order)

    def get_event_label(self, key: str) -> str:
        return self._event_meta.get(key, {}).get('label', key)

    def get_event_color(self, key: str) -> str:
        return self._event_meta.get(key, {}).get('color', '#999999')

    # ── Adding new events ─────────────────────────────────────────────────────
    def _next_auto_color(self) -> str:
        color = _AUTO_COLORS[self._auto_color_idx % len(_AUTO_COLORS)]
        self._auto_color_idx += 1
        return color

    def add_event(self, key: str, times: np.ndarray,
                  units: str = 'samples',
                  label: str | None = None,
                  color: str | None = None):
        """
        Add a new event to the session.

        Parameters
        ----------
        key    : unique identifier string
        times  : 1-D array of event onsets
        units  : 'samples' (neural, 30 kHz)  |  'index' (direct frame indices)
        label  : display name  (defaults to key)
        color  : hex colour string  (auto-assigned if None)
        """
        times = np.atleast_1d(times.astype(np.float64))

        if key not in self._event_key_order:
            self._event_key_order.append(key)
        self._event_meta[key] = {
            'label': label or key,
            'color': color or self._next_auto_color(),
        }
        self.n_trials[key] = len(times)

        if units == 'index':
            # times are already direct frame indices
            n_frames = len(self.T_bin) if self.T_bin is not None else 0
            idx = np.clip(times.astype(np.int64), 0, max(n_frames - 1, 0)).astype(int)
            self._event_frames[key]  = idx
            self._event_samples[key] = (self.T_bin[idx] * FS
                                        if self.T_bin is not None
                                        else times)
        else:
            # 'samples' – 30 kHz neural sample numbers
            samples = times
            self._event_samples[key] = samples
            if self.T_bin is not None:
                if self.is_new_format:
                    t_sec = samples / FS
                    idx   = np.searchsorted(self.T_bin, t_sec)
                    self._event_frames[key] = np.clip(
                        idx, 0, len(self.T_bin) - 1).astype(int)
                else:
                    self._event_frames[key] = self._samples_to_frames(samples)

        self._heatmap_cache = {k: v for k, v in self._heatmap_cache.items()
                               if k[1] != key}

    @staticmethod
    def detect_event_vectors(path: str) -> dict[str, np.ndarray]:
        """Open a .mat file and return all 1-D numeric arrays as candidate events."""
        try:
            mat = _load_mat(path)
        except Exception:
            return {}
        candidates = {}
        for k, v in mat.items():
            if k.startswith('_'):
                continue
            arr = np.asarray(v) if not isinstance(v, np.ndarray) else v
            if arr.ndim == 1 and arr.size > 0 and np.issubdtype(arr.dtype, np.number):
                candidates[k] = arr.astype(np.float64)
        return candidates

    # ── Heatmap alignment ─────────────────────────────────────────────────────
    def get_heatmap(self, unit_idx: int, event_key: str,
                    window_sec: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns
        -------
        aligned  : (n_trials, n_time) – NaN where window exceeds recording
        time_ax  : (n_time,) seconds centred on 0
        """
        cache_key = (unit_idx, event_key, window_sec)
        if cache_key in self._heatmap_cache:
            return self._heatmap_cache[cache_key]

        # Fallback when no timing is available (old format without T_bin)
        if self.T_bin is None or event_key not in self._event_frames:
            if self._psth_precomp and event_key in self._psth_precomp:
                mean   = self._psth_precomp[event_key][unit_idx, :]
                result = (mean[np.newaxis, :], self._t_bins_precomp)
            else:
                t      = np.linspace(-window_sec, window_sec, 301)
                result = (np.zeros((1, 301)), t)
            self._heatmap_cache[cache_key] = result
            return result

        fps    = self.fps
        n_pre  = int(window_sec * fps)
        n_post = int(window_sec * fps)
        n_cols = n_pre + n_post
        time_ax = np.linspace(-window_sec, window_sec, n_cols)

        frames   = self._event_frames[event_key]
        n_trials = len(frames)
        aligned  = np.full((n_trials, n_cols), np.nan, dtype=np.float64)

        fr_row = self.fr[unit_idx]
        n_fr   = len(fr_row)

        for i, fi in enumerate(frames):
            start = fi - n_pre
            stop  = fi + n_post
            if start >= 0 and stop <= n_fr:
                aligned[i, :] = fr_row[start:stop]

        result = (aligned, time_ax)
        self._heatmap_cache[cache_key] = result
        return result

    # ── Mean PSTH ─────────────────────────────────────────────────────────────
    def get_mean_psth(self, unit_idx: int, event_key: str,
                      smooth_samples: int = 0,
                      window_sec: float = 3.0
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (mean, sem, time_ax)."""
        aligned, time_ax = self.get_heatmap(unit_idx, event_key, window_sec)
        valid = aligned[~np.any(np.isnan(aligned), axis=1)]

        if len(valid) == 0:
            return np.zeros(len(time_ax)), np.zeros(len(time_ax)), time_ax

        mean = np.mean(valid, axis=0)
        sem  = np.std(valid,  axis=0) / np.sqrt(len(valid))

        if smooth_samples > 0:
            mean = gaussian_filter1d(mean, smooth_samples)
            sem  = gaussian_filter1d(sem,  smooth_samples)

        return mean, sem, time_ax

    # ── Block PSTH ────────────────────────────────────────────────────────────
    def compute_block_psth(self, unit_idx: int, event_key: str,
                           block_size: int = 10,
                           smooth_samples: int = 0,
                           window_sec: float = 3.0
                           ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns (block_matrix, time_ax).
            block_matrix : (n_time, n_blocks) – each column = mean over block
        """
        aligned, time_ax = self.get_heatmap(unit_idx, event_key, window_sec)
        valid    = aligned[~np.any(np.isnan(aligned), axis=1)]
        n_trials = len(valid)

        if n_trials == 0:
            return np.zeros((len(time_ax), 1)), time_ax

        block_size = max(1, block_size)
        n_blocks   = max(1, n_trials // block_size)
        block_cols = []

        for b in range(n_blocks):
            chunk      = valid[b * block_size: (b + 1) * block_size, :]
            mean_chunk = np.mean(chunk, axis=0)
            if smooth_samples > 0:
                mean_chunk = gaussian_filter1d(mean_chunk, smooth_samples)
            block_cols.append(mean_chunk)

        return np.column_stack(block_cols), time_ax

    # ── Block metrics ─────────────────────────────────────────────────────────
    def compute_block_metrics(self, unit_idx: int, event_key: str,
                              block_size: int = 10,
                              smooth_samples: int = 0,
                              window_sec: float = 3.0,
                              xlim: tuple | None = None) -> dict:
        """
        Returns per-block scalar metrics (each shape (n_blocks,)):
            block_nums, amplitude, best_lag_ms

        amplitude   – normalised dot product (zero-meaned block vs zero-meaned
                      mean PSTH), divided by ||Mat||².  1.0 = same modulation
                      depth as the mean.
        best_lag_ms – lag in ms at which the zero-meaned cross-correlation
                      peaks (±1 s window).  Positive = block is delayed.

        xlim : (t_lo, t_hi) – restrict metrics to this time window.
        """
        block_mat, time_ax = self.compute_block_psth(
            unit_idx, event_key, block_size, smooth_samples, window_sec)
        mean_psth, _, _ = self.get_mean_psth(
            unit_idx, event_key, smooth_samples, window_sec)

        if xlim is not None:
            t_lo, t_hi = xlim
            mask = (time_ax >= t_lo) & (time_ax <= t_hi)
            if mask.any():
                block_mat = block_mat[mask, :]
                time_ax   = time_ax[mask]
                mean_psth = mean_psth[mask]

        amplitude   = block_amplitude(block_mat, mean_psth)
        best_lag_ms = block_best_lag(block_mat, mean_psth, time_ax)
        return {
            'block_nums':   np.arange(1, block_mat.shape[1] + 1, dtype=float),
            'amplitude':    amplitude,
            'best_lag_ms':  best_lag_ms,
        }

    # ── Group-level helpers ───────────────────────────────────────────────────
    def get_group_mean_psth(self,
                            unit_indices: list[int],
                            event_key: str,
                            smooth_samples: int = 0,
                            window_sec: float = 3.0,
                            normalize: str = 'none',
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Mean ± SEM PSTH across a group of units.

        normalize : 'none' | 'zscore' | 'peak'
            'zscore' – z-score each cell's full PSTH (subtract mean, divide by std)
            'peak'   – shift by min then divide by max → maps each cell to [0, 1]
        Returns (group_mean, group_sem, time_ax).
        """
        psths, time_ax = [], None
        for u in unit_indices:
            mean, _, tax = self.get_mean_psth(u, event_key, smooth_samples, window_sec)
            time_ax = tax
            if normalize == 'zscore':
                mu, sigma = float(mean.mean()), float(mean.std())
                if sigma > 0:
                    mean = (mean - mu) / sigma
            elif normalize == 'peak':
                mean = mean - float(mean.min())        # shift so min = 0
                mx = float(mean.max())
                if mx > 0:
                    mean = mean / mx                   # scale so max = 1
            psths.append(mean)

        if not psths or time_ax is None:
            n = 100
            return np.zeros(n), np.zeros(n), np.linspace(-window_sec, window_sec, n)

        arr = np.array(psths)   # (n_units, n_time)
        return arr.mean(axis=0), arr.std(axis=0) / np.sqrt(len(psths)), time_ax

    def compute_group_block_metrics(self,
                                    unit_indices: list[int],
                                    event_key: str,
                                    block_size: int = 10,
                                    smooth_samples: int = 0,
                                    window_sec: float = 3.0,
                                    xlim: tuple | None = None,
                                    mode: str = 'percell',
                                    ) -> dict:
        """Block metrics averaged across a group.

        mode='percell'    – compute per-cell metrics then average (with SEM)
        mode='population' – average cell PSTHs first, then compute metrics
                            (no between-cell SEM; shows population signal)

        Returns dict with keys:
            block_nums, amplitude_mean, amplitude_sem,
            best_lag_ms_mean, best_lag_ms_sem
        """
        if mode == 'population':
            block_mats, mean_psths, time_ax = [], [], None
            for u in unit_indices:
                bm, tax = self.compute_block_psth(
                    u, event_key, block_size, smooth_samples, window_sec)
                mp, _, _ = self.get_mean_psth(
                    u, event_key, smooth_samples, window_sec)
                block_mats.append(bm)
                mean_psths.append(mp)
                time_ax = tax
            if not block_mats:
                return {}
            pop_block = np.mean(np.stack(block_mats, axis=0), axis=0)
            pop_mean  = np.mean(np.array(mean_psths), axis=0)
            if xlim is not None:
                t_lo, t_hi = xlim
                mask = (time_ax >= t_lo) & (time_ax <= t_hi)
                if mask.any():
                    pop_block = pop_block[mask, :]
                    time_ax   = time_ax[mask]
                    pop_mean  = pop_mean[mask]
            amp = block_amplitude(pop_block, pop_mean)
            lag = block_best_lag(pop_block, pop_mean, time_ax)
            n_b = pop_block.shape[1]
            return {
                'block_nums':       np.arange(1, n_b + 1, dtype=float),
                'amplitude_mean':   amp,
                'amplitude_sem':    np.zeros(n_b),
                'best_lag_ms_mean': lag,
                'best_lag_ms_sem':  np.zeros(n_b),
            }

        else:   # 'percell'
            all_amp, all_lag, block_nums = [], [], None
            for u in unit_indices:
                try:
                    m = self.compute_block_metrics(
                        u, event_key, block_size, smooth_samples, window_sec, xlim)
                    all_amp.append(m['amplitude'])
                    all_lag.append(m['best_lag_ms'])
                    block_nums = m['block_nums']
                except Exception:
                    pass
            if not all_amp:
                return {}
            min_b   = min(len(a) for a in all_amp)
            amp_arr = np.array([a[:min_b] for a in all_amp])
            lag_arr = np.array([l[:min_b] for l in all_lag])
            n_cells = len(all_amp)
            return {
                'block_nums':       block_nums[:min_b],
                'amplitude_mean':   amp_arr.mean(axis=0),
                'amplitude_sem':    amp_arr.std(axis=0) / np.sqrt(n_cells),
                'best_lag_ms_mean': lag_arr.mean(axis=0),
                'best_lag_ms_sem':  lag_arr.std(axis=0) / np.sqrt(n_cells),
            }

    # ── Combined events ───────────────────────────────────────────────────────
    def combine_events(self, unit_idx: int, event_keys: list[str],
                       window_sec: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
        """Pool trials from multiple events into one heatmap."""
        parts, time_ax = [], None
        for key in event_keys:
            aligned, time_ax = self.get_heatmap(unit_idx, key, window_sec)
            parts.append(aligned)
        return np.concatenate(parts, axis=0), time_ax
