"""
Standalone runner for pc_analysis notebook logic.
Results saved to neuro_viz/notebooks/output/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.cluster import KMeans

from data_loader import SessionData

plt.rcParams.update({
    'figure.dpi': 130,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.labelsize': 9,
    'axes.titlesize': 10,
    'legend.fontsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
RES_PATH    = r'C:\Users\lidor\Nate_lab\code\MATLAB\analisys\Lidor_code\Maze\behavior\AA23_01\res23_05.mat'
WINDOW_SEC  = 1.5
SMOOTH_SIG  = 3          # gaussian sigma in FR bins (~50 ms at 60 Hz)
BASELINE_WIN = (-1.5, -0.5)
PRE_WIN      = (-0.5,  0.0)
MAX_LAG_SEC  = 0.75

# ── Load ─────────────────────────────────────────────────────────────────────
print('Loading session…')
sd = SessionData(RES_PATH)
print(f'  {sd.n_neural_units} neural units | {sd.fps:.1f} Hz | events: {sd.event_keys}')

# ── Identify key units ────────────────────────────────────────────────────────
pc_idx = [i for i in range(sd.n_neural_units) if sd.cell_type_label[i] == 'PC']
print(f'  PCs: {len(pc_idx)}')

speed_idx = next(
    (i for i in sd.beh_unit_indices if sd.cell_type_label[i] == 'Speed'), None)
if speed_idx is None:
    raise RuntimeError('Speed channel not found')

allr_key = next(
    (k for k in sd.event_keys if 'allr' in k.lower()), sd.event_keys[0])
print(f'  AllR key: {allr_key}')

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_psth(uid, key):
    return sd.get_mean_psth(uid, key, SMOOTH_SIG, WINDOW_SEC)

def norm_max(arr):
    mx = np.abs(arr).max()
    return arr / mx if mx > 0 else arr

def norm_peak(arr):
    arr = arr - arr.min()
    mx  = arr.max()
    return arr / mx if mx > 0 else arr

def decrease_score(mean, time_ax):
    b = (time_ax >= BASELINE_WIN[0]) & (time_ax <= BASELINE_WIN[1])
    p = (time_ax >= PRE_WIN[0])      & (time_ax <= PRE_WIN[1])
    bm = float(mean[b].mean()) if b.any() else 1.0
    pm = float(mean[p].mean()) if p.any() else 1.0
    return pm / bm if bm != 0 else 1.0

def xcorr_lag_ms(a, b, time_ax, max_lag=MAX_LAG_SEC):
    dt     = float(time_ax[1] - time_ax[0])
    n      = len(a)
    a_zm   = a - a.mean()
    b_zm   = b - b.mean()
    xc     = correlate(a_zm, b_zm, mode='full')
    lags   = np.arange(-(n-1), n)
    lags_s = lags * dt
    mask   = np.abs(lags_s) <= max_lag
    xc_w   = xc[mask]
    lags_w = lags_s[mask]
    norm   = np.sqrt(np.dot(a_zm, a_zm) * np.dot(b_zm, b_zm))
    xc_w   = xc_w / norm if norm > 0 else xc_w
    idx    = int(np.argmax(np.abs(xc_w)))
    return float(lags_w[idx]) * 1000.0, lags_w * 1000.0, xc_w

# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 – Clustering
# ═══════════════════════════════════════════════════════════════════════════
print('\n-- Part 1: Clustering --')

pc_psths  = {}
pc_scores = {}
for uid in pc_idx:
    mean, sem, tax = get_psth(uid, allr_key)
    pc_psths[uid]  = (mean, sem, tax)
    pc_scores[uid] = decrease_score(mean, tax)

scores_arr = np.array([pc_scores[u] for u in pc_idx])
print(f'  Scores  min={scores_arr.min():.3f}  max={scores_arr.max():.3f}  mean={scores_arr.mean():.3f}')

km     = KMeans(n_clusters=2, n_init=50, random_state=42)
labels = km.fit_predict(scores_arr.reshape(-1, 1))

# ensure label 0 = decreasing (lower score)
if scores_arr[labels==0].mean() > scores_arr[labels==1].mean():
    labels = 1 - labels

dec_idx    = [pc_idx[i] for i, l in enumerate(labels) if l == 0]
nondec_idx = [pc_idx[i] for i, l in enumerate(labels) if l == 1]
print(f'  Decreasing: {len(dec_idx)}  score={scores_arr[labels==0].mean():.3f}')
print(f'  Non-decr  : {len(nondec_idx)}  score={scores_arr[labels==1].mean():.3f}')

tax_ref = pc_psths[pc_idx[0]][2]


def make_heatmap_matrix(uid_list, norm_fn=norm_peak):
    rows = [norm_fn(pc_psths[u][0]) for u in uid_list]
    mat  = np.array(rows)
    order = np.argsort([pc_scores[u] for u in uid_list])
    return mat[order], order


def group_mean_sem(uid_list, norm_fn=norm_peak):
    rows = [norm_fn(pc_psths[u][0]) for u in uid_list]
    arr  = np.array(rows)
    return arr.mean(0), arr.std(0) / np.sqrt(len(rows))


# ── Figure 1: clustering overview ─────────────────────────────────────────────
fig = plt.figure(figsize=(17, 9))
gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.50, wspace=0.38)

# Score histogram
ax  = fig.add_subplot(gs[0, 0])
ax.hist(scores_arr[labels==0], bins=12, alpha=0.75, color='#E53935', label='Decreasing')
ax.hist(scores_arr[labels==1], bins=12, alpha=0.75, color='#1E88E5', label='Non-decr')
ax.axvline(1.0, color='k', lw=1, ls='--')
ax.set_xlabel('Decrease score (pre/baseline)')
ax.set_ylabel('# PCs')
ax.set_title('Score distribution')
ax.legend()

# Mean PSTHs
ax = fig.add_subplot(gs[1, 0])
for uid_list, color, lbl in [(dec_idx,'#E53935','Decreasing'), (nondec_idx,'#1E88E5','Non-decr')]:
    if not uid_list: continue
    m, s = group_mean_sem(uid_list)
    ax.plot(tax_ref, m, color=color, lw=2, label=lbl)
    ax.fill_between(tax_ref, m-s, m+s, color=color, alpha=0.18)
ax.axvline(0, color='gray', lw=1, ls='--')
ax.set_xlabel('Time from reward (s)')
ax.set_ylabel('Norm FR (peak)')
ax.set_title(f'Mean PSTH  [{allr_key}]')
ax.legend()

# Heatmaps per group
for col, (uid_list, glabel, cmap) in enumerate(
        [(dec_idx,'Decreasing','RdBu_r'), (nondec_idx,'Non-decr','RdBu_r')], start=1):
    if not uid_list: continue
    mat, _ = make_heatmap_matrix(uid_list)
    ax = fig.add_subplot(gs[:, col])
    vmax = np.nanpercentile(np.abs(mat), 98)
    im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=-vmax, vmax=vmax,
                   extent=[tax_ref[0], tax_ref[-1], len(uid_list), 0],
                   interpolation='nearest')
    ax.axvline(0, color='w', lw=0.8, ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PC (sorted by score)')
    ax.set_title(f'{glabel}  n={len(uid_list)}')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Norm FR')

# All PCs sorted
all_norm = np.array([norm_peak(pc_psths[u][0]) for u in pc_idx])
sc_order = np.argsort(scores_arr)
ax = fig.add_subplot(gs[:, 3])
im = ax.imshow(all_norm[sc_order], aspect='auto', cmap='RdBu_r',
               extent=[tax_ref[0], tax_ref[-1], len(pc_idx), 0],
               vmin=-1, vmax=1, interpolation='nearest')
boundary = int((labels[sc_order] == 0).sum())
ax.axhline(boundary, color='yellow', lw=1.8)
ax.axvline(0, color='w', lw=0.8, ls='--')
ax.set_xlabel('Time (s)')
ax.set_ylabel('All PCs sorted by score')
ax.set_title('All PCs (yellow=boundary)')
plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

fig.suptitle(f'PC clustering  –  {allr_key}  ({len(pc_idx)} PCs)', fontsize=12)
out1 = os.path.join(OUT_DIR, 'fig1_pc_clusters.png')
fig.savefig(out1, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out1}')

# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 – Speed cross-correlation
# ═══════════════════════════════════════════════════════════════════════════
print('\n-- Part 2: Speed xcorr --')

groups = {'Decreasing': dec_idx, 'Non-decr': nondec_idx}
gcolors = {'Decreasing': '#E53935', 'Non-decr': '#1E88E5'}

# ── Per-cell lags ─────────────────────────────────────────────────────────────
all_lags = {k: {} for k in sd.event_keys}
all_xc   = {k: {} for k in sd.event_keys}

for key in sd.event_keys:
    sp_mean, _, sp_tax = get_psth(speed_idx, key)
    for uid in pc_idx:
        pc_mean, _, tax = get_psth(uid, key)
        lag_ms, lags_ms, xc = xcorr_lag_ms(pc_mean, sp_mean, tax)
        all_lags[key][uid] = lag_ms
        all_xc[key][uid]   = (lags_ms, xc)

# ── Method A – mean of per-cell lags ─────────────────────────────────────────
method_a = {}
for gname, uid_list in groups.items():
    method_a[gname] = {}
    for key in sd.event_keys:
        lags = np.array([all_lags[key][u] for u in uid_list])
        method_a[gname][key] = (lags.mean(), lags.std() / np.sqrt(len(lags)))

# ── Method B – lag of group-avg (max-norm) PSTHs ─────────────────────────────
method_b    = {}
method_b_xc = {}

for gname, uid_list in groups.items():
    method_b[gname]    = {}
    method_b_xc[gname] = {}
    for key in sd.event_keys:
        sp_mean, _, sp_tax = get_psth(speed_idx, key)
        pc_rows = [norm_max(get_psth(uid, key)[0]) for uid in uid_list]
        if not pc_rows:
            method_b[gname][key] = np.nan
            continue
        grp_mean = np.mean(np.array(pc_rows), axis=0)
        sp_norm  = norm_max(sp_mean)
        lag_ms, lags_ms, xc = xcorr_lag_ms(grp_mean, sp_norm, sp_tax)
        method_b[gname][key]    = lag_ms
        method_b_xc[gname][key] = (lags_ms, xc, grp_mean, sp_norm, sp_tax)

# ── Figure 2: bar charts (both methods) ──────────────────────────────────────
n_cond = len(sd.event_keys)
x      = np.arange(n_cond)
bar_w  = 0.32

fig, axes = plt.subplots(1, 2, figsize=(max(10, n_cond * 2.4), 5))

for ax, (title, get_val, get_err) in zip(axes, [
    ('Method A: mean of per-cell lags',
     lambda g, k: method_a[g][k][0],
     lambda g, k: method_a[g][k][1]),
    ('Method B: lag of group-avg PSTH',
     lambda g, k: method_b[g][k],
     lambda g, k: 0),
]):
    for gi, (gname, _) in enumerate(groups.items()):
        offset = (gi - 0.5) * bar_w
        vals = [get_val(gname, k) for k in sd.event_keys]
        errs = [get_err(gname, k) for k in sd.event_keys]
        ax.bar(x + offset, vals, bar_w, yerr=errs,
               color=gcolors[gname], alpha=0.85, label=gname,
               capsize=4, error_kw={'lw': 1.5})
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(sd.event_keys, rotation=35, ha='right')
    ax.set_ylabel('Lag (ms)  [+ = PC delayed vs speed]')
    ax.set_title(title)
    ax.legend()

plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'fig2_xcorr_lags.png')
fig.savefig(out2, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out2}')

# ── Figure 3: xcorr curves per group × condition ──────────────────────────────
n_rows = len(groups)
fig, axes = plt.subplots(n_rows, n_cond,
                         figsize=(max(4*n_cond, 8), 3.8*n_rows),
                         squeeze=False, sharey='row')

for ri, (gname, _) in enumerate(groups.items()):
    for ci, key in enumerate(sd.event_keys):
        ax = axes[ri][ci]
        if key not in method_b_xc.get(gname, {}):
            ax.axis('off'); continue
        lags_ms, xc, grp_mean, sp_norm, tax = method_b_xc[gname][key]
        lag_ms = method_b[gname][key]
        ax.plot(lags_ms, xc, color=gcolors[gname], lw=1.5)
        ax.axvline(0,      color='gray', lw=0.8, ls='--')
        ax.axvline(lag_ms, color='k',   lw=1.0,
                   label=f'{lag_ms:.0f} ms')
        ax.set_title(f'{gname} · {key}', fontsize=8)
        ax.set_xlabel('Lag (ms)')
        if ci == 0: ax.set_ylabel('Norm xcorr')
        ax.legend(fontsize=7)

fig.suptitle('Method B: xcorr curves', fontsize=11)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'fig3_xcorr_curves.png')
fig.savefig(out3, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out3}')

# ── Figure 4: group PSTH vs speed ─────────────────────────────────────────────
fig, axes = plt.subplots(n_rows, n_cond,
                         figsize=(max(4*n_cond, 8), 3.8*n_rows),
                         squeeze=False)

for ri, (gname, _) in enumerate(groups.items()):
    for ci, key in enumerate(sd.event_keys):
        ax = axes[ri][ci]
        if key not in method_b_xc.get(gname, {}):
            ax.axis('off'); continue
        _, _, grp_mean, sp_norm, tax = method_b_xc[gname][key]
        lag_ms = method_b[gname][key]
        ax.plot(tax, grp_mean, color=gcolors[gname], lw=2, label='PC group')
        ax.plot(tax, sp_norm,  color='#FF9800', lw=1.5, ls='--', label='Speed')
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_title(f'{gname} · {key}  lag={lag_ms:.0f} ms', fontsize=8)
        ax.set_xlabel('Time (s)')
        if ci == 0: ax.set_ylabel('Norm (max-norm)')
        if ri == 0 and ci == 0: ax.legend(fontsize=8)

fig.suptitle('Group-avg PSTH vs Speed (max-normalised)', fontsize=11)
plt.tight_layout()
out4 = os.path.join(OUT_DIR, 'fig4_psth_vs_speed.png')
fig.savefig(out4, bbox_inches='tight')
plt.close(fig)
print(f'  Saved {out4}')

# ── Text summary ──────────────────────────────────────────────────────────────
print('\n-- Summary --')
header = f"{'Condition':<12} {'Group':<14} {'n':>4}  {'A_mean':>8}  {'A_sem':>6}  {'B':>8}"
print(header)
print('-' * len(header))
for key in sd.event_keys:
    for gname, uid_list in groups.items():
        am, ae = method_a[gname][key]
        bm = method_b[gname][key]
        print(f'{key:<12} {gname:<14} {len(uid_list):>4}  '
              f'{am:>8.1f}  {ae:>6.1f}  {bm:>8.1f}')

print(f'\nAll figures saved to: {OUT_DIR}')
