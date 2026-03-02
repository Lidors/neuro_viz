"""
Quick diagnostic: plot decreasing PC group PSTH vs Speed for all conditions.
Shows individual cells (thin lines) + group mean (thick), alongside speed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

from data_loader import SessionData

plt.rcParams.update({'figure.dpi': 140, 'axes.spines.top': False, 'axes.spines.right': False})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

RES_PATH     = r'C:\Users\lidor\Nate_lab\code\MATLAB\analisys\Lidor_code\Maze\behavior\AA23_01\res23_05.mat'
WINDOW_SEC   = 1.5
SMOOTH_SIG   = 3
BASELINE_WIN = (-1.5, -0.5)
PRE_WIN      = (-0.5,  0.0)

# ── Load ─────────────────────────────────────────────────────────────────────
sd = SessionData(RES_PATH)

pc_idx    = [i for i in range(sd.n_neural_units) if sd.cell_type_label[i] == 'PC']
speed_idx = next(i for i in sd.beh_unit_indices if sd.cell_type_label[i] == 'Speed')
allr_key  = next(k for k in sd.event_keys if 'allr' in k.lower())

def get_psth(uid, key):
    return sd.get_mean_psth(uid, key, SMOOTH_SIG, WINDOW_SEC)

def norm_peak(arr):
    arr = arr - arr.min()
    mx  = arr.max()
    return arr / mx if mx > 0 else arr

def norm_max(arr):
    mx = np.abs(arr).max()
    return arr / mx if mx > 0 else arr

def decrease_score(mean, tax):
    b = (tax >= BASELINE_WIN[0]) & (tax <= BASELINE_WIN[1])
    p = (tax >= PRE_WIN[0])      & (tax <= PRE_WIN[1])
    bm = float(mean[b].mean()) if b.any() else 1.0
    pm = float(mean[p].mean()) if p.any() else 1.0
    return pm / bm if bm != 0 else 1.0

# ── Cluster ──────────────────────────────────────────────────────────────────
pc_psths  = {uid: get_psth(uid, allr_key) for uid in pc_idx}
scores    = np.array([decrease_score(pc_psths[u][0], pc_psths[u][2]) for u in pc_idx])

km     = KMeans(n_clusters=2, n_init=50, random_state=42)
labels = km.fit_predict(scores.reshape(-1, 1))
if scores[labels==0].mean() > scores[labels==1].mean():
    labels = 1 - labels

dec_idx    = [pc_idx[i] for i, l in enumerate(labels) if l == 0]
nondec_idx = [pc_idx[i] for i, l in enumerate(labels) if l == 1]
print(f'Decreasing: {len(dec_idx)}  Non-decr: {len(nondec_idx)}')

# ── Figure: all conditions, decreasing PCs + speed ───────────────────────────
conditions = sd.event_keys
n_cond = len(conditions)

fig, axes = plt.subplots(2, n_cond, figsize=(3.5 * n_cond, 8), squeeze=False,
                         sharey='row')

for ci, key in enumerate(conditions):
    sp_mean, _, tax = get_psth(speed_idx, key)
    sp_norm = norm_peak(sp_mean)

    # ── Top row: individual cells + mean ────────────────────────────────────
    ax = axes[0][ci]
    cell_curves = []
    for uid in dec_idx:
        m, _, _ = get_psth(uid, key)
        mn = norm_peak(m)
        cell_curves.append(mn)
        ax.plot(tax, mn, color='#E53935', alpha=0.25, lw=0.8)

    # group mean
    grp = np.mean(np.array(cell_curves), axis=0)
    ax.plot(tax, grp, color='#B71C1C', lw=2.5, label='PC mean')

    # speed
    ax.plot(tax, sp_norm, color='#FF9800', lw=2, ls='--', label='Speed')
    ax.axvline(0, color='gray', lw=0.8, ls='--')

    ax.set_title(key, fontsize=9)
    ax.set_xlabel('Time from event (s)')
    if ci == 0:
        ax.set_ylabel('Norm FR / Speed (peak)')
        ax.legend(fontsize=8)

    # ── Bottom row: max-norm (for xcorr context) ─────────────────────────────
    ax2 = axes[1][ci]
    cell_curves_mx = []
    for uid in dec_idx:
        m, _, _ = get_psth(uid, key)
        mn = norm_max(m)
        cell_curves_mx.append(mn)
        ax2.plot(tax, mn, color='#E53935', alpha=0.25, lw=0.8)

    grp_mx = np.mean(np.array(cell_curves_mx), axis=0)
    sp_mx  = norm_max(sp_mean)
    ax2.plot(tax, grp_mx, color='#B71C1C', lw=2.5, label='PC mean (max-norm)')
    ax2.plot(tax, sp_mx,  color='#FF9800', lw=2,   ls='--', label='Speed (max-norm)')
    ax2.axvline(0, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel('Time from event (s)')
    if ci == 0:
        ax2.set_ylabel('Max-normalised')
        ax2.legend(fontsize=8)

axes[0][n_cond//2].set_title(
    f'Decreasing PCs (n={len(dec_idx)})  —  top: peak-norm  |  bottom: max-norm',
    fontsize=10, pad=8)

fig.suptitle('Decreasing PC group vs Speed — all conditions', fontsize=12, y=1.01)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'diag_dec_vs_speed.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')

# ── Also plot non-decreasing for comparison ───────────────────────────────────
fig2, axes2 = plt.subplots(1, n_cond, figsize=(3.5 * n_cond, 4), squeeze=False)

for ci, key in enumerate(conditions):
    sp_mean, _, tax = get_psth(speed_idx, key)
    sp_norm = norm_peak(sp_mean)
    ax = axes2[0][ci]

    for uid_list, color, lbl in [
        (dec_idx,    '#E53935', 'Decreasing'),
        (nondec_idx, '#1E88E5', 'Non-decr'),
    ]:
        rows = [norm_peak(get_psth(u, key)[0]) for u in uid_list]
        grp  = np.mean(np.array(rows), axis=0)
        ax.plot(tax, grp, color=color, lw=2, label=lbl)

    ax.plot(tax, sp_norm, color='#FF9800', lw=1.8, ls='--', label='Speed')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_title(key, fontsize=9)
    ax.set_xlabel('Time (s)')
    if ci == 0:
        ax.set_ylabel('Norm FR (peak)')
        ax.legend(fontsize=8)

fig2.suptitle('Both groups vs Speed (peak-norm)', fontsize=11)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, 'diag_both_groups_vs_speed.png')
fig2.savefig(out2, bbox_inches='tight')
plt.close(fig2)
print(f'Saved {out2}')

# ── Scatter plots ─────────────────────────────────────────────────────────────
# Note: peak-norm and max-norm give identical per-cell lags (both reduce to
# (y - mean(y))/const after zero-meaning in xcorr), so we use peak-norm throughout.
from scipy.signal import correlate

MAX_LAG_SEC = 0.75

def xcorr_lag_ms(a, b, tax, max_lag=MAX_LAG_SEC):
    dt     = float(tax[1] - tax[0])
    n      = len(a)
    a_zm   = a - a.mean()
    b_zm   = b - b.mean()
    xc     = correlate(a_zm, b_zm, mode='full')
    lags_s = np.arange(-(n-1), n) * dt
    mask   = np.abs(lags_s) <= max_lag
    xc_w   = xc[mask];  lw = lags_s[mask]
    nr     = np.sqrt(np.dot(a_zm, a_zm) * np.dot(b_zm, b_zm))
    xc_w   = xc_w / nr if nr > 0 else xc_w
    return float(lw[np.argmax(np.abs(xc_w))]) * 1000.0

group_info = [
    (dec_idx,    '#E53935', 'Decreasing'),
    (nondec_idx, '#1E88E5', 'Non-decr'),
]

# Pre-compute per-cell lags for all conditions
cell_lags = {}   # uid -> {key: lag_ms}
for uid in pc_idx:
    cell_lags[uid] = {}
    for key in conditions:
        sp_mean, _, tax = get_psth(speed_idx, key)
        m, _, _         = get_psth(uid, key)
        cell_lags[uid][key] = xcorr_lag_ms(norm_peak(m), norm_peak(sp_mean), tax)

# ── Scatter 1: decrease score vs lag (AllR) ───────────────────────────────────
fig3, axes3 = plt.subplots(1, n_cond, figsize=(3.5 * n_cond, 4), squeeze=False)

for ci, key in enumerate(conditions):
    ax = axes3[0][ci]
    for uid_list, color, lbl in group_info:
        sc = [scores[pc_idx.index(u)] for u in uid_list]
        lg = [cell_lags[u][key] for u in uid_list]
        ax.scatter(sc, lg, color=color, alpha=0.8, s=50, edgecolors='none', label=lbl)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(1, color='gray', lw=0.8, ls=':')   # score=1 (no change)
    ax.set_xlabel('Decrease score (pre/baseline)')
    ax.set_ylabel('Lag vs Speed (ms)')
    ax.set_title(key, fontsize=9)
    if ci == 0:
        ax.legend(fontsize=8)

fig3.suptitle('Decrease score vs Speed lag  (per cell, each condition)', fontsize=11)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, 'diag_score_vs_lag.png')
fig3.savefig(out3, bbox_inches='tight')
plt.close(fig3)
print(f'Saved {out3}')

# ── Scatter 2: reward lag vs no-reward lag per cell ───────────────────────────
reward_keys   = [k for k in conditions if 'nr' not in k.lower()]
noreward_keys = [k for k in conditions if 'nr' in k.lower()]

if reward_keys and noreward_keys:
    # Use AllR vs allNR (or first reward vs first no-reward)
    r_key  = next((k for k in reward_keys  if 'all' in k.lower()), reward_keys[0])
    nr_key = next((k for k in noreward_keys if 'all' in k.lower()), noreward_keys[0])

    fig4, ax4 = plt.subplots(figsize=(5, 5))
    lo, hi = -MAX_LAG_SEC*1000, MAX_LAG_SEC*1000

    for uid_list, color, lbl in group_info:
        lag_r  = [cell_lags[u][r_key]  for u in uid_list]
        lag_nr = [cell_lags[u][nr_key] for u in uid_list]
        ax4.scatter(lag_r, lag_nr, color=color, alpha=0.85,
                    s=55, edgecolors='w', linewidths=0.4, label=lbl)

    ax4.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, label='identity')
    ax4.axhline(0, color='gray', lw=0.5)
    ax4.axvline(0, color='gray', lw=0.5)
    ax4.set_xlim(lo, hi); ax4.set_ylim(lo, hi)
    ax4.set_aspect('equal')
    ax4.set_xlabel(f'Lag vs Speed — {r_key} (ms)')
    ax4.set_ylabel(f'Lag vs Speed — {nr_key} (ms)')
    ax4.set_title(f'Reward vs No-reward lag per cell\n({r_key} vs {nr_key})')
    ax4.legend(fontsize=8)

    plt.tight_layout()
    out4 = os.path.join(OUT_DIR, 'diag_lag_reward_vs_noreward.png')
    fig4.savefig(out4, bbox_inches='tight')
    plt.close(fig4)
    print(f'Saved {out4}')
