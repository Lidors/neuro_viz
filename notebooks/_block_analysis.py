"""
_block_analysis.py  –  Full PC × Speed analysis pipeline.

Runs in priority order:
  1.  Block PSTH amplitude vs speed amplitude (Pearson r per cell)
  2.  Block amplitude and xcorr lag trajectory over time
  3a. Reward vs No-reward FR comparison
  3b. Port selectivity (port1R vs port2R)
  3c. Speed-corrected R vs NR
  3e. Population speed decoding: early vs late trials
  3f. CS-SS pair PSTH comparison
  3g. Single-trial heatmap stability + CV over blocks
  3h. PCA of population PSTH

Outputs saved to neuro_viz/notebooks/output/block_analysis/
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import correlate
from scipy.stats import pearsonr, ttest_ind, sem
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

from data_loader import SessionData
from analysis import block_amplitude, block_best_lag

# ── style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.dpi': 130, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
})

OUT_DIR = os.path.join(os.path.dirname(__file__), 'output', 'block_analysis')
os.makedirs(OUT_DIR, exist_ok=True)

# ── config ────────────────────────────────────────────────────────────────────
RES_PATH     = r'C:\Users\lidor\Nate_lab\code\MATLAB\analisys\Lidor_code\Maze\behavior\AA23_01\res23_05.mat'
WINDOW_SEC   = 1.5
SMOOTH_SIG   = 3
BLOCK_SIZE   = 10
PRE_WIN      = (-0.5,  0.0)
BASELINE_WIN = (-1.5, -0.5)
MAX_LAG_SEC  = 0.75

# ── load ──────────────────────────────────────────────────────────────────────
print('Loading session...')
sd = SessionData(RES_PATH)
fps = sd.fps

pc_idx    = [i for i in range(sd.n_neural_units) if sd.cell_type_label[i] == 'PC']
speed_idx = next(i for i in sd.beh_unit_indices if sd.cell_type_label[i] == 'Speed')
allr_key  = next(k for k in sd.event_keys if 'allr' in k.lower())
allnr_key = next((k for k in sd.event_keys if k.lower() == 'allnr'), None)
r_keys    = [k for k in sd.event_keys if 'nr' not in k.lower()]
nr_keys   = [k for k in sd.event_keys if 'nr'     in k.lower()]

print(f'  {len(pc_idx)} PCs | fps={fps:.1f} | events={sd.event_keys}')

# ── helpers ───────────────────────────────────────────────────────────────────
def norm_peak(a):
    a = a - a.min(); mx = a.max(); return a / mx if mx > 0 else a

def get_psth(uid, key):
    return sd.get_mean_psth(uid, key, SMOOTH_SIG, WINDOW_SEC)

def decrease_score(mean, tax):
    b = (tax >= BASELINE_WIN[0]) & (tax <= BASELINE_WIN[1])
    p = (tax >= PRE_WIN[0])      & (tax <= PRE_WIN[1])
    bm = float(mean[b].mean()) if b.any() else 1.0
    pm = float(mean[p].mean()) if p.any() else 1.0
    return pm / bm if bm != 0 else 1.0

def xcorr_lag_ms(a, b, tax, max_lag=MAX_LAG_SEC):
    dt = float(tax[1] - tax[0]); n = len(a)
    az = a - a.mean(); bz = b - b.mean()
    xc = correlate(az, bz, mode='full')
    ls = np.arange(-(n-1), n) * dt
    mask = np.abs(ls) <= max_lag
    xc_w = xc[mask]; lw = ls[mask]
    nr = np.sqrt(np.dot(az, az) * np.dot(bz, bz))
    xc_w = xc_w / nr if nr > 0 else xc_w
    return float(lw[np.argmax(np.abs(xc_w))]) * 1000.0

def pre_win_mean(block_mat, tax):
    """Mean of block PSTHs in the PRE_WIN window. Returns (n_blocks,)."""
    mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
    return block_mat[mask, :].mean(axis=0)

def errbar(ax, x, y_mean, y_sem, color, label='', lw=1.8, alpha=0.2):
    ax.plot(x, y_mean, color=color, lw=lw, label=label)
    ax.fill_between(x, y_mean - y_sem, y_mean + y_sem, color=color, alpha=alpha)

# ── cluster PCs ───────────────────────────────────────────────────────────────
pc_psths = {uid: get_psth(uid, allr_key) for uid in pc_idx}
scores   = np.array([decrease_score(pc_psths[u][0], pc_psths[u][2]) for u in pc_idx])

km     = KMeans(n_clusters=2, n_init=50, random_state=42)
labels = km.fit_predict(scores.reshape(-1, 1))
if scores[labels == 0].mean() > scores[labels == 1].mean():
    labels = 1 - labels

dec_idx    = [pc_idx[i] for i, l in enumerate(labels) if l == 0]
nondec_idx = [pc_idx[i] for i, l in enumerate(labels) if l == 1]
groups     = {'Decreasing': dec_idx, 'Non-decr': nondec_idx}
gcolors    = {'Decreasing': '#E53935', 'Non-decr': '#1E88E5'}
print(f'  Decreasing: {len(dec_idx)} | Non-decr: {len(nondec_idx)}')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 1: Block PC amplitude vs Speed amplitude  (Pearson r per cell)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[1] Block PC amplitude vs Speed amplitude...')

# r_vals[uid][key] = Pearson r between pre-win PC FR and pre-win speed across blocks
r_vals = {uid: {} for uid in pc_idx}

for key in sd.event_keys:
    sp_bm, tax = sd.compute_block_psth(speed_idx, key, BLOCK_SIZE, SMOOTH_SIG, WINDOW_SEC)
    sp_pre = pre_win_mean(sp_bm, tax)   # (n_blocks,)

    for uid in pc_idx:
        pc_bm, _ = sd.compute_block_psth(uid, key, BLOCK_SIZE, SMOOTH_SIG, WINDOW_SEC)
        n_b = min(pc_bm.shape[1], sp_bm.shape[1])
        pc_pre = pre_win_mean(pc_bm[:, :n_b], tax)
        sp_pre_tr = sp_pre[:n_b]
        if np.std(pc_pre) < 1e-9 or np.std(sp_pre_tr) < 1e-9:
            r_vals[uid][key] = np.nan
        else:
            r_vals[uid][key], _ = pearsonr(pc_pre, sp_pre_tr)

n_cond = len(sd.event_keys)
fig1, axes1 = plt.subplots(1, n_cond, figsize=(3.2 * n_cond, 4), squeeze=False)

for ci, key in enumerate(sd.event_keys):
    ax = axes1[0][ci]
    for gname, uid_list in groups.items():
        rv = [r_vals[u][key] for u in uid_list if not np.isnan(r_vals[u][key])]
        x  = np.full(len(rv), list(groups).index(gname))
        ax.scatter(x + np.random.uniform(-0.12, 0.12, len(rv)),
                   rv, color=gcolors[gname], alpha=0.75, s=40, zorder=3)
        ax.plot([list(groups).index(gname) - 0.2, list(groups).index(gname) + 0.2],
                [np.nanmean(rv), np.nanmean(rv)],
                color=gcolors[gname], lw=2.5)
    # t-test
    rv_d  = [r_vals[u][key] for u in dec_idx    if not np.isnan(r_vals[u][key])]
    rv_nd = [r_vals[u][key] for u in nondec_idx if not np.isnan(r_vals[u][key])]
    if len(rv_d) > 1 and len(rv_nd) > 1:
        _, p = ttest_ind(rv_d, rv_nd)
        ax.set_title(f'{key}\np={p:.3f}', fontsize=9)
    else:
        ax.set_title(key, fontsize=9)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.set_xticks([0, 1]); ax.set_xticklabels(['Dec', 'Non'], fontsize=8)
    ax.set_ylabel('Pearson r (PC amp vs Speed amp)')
    if ci == 0:
        for gname, color in gcolors.items():
            ax.plot([], [], color=color, lw=3, label=gname)
        ax.legend()

fig1.suptitle('Analysis 1: Block amplitude correlation with speed  (pre-event window)',
              fontsize=11)
plt.tight_layout()
fig1.savefig(os.path.join(OUT_DIR, 'an1_block_amp_vs_speed.png'), bbox_inches='tight')
plt.close(fig1)
print('  Saved an1_block_amp_vs_speed.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 2: Block amplitude & xcorr lag TRAJECTORY over time
# ─────────────────────────────────────────────────────────────────────────────
print('\n[2] Block amplitude and lag trajectory over time...')

# For each cell × condition: get per-block amplitude (vs mean PSTH) and
# xcorr lag (PC block PSTH vs Speed block PSTH for the SAME block)

def block_xcorr_lags(pc_bm, sp_bm, tax):
    """Per-block xcorr lag between PC and Speed block PSTHs. Returns (n_blocks,)."""
    n_b = min(pc_bm.shape[1], sp_bm.shape[1])
    lags = []
    for b in range(n_b):
        lags.append(xcorr_lag_ms(pc_bm[:, b], sp_bm[:, b], tax))
    return np.array(lags)

# Collect per-cell trajectories: amp_traj[uid][key] = (n_blocks,), lag_traj[uid][key] = (n_blocks,)
amp_traj = {uid: {} for uid in pc_idx}
lag_traj = {uid: {} for uid in pc_idx}

for key in sd.event_keys:
    sp_bm, tax = sd.compute_block_psth(speed_idx, key, BLOCK_SIZE, SMOOTH_SIG, WINDOW_SEC)
    for uid in pc_idx:
        pc_bm, _ = sd.compute_block_psth(uid, key, BLOCK_SIZE, SMOOTH_SIG, WINDOW_SEC)
        pc_mean, _, _ = get_psth(uid, key)
        n_b = min(pc_bm.shape[1], sp_bm.shape[1])
        amp_traj[uid][key] = block_amplitude(pc_bm[:, :n_b], pc_mean)
        lag_traj[uid][key] = block_xcorr_lags(pc_bm[:, :n_b], sp_bm[:, :n_b], tax)

# Plot: mean ± SEM across cells in each group, vs block number
fig2, axes2 = plt.subplots(2, n_cond, figsize=(3.2 * n_cond, 7), squeeze=False,
                            sharex='col')

for ci, key in enumerate(sd.event_keys):
    # find common block count
    n_b_all = [amp_traj[uid][key].shape[0] for uid in pc_idx]
    n_b = min(n_b_all)
    blocks = np.arange(1, n_b + 1)

    for gname, uid_list in groups.items():
        color = gcolors[gname]

        # amplitude
        amp_mat = np.array([amp_traj[uid][key][:n_b] for uid in uid_list])
        errbar(axes2[0][ci], blocks, amp_mat.mean(0),
               amp_mat.std(0) / np.sqrt(len(uid_list)), color, gname)

        # xcorr lag
        lag_mat = np.array([lag_traj[uid][key][:n_b] for uid in uid_list])
        errbar(axes2[1][ci], blocks, lag_mat.mean(0),
               lag_mat.std(0) / np.sqrt(len(uid_list)), color)

    axes2[0][ci].axhline(1, color='gray', lw=0.8, ls='--')   # amplitude=1 → mean
    axes2[1][ci].axhline(0, color='gray', lw=0.8, ls='--')
    axes2[0][ci].set_title(key, fontsize=9)
    axes2[1][ci].set_xlabel('Block #')
    if ci == 0:
        axes2[0][ci].set_ylabel('Block amplitude (norm)')
        axes2[1][ci].set_ylabel('PC vs Speed lag (ms)')
        axes2[0][ci].legend()

fig2.suptitle('Analysis 2: Block amplitude & lag trajectory over time', fontsize=11)
plt.tight_layout()
fig2.savefig(os.path.join(OUT_DIR, 'an2_block_trajectory.png'), bbox_inches='tight')
plt.close(fig2)
print('  Saved an2_block_trajectory.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3a: Reward vs No-reward FR comparison
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3a] Reward vs No-reward FR...')

if allnr_key:
    r_key_all  = allr_key
    nr_key_all = allnr_key

    pre_fr = {uid: {} for uid in pc_idx}
    for uid in pc_idx:
        for key in [r_key_all, nr_key_all]:
            aligned, tax = sd.get_heatmap(uid, key, WINDOW_SEC)
            mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
            valid = ~np.any(np.isnan(aligned), axis=1)
            pre_fr[uid][key] = float(aligned[valid][:, mask].mean())

    fig3a, axes3a = plt.subplots(1, 2, figsize=(9, 4))

    # scatter per cell
    ax = axes3a[0]
    for gname, uid_list in groups.items():
        fr_r  = [pre_fr[u][r_key_all]  for u in uid_list]
        fr_nr = [pre_fr[u][nr_key_all] for u in uid_list]
        ax.scatter(fr_nr, fr_r, color=gcolors[gname], alpha=0.8, s=50,
                   edgecolors='none', label=gname)
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5)
    ax.set_xlabel(f'Mean FR pre-event  [{nr_key_all}]  (Hz)')
    ax.set_ylabel(f'Mean FR pre-event  [{r_key_all}]  (Hz)')
    ax.set_title('Per-cell pre-event FR: R vs NR')
    ax.legend()

    # bar per group
    ax = axes3a[1]
    for gi, (gname, uid_list) in enumerate(groups.items()):
        fr_r  = np.array([pre_fr[u][r_key_all]  for u in uid_list])
        fr_nr = np.array([pre_fr[u][nr_key_all] for u in uid_list])
        x = np.array([gi * 2.5, gi * 2.5 + 1.0])
        ax.bar([x[0]], [fr_r.mean()],  yerr=[sem(fr_r)],
               color=gcolors[gname], alpha=0.9, width=0.8, capsize=4)
        ax.bar([x[1]], [fr_nr.mean()], yerr=[sem(fr_nr)],
               color=gcolors[gname], alpha=0.45, width=0.8, capsize=4)
        ax.text(x[0], fr_r.mean()  * 1.02, 'R',  ha='center', fontsize=8)
        ax.text(x[1], fr_nr.mean() * 1.02, 'NR', ha='center', fontsize=8)

    ax.set_xticks([0.5, 3.0])
    ax.set_xticklabels(list(groups.keys()))
    ax.set_ylabel('Mean FR in pre-event window (Hz)')
    ax.set_title('Group mean ± SEM')

    fig3a.suptitle('Analysis 3a: Pre-event FR  —  Reward vs No-reward', fontsize=11)
    plt.tight_layout()
    fig3a.savefig(os.path.join(OUT_DIR, 'an3a_R_vs_NR_FR.png'), bbox_inches='tight')
    plt.close(fig3a)
    print('  Saved an3a_R_vs_NR_FR.png')
else:
    print('  allNR key not found, skipping 3a')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3b: Port selectivity  (port1R vs port2R)
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3b] Port selectivity...')

p1r = next((k for k in sd.event_keys if k.lower() in ('port1r', 'p1r', 'r1')), None)
p2r = next((k for k in sd.event_keys if k.lower() in ('port2r', 'p2r', 'r2')), None)

if p1r and p2r:
    port_amp = {uid: {} for uid in pc_idx}
    for uid in pc_idx:
        for key in [p1r, p2r]:
            m, _, tax = get_psth(uid, key)
            mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
            port_amp[uid][key] = float(m[mask].mean())

    fig3b, ax3b = plt.subplots(figsize=(5, 5))
    for gname, uid_list in groups.items():
        a1 = [port_amp[u][p1r] for u in uid_list]
        a2 = [port_amp[u][p2r] for u in uid_list]
        ax3b.scatter(a1, a2, color=gcolors[gname], alpha=0.8, s=55,
                     edgecolors='none', label=gname)
    lo = min(ax3b.get_xlim()[0], ax3b.get_ylim()[0])
    hi = max(ax3b.get_xlim()[1], ax3b.get_ylim()[1])
    ax3b.plot([lo, hi], [lo, hi], 'k--', lw=0.8, alpha=0.5, label='identity')
    ax3b.set_xlabel(f'Mean pre-event FR  [{p1r}]  (Hz)')
    ax3b.set_ylabel(f'Mean pre-event FR  [{p2r}]  (Hz)')
    ax3b.set_title('Analysis 3b: Port selectivity')
    ax3b.legend()
    plt.tight_layout()
    fig3b.savefig(os.path.join(OUT_DIR, 'an3b_port_selectivity.png'), bbox_inches='tight')
    plt.close(fig3b)
    print('  Saved an3b_port_selectivity.png')
else:
    print(f'  port keys not found ({p1r}, {p2r}), skipping 3b')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3c: Speed-corrected R vs NR
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3c] Speed-corrected R vs NR...')

if allnr_key:
    # Trial-level pre-event speed and PC FR for R and NR
    def trial_pre_means(uid, key):
        """Returns (pc_pre, speed_pre) — both (n_valid_trials,)."""
        pc_h,  tax = sd.get_heatmap(uid,       key, WINDOW_SEC)
        sp_h,  _   = sd.get_heatmap(speed_idx, key, WINDOW_SEC)
        mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
        valid = ~(np.any(np.isnan(pc_h), axis=1) | np.any(np.isnan(sp_h), axis=1))
        return pc_h[valid][:, mask].mean(1), sp_h[valid][:, mask].mean(1)

    # Pool group-averaged data
    def group_trial_data(uid_list, key):
        pc_all, sp_all = [], []
        for uid in uid_list:
            pc, sp = trial_pre_means(uid, key)
            # z-score each cell before pooling so units are comparable
            mu = pc.mean(); sd_ = pc.std()
            pc_z = (pc - mu) / sd_ if sd_ > 0 else pc - mu
            pc_all.append(pc_z)
            sp_all.append(sp)
        # use first cell's speed (all same event → same speed values per trial)
        return np.mean(np.array(pc_all), axis=0), sp_all[0] if sp_all else np.array([])

    n_bins = 4   # speed quartiles

    fig3c, axes3c = plt.subplots(1, 2, figsize=(11, 4))

    # Left: speed distribution R vs NR
    ax = axes3c[0]
    _, sp_r  = trial_pre_means(dec_idx[0], allr_key)
    _, sp_nr = trial_pre_means(dec_idx[0], allnr_key)
    ax.hist(sp_r,  bins=20, alpha=0.6, color='#E53935', label=allr_key,  density=True)
    ax.hist(sp_nr, bins=20, alpha=0.6, color='steelblue', label=allnr_key, density=True)
    ax.set_xlabel('Mean speed in pre-event window (cm/s)')
    ax.set_ylabel('Density')
    ax.set_title('Speed distribution: R vs NR')
    ax.legend()

    # Right: within speed bins, compare PC FR R vs NR (decreasing group only)
    ax = axes3c[1]
    pc_r,  sp_r  = group_trial_data(dec_idx, allr_key)
    pc_nr, sp_nr = group_trial_data(dec_idx, allnr_key)

    # Bin edges from combined speed distribution
    all_sp  = np.concatenate([sp_r, sp_nr])
    bin_edges = np.nanpercentile(all_sp, np.linspace(0, 100, n_bins + 1))
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    fr_r_bins, fr_nr_bins, n_r_bins, n_nr_bins = [], [], [], []
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        m_r  = (sp_r  >= lo) & (sp_r  < hi)
        m_nr = (sp_nr >= lo) & (sp_nr < hi)
        fr_r_bins.append(pc_r[m_r].mean()   if m_r.any()  else np.nan)
        fr_nr_bins.append(pc_nr[m_nr].mean() if m_nr.any() else np.nan)
        n_r_bins.append(m_r.sum());  n_nr_bins.append(m_nr.sum())

    ax.plot(bin_centres, fr_r_bins,  'o-', color='#E53935', lw=2, label=allr_key)
    ax.plot(bin_centres, fr_nr_bins, 's--', color='steelblue', lw=2, label=allnr_key)
    ax.set_xlabel('Speed bin (cm/s)')
    ax.set_ylabel('Mean z-scored PC FR (pre-event)')
    ax.set_title('Decreasing PCs: FR vs Speed bin\n(speed-corrected R vs NR)')
    ax.legend()

    fig3c.suptitle('Analysis 3c: Speed-corrected reward signal', fontsize=11)
    plt.tight_layout()
    fig3c.savefig(os.path.join(OUT_DIR, 'an3c_speed_corrected.png'), bbox_inches='tight')
    plt.close(fig3c)
    print('  Saved an3c_speed_corrected.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3e: Population speed decoding — early vs late
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3e] Population speed decoding early vs late...')

def decode_r2(uid_list, key, alpha=1.0):
    """
    For the given unit list, build a (n_trials, n_units) feature matrix
    of pre-event mean FR and predict pre-event speed.
    Returns (r2_early, r2_late) using leave-one-out on each half.
    """
    # Build feature matrix
    heatmaps, sp_h, tax = [], None, None
    for uid in uid_list:
        h, t = sd.get_heatmap(uid, key, WINDOW_SEC)
        heatmaps.append(h); tax = t
    sp_h, _ = sd.get_heatmap(speed_idx, key, WINDOW_SEC)

    mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
    valid = ~np.any(np.isnan(sp_h), axis=1)
    for h in heatmaps:
        valid &= ~np.any(np.isnan(h), axis=1)

    X = np.column_stack([h[valid][:, mask].mean(1) for h in heatmaps])
    y = sp_h[valid][:, mask].mean(1)

    n = X.shape[0]
    if n < 8 or X.shape[1] < 1:
        return np.nan, np.nan

    half = n // 2
    X_e, y_e = X[:half], y[:half]
    X_l, y_l = X[half:], y[half:]

    def cv_r2(Xd, yd):
        scores = cross_val_score(Ridge(alpha=alpha), Xd, yd,
                                 cv=min(5, len(yd)), scoring='r2')
        return float(np.clip(scores, -1, 1).mean())

    return cv_r2(X_e, y_e), cv_r2(X_l, y_l)

fig3e, axes3e = plt.subplots(1, n_cond, figsize=(3.2 * n_cond, 4), squeeze=False)

for ci, key in enumerate(sd.event_keys):
    ax = axes3e[0][ci]
    x_pos = 0
    for gname, uid_list in groups.items():
        r2_e, r2_l = decode_r2(uid_list, key)
        ax.bar([x_pos],     [r2_e], color=gcolors[gname], alpha=0.5, width=0.7)
        ax.bar([x_pos + 1], [r2_l], color=gcolors[gname], alpha=1.0, width=0.7)
        ax.text(x_pos,     max(r2_e, 0) + 0.01, 'E', ha='center', fontsize=7)
        ax.text(x_pos + 1, max(r2_l, 0) + 0.01, 'L', ha='center', fontsize=7)
        x_pos += 3
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(list(groups.keys()), fontsize=8)
    ax.set_ylabel('Cross-val R²  (early E / late L)')
    ax.set_title(key, fontsize=9)
    if ci == 0:
        for gname, color in gcolors.items():
            ax.bar([], [], color=color, label=gname)
        ax.legend()

fig3e.suptitle('Analysis 3e: Population speed decoding  (early vs late trials)',
               fontsize=11)
plt.tight_layout()
fig3e.savefig(os.path.join(OUT_DIR, 'an3e_pop_decoding.png'), bbox_inches='tight')
plt.close(fig3e)
print('  Saved an3e_pop_decoding.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3f: CS-SS pair PSTH comparison
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3f] CS-SS pair analysis...')

pc_pairs = []
seen = set()
for uid, paired_uid in sd.pair_map.items():
    if paired_uid is not None and uid < sd.n_neural_units:
        pair_key = tuple(sorted([uid, paired_uid]))
        if pair_key not in seen:
            pc_pairs.append(pair_key)
            seen.add(pair_key)

# Keep only pairs where at least one member is a PC
pc_pairs = [(a, b) for a, b in pc_pairs
            if sd.cell_type_label[a] == 'PC' or sd.cell_type_label[b] == 'PC']

print(f'  PC-related pairs: {len(pc_pairs)}')

if pc_pairs:
    n_pairs = min(len(pc_pairs), 9)
    ncols   = 3
    nrows   = int(np.ceil(n_pairs / ncols))

    fig3f, axes3f = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.5 * nrows),
                                  squeeze=False)

    for pi, (a, b) in enumerate(pc_pairs[:n_pairs]):
        ax = axes3f[pi // ncols][pi % ncols]
        for uid, ls, lw in [(a, '-', 2), (b, '--', 1.5)]:
            m, s, tax = get_psth(uid, allr_key)
            ct = sd.cell_type_label[uid]
            ax.plot(tax, m, ls=ls, lw=lw, label=f'U{uid} ({ct})')
            ax.fill_between(tax, m-s, m+s, alpha=0.15)
        ax.axvline(0, color='gray', lw=0.8, ls='--')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('FR (Hz)')
        ax.set_title(f'Pair {pi+1}: U{a}({sd.cell_type_label[a]})–U{b}({sd.cell_type_label[b]})',
                     fontsize=8)
        ax.legend(fontsize=7)

    # Hide unused panels
    for pi in range(n_pairs, nrows * ncols):
        axes3f[pi // ncols][pi % ncols].axis('off')

    fig3f.suptitle(f'Analysis 3f: CS-SS pair PSTHs  [{allr_key}]', fontsize=11)
    plt.tight_layout()
    fig3f.savefig(os.path.join(OUT_DIR, 'an3f_cs_ss_pairs.png'), bbox_inches='tight')
    plt.close(fig3f)
    print('  Saved an3f_cs_ss_pairs.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3g: Single-trial heatmap stability + CV over blocks
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3g] Single-trial stability...')

fig3g = plt.figure(figsize=(14, 10))
gs = gridspec.GridSpec(2, 2, figure=fig3g, hspace=0.45, wspace=0.35)

for gi, (gname, uid_list) in enumerate(groups.items()):
    # Mean heatmap across cells in the group (trials × time)
    all_h = []
    for uid in uid_list:
        h, tax = sd.get_heatmap(uid, allr_key, WINDOW_SEC)
        valid = ~np.any(np.isnan(h), axis=1)
        h_v = h[valid]
        # normalize each trial by its mean FR to remove mean rate differences
        trial_mean = h_v.mean(axis=1, keepdims=True)
        h_n = h_v / (np.abs(trial_mean) + 1e-6)
        all_h.append(h_n)

    min_trials = min(x.shape[0] for x in all_h)
    group_h = np.nanmean(np.stack([x[:min_trials] for x in all_h]), axis=0)

    # Heatmap: rows = trials (ordered by time), cols = time
    ax_hm = fig3g.add_subplot(gs[0, gi])
    vmax  = np.nanpercentile(np.abs(group_h), 95)
    im = ax_hm.imshow(group_h, aspect='auto', cmap='RdBu_r',
                       vmin=-vmax, vmax=vmax,
                       extent=[tax[0], tax[-1], min_trials, 0],
                       interpolation='nearest')
    ax_hm.axvline(0, color='w', lw=0.8, ls='--')
    ax_hm.set_xlabel('Time from event (s)')
    ax_hm.set_ylabel('Trial # (chronological)')
    ax_hm.set_title(f'{gname}  n={len(uid_list)} cells')
    plt.colorbar(im, ax=ax_hm, fraction=0.03, pad=0.02)

    # CV over blocks
    ax_cv = fig3g.add_subplot(gs[1, gi])
    pre_mask = (tax >= PRE_WIN[0]) & (tax <= PRE_WIN[1])
    cv_per_block = []
    for uid in uid_list:
        h_uid, _ = sd.get_heatmap(uid, allr_key, WINDOW_SEC)
        valid = ~np.any(np.isnan(h_uid), axis=1)
        h_uid = h_uid[valid]
        pre_fr_trials = h_uid[:, pre_mask].mean(axis=1)  # (n_trials,)
        n_b = len(pre_fr_trials) // BLOCK_SIZE
        cvs = []
        for b in range(n_b):
            chunk = pre_fr_trials[b * BLOCK_SIZE:(b + 1) * BLOCK_SIZE]
            mu = chunk.mean()
            cvs.append(chunk.std() / (abs(mu) + 1e-6))
        cv_per_block.append(cvs)

    min_b = min(len(c) for c in cv_per_block)
    cv_mat = np.array([c[:min_b] for c in cv_per_block])
    errbar(ax_cv, np.arange(1, min_b + 1), cv_mat.mean(0),
           cv_mat.std(0) / np.sqrt(len(uid_list)), gcolors[gname])
    ax_cv.set_xlabel('Block #')
    ax_cv.set_ylabel('CV of pre-event FR')
    ax_cv.set_title(f'{gname} — trial variability over time')

fig3g.suptitle('Analysis 3g: Single-trial stability', fontsize=11)
fig3g.savefig(os.path.join(OUT_DIR, 'an3g_trial_stability.png'), bbox_inches='tight')
plt.close(fig3g)
print('  Saved an3g_trial_stability.png')

# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSIS 3h: PCA of PC population PSTH
# ─────────────────────────────────────────────────────────────────────────────
print('\n[3h] PCA of population PSTH...')

# Stack peak-normalised mean PSTHs: (n_pcs, n_time)
psth_mat = np.array([norm_peak(get_psth(uid, allr_key)[0]) for uid in pc_idx])
_, _, tax_ref = get_psth(pc_idx[0], allr_key)

n_comp = min(6, len(pc_idx))
pca    = PCA(n_components=n_comp)
scores_pca = pca.fit_transform(psth_mat)   # (n_pcs, n_comp)
comps      = pca.components_               # (n_comp, n_time)

fig3h = plt.figure(figsize=(15, 9))
gs_h  = gridspec.GridSpec(3, n_comp, figure=fig3h, hspace=0.55, wspace=0.35)

# Row 0: scree plot (just one panel) + first n_comp-1 PC traces
ax_scree = fig3h.add_subplot(gs_h[0, :2])
ax_scree.bar(np.arange(1, n_comp + 1), pca.explained_variance_ratio_ * 100,
             color='#607D8B')
ax_scree.set_xlabel('Component')
ax_scree.set_ylabel('Variance explained (%)')
ax_scree.set_title('Scree plot')

# Row 0 remaining: PC1 and PC2 traces vs speed
sp_mean, _, _ = get_psth(speed_idx, allr_key)
sp_norm = norm_peak(sp_mean)

for c in range(min(2, n_comp)):
    ax = fig3h.add_subplot(gs_h[0, 2 + c])
    ax.plot(tax_ref, comps[c], color='#37474F', lw=2,
            label=f'PC{c+1} ({pca.explained_variance_ratio_[c]*100:.1f}%)')
    ax.plot(tax_ref, sp_norm * comps[c].std() * 2,
            color='#FF9800', lw=1.5, ls='--', label='Speed (scaled)')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_xlabel('Time (s)')
    ax.set_title(f'PC{c+1}')
    ax.legend(fontsize=7)

# Row 1-2: scores scatter (PC1 vs PC2), colored by cluster and by decrease score
ax_s1 = fig3h.add_subplot(gs_h[1, :3])
for gname, uid_list in groups.items():
    idx_in_pc = [pc_idx.index(u) for u in uid_list]
    ax_s1.scatter(scores_pca[idx_in_pc, 0], scores_pca[idx_in_pc, 1],
                  color=gcolors[gname], alpha=0.8, s=60, edgecolors='none',
                  label=gname)
ax_s1.set_xlabel('PCA score  PC1')
ax_s1.set_ylabel('PCA score  PC2')
ax_s1.set_title('PCA scores colored by cluster')
ax_s1.legend()

ax_s2 = fig3h.add_subplot(gs_h[1, 3:])
sc = ax_s2.scatter(scores_pca[:, 0], scores_pca[:, 1],
                   c=scores, cmap='RdBu', alpha=0.85, s=60)
plt.colorbar(sc, ax=ax_s2, label='Decrease score')
ax_s2.set_xlabel('PCA score  PC1')
ax_s2.set_ylabel('PCA score  PC2')
ax_s2.set_title('PCA scores colored by decrease score')

# Row 2: PC1 score (per cell) vs decrease score scatter
ax_r = fig3h.add_subplot(gs_h[2, :3])
ax_r.scatter(scores, scores_pca[:, 0],
             c=[gcolors['Decreasing'] if l == 0 else gcolors['Non-decr']
                for l in labels],
             alpha=0.8, s=55, edgecolors='none')
ax_r.set_xlabel('Decrease score (pre/baseline)')
ax_r.set_ylabel('PCA PC1 score')
ax_r.set_title('Decrease score vs PC1 projection')
r_corr, p_corr = pearsonr(scores, scores_pca[:, 0])
ax_r.set_title(f'Decrease score vs PC1  (r={r_corr:.2f}, p={p_corr:.3f})')

fig3h.suptitle('Analysis 3h: PCA of population PSTH', fontsize=12)
fig3h.savefig(os.path.join(OUT_DIR, 'an3h_pca.png'), bbox_inches='tight')
plt.close(fig3h)
print('  Saved an3h_pca.png')

print(f'\nAll figures saved to: {OUT_DIR}')
