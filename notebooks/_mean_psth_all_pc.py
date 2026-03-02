"""
Plot mean PSTH of all PC units for each condition (res23_05).
One row per condition, individual cells (thin) + group mean (thick).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from data_loader import SessionData

plt.rcParams.update({
    'figure.dpi': 140,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'font.size': 9,
})

RES_PATH   = r'C:\Users\lidor\Nate_lab\code\MATLAB\analisys\Lidor_code\Maze\behavior\AA23_01\res23_05.mat'
WINDOW_SEC = 1.5
SMOOTH_SIG = 3
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'output')
os.makedirs(OUT_DIR, exist_ok=True)

sd = SessionData(RES_PATH)
pc_idx = [i for i in range(sd.n_neural_units) if sd.cell_type_label[i] == 'PC']
print(f'PCs: {len(pc_idx)}  |  conditions: {sd.event_keys}')

conditions = sd.event_keys
n_cond = len(conditions)

fig, axes = plt.subplots(1, n_cond, figsize=(3.5 * n_cond, 4),
                         squeeze=False, sharey=True)

for ci, key in enumerate(conditions):
    ax = axes[0][ci]
    curves = []
    for uid in pc_idx:
        mean_fr, _, tax = sd.get_mean_psth(uid, key, SMOOTH_SIG, WINDOW_SEC)
        ax.plot(tax, mean_fr, color='#5C6BC0', alpha=0.30, lw=0.7)
        curves.append(mean_fr)

    grp = np.mean(np.array(curves), axis=0)
    sem = np.std(np.array(curves), axis=0) / np.sqrt(len(curves))
    ax.plot(tax, grp, color='#1A237E', lw=2.2, label=f'mean (n={len(pc_idx)})')
    ax.fill_between(tax, grp - sem, grp + sem, color='#1A237E', alpha=0.18)
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    ax.set_title(key, fontsize=9)
    ax.set_xlabel('Time from event (s)')
    if ci == 0:
        ax.set_ylabel('Firing rate (spk/s)')
        ax.legend(fontsize=8)

fig.suptitle(f'All PCs — mean PSTH  (n={len(pc_idx)}, smooth={SMOOTH_SIG} bins)',
             fontsize=11, y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, 'mean_psth_all_pc.png')
fig.savefig(out, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out}')
