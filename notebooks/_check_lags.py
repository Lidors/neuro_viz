import sys; sys.path.insert(0, r'C:\Users\lidor\neuro_viz')
import numpy as np
from scipy.signal import correlate
from data_loader import SessionData

sd = SessionData(r'C:\Users\lidor\Nate_lab\code\MATLAB\analisys\Lidor_code\Maze\behavior\AA23_01\res23_05.mat')
pc_idx    = [i for i in range(sd.n_neural_units) if sd.cell_type_label[i] == 'PC']
speed_idx = next(i for i in sd.beh_unit_indices if sd.cell_type_label[i] == 'Speed')

def norm_peak(a):
    a = a - a.min(); mx = a.max(); return a/mx if mx>0 else a
def norm_max(a):
    mx = np.abs(a).max(); return a/mx if mx>0 else a
def get_lag(a, b, tax, ml=0.75):
    dt=float(tax[1]-tax[0]); n=len(a)
    az=a-a.mean(); bz=b-b.mean()
    xc=correlate(az,bz,mode='full')
    ls=np.arange(-(n-1),n)*dt
    mask=np.abs(ls)<=ml
    xc_w=xc[mask]; lw=ls[mask]
    nr=np.sqrt(np.dot(az,az)*np.dot(bz,bz))
    xc_w=xc_w/nr if nr>0 else xc_w
    return float(lw[np.argmax(np.abs(xc_w))])*1000

key = next(k for k in sd.event_keys if 'allr' in k.lower())
sp,_,tax = sd.get_mean_psth(speed_idx, key, 3, 1.5)
diffs = []
for uid in pc_idx:
    m,_,_ = sd.get_mean_psth(uid, key, 3, 1.5)
    lpk = get_lag(norm_peak(m), norm_peak(sp), tax)
    lmx = get_lag(norm_max(m),  norm_max(sp),  tax)
    diffs.append(abs(lpk - lmx))
    if abs(lpk - lmx) > 1.0:
        print(f'  uid={uid}  peak_lag={lpk:.1f}  max_lag={lmx:.1f}  diff={lpk-lmx:.1f} ms')
print(f'Max diff: {max(diffs):.6f} ms')
