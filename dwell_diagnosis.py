"""Dwell diagnosis: is dwell torque explained by tracking error / feedback,
by last-motion direction, or by time inside the dwell?

Raw 250 Hz recordings only (filtered 10 Hz file has no q_ref / tau_fb).
Dwell = contiguous run where max|qd_ref| < 0.1 deg/s AND max|q_ref| < 0.01 deg,
longer than 1 s. q_meas = secondary encoder joints 1-3, motor resolver 4-6
(same channel choice as the official scripts).
"""
import json, re
import numpy as np
from scipy.io import loadmat
from pathlib import Path

BASE = Path(__file__).parent / 'data' / 'raw_data'

def parse_time(name):
    m = re.search(r'(\d+)H_(\d+)M', name)
    return int(m.group(1)) * 60 + int(m.group(2))

files = sorted(BASE.glob('recording_*.mat'), key=lambda p: parse_time(p.name))

def groups_from_mask(mask, min_len):
    groups, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                groups.append((start, i))
            start = None
    if start is not None and len(mask) - start >= min_len:
        groups.append((start, len(mask)))
    return groups

all_rows = []   # one row per dwell group
pooled = {k: [] for k in ['tau', 'tau_fb', 'tau_ff', 'e', 'edot', 't_in_dwell']}
for f in files:
    m = loadmat(f)
    q_ref = m['q_ref']; qd_ref = m['qd_ref']
    q_se, q_mot = m['q_se_meas'], m['q_mot_meas']
    qd_se, qd_mot = m['qd_se_meas'], m['qd_mot_meas']
    q = np.vstack([q_se[:3], q_mot[3:]])
    qd = np.vstack([qd_se[:3], qd_mot[3:]])
    tau, tau_fb, tau_ff = m['tau_meas'], m['tau_fb_meas'], m['tau_ref_ff']
    # identity check tau == ff + fb ?
    resid = tau - (tau_ff + tau_fb)
    mask = (np.max(np.abs(qd_ref), axis=0) < 0.1) & (np.max(np.abs(q_ref), axis=0) < 0.01)
    for gi, (a, b) in enumerate(groups_from_mask(mask, 250)):
        sl = slice(a, b)
        e = q_ref[:, sl] - q[:, sl]          # deg
        edot = qd_ref[:, sl] - qd[:, sl]     # deg/s
        # last-motion direction: velocity sign in the 1 s window ending 0.5 s
        # before dwell start, and approach side (mean position just before dwell)
        row = {'file': f.name, 'group': gi, 'n': int(b - a),
               'tau_mean': tau[:, sl].mean(axis=1).tolist(),
               'tau_std': tau[:, sl].std(axis=1).tolist(),
               'tau_ff_mean': tau_ff[:, sl].mean(axis=1).tolist(),
               'tau_ff_std': tau_ff[:, sl].std(axis=1).tolist(),
               'tau_fb_mean': tau_fb[:, sl].mean(axis=1).tolist(),
               'e_mean': e.mean(axis=1).tolist(),
               'edot_mean': edot.mean(axis=1).tolist(),
               'identity_resid_rms': float(np.sqrt(np.mean(resid[:, sl] ** 2)))}
        if a >= 375:  # has a preceding motion window
            w = slice(a - 375, a - 125)  # 1.5-0.5 s before dwell
            row['approach_vel_sign'] = np.sign(qd[:, w].mean(axis=1)).tolist()
            row['approach_pos_mean'] = q[:, w].mean(axis=1).tolist()
            row['approach_pos_sign'] = np.sign(q[:, w].mean(axis=1)).tolist()
        else:
            row['approach_vel_sign'] = [0] * 6
            row['approach_pos_mean'] = [0.0] * 6
            row['approach_pos_sign'] = [0] * 6
        # within-dwell slope of tau vs time (Nm per second), per joint
        t = np.arange(b - a) / 250.0
        row['tau_slope_per_s'] = [float(np.polyfit(t, tau[j, sl], 1)[0]) for j in range(6)]
        all_rows.append(row)
        # pooled dwell samples (all groups, raw 250 Hz)
        pooled['tau'].append(tau[:, sl]); pooled['tau_fb'].append(tau_fb[:, sl])
        pooled['tau_ff'].append(tau_ff[:, sl]); pooled['e'].append(e)
        pooled['edot'].append(edot)
        pooled['t_in_dwell'].append(np.tile(t, (6, 1)))

P = {k: np.concatenate(v, axis=1) for k, v in pooled.items()}
print(f'dwell groups: {len(all_rows)}, pooled dwell samples: {P["tau"].shape[1]} '
      f'({P["tau"].shape[1] / 250:.0f} s)')
print(f'identity tau-(ff+fb) RMS over dwell: '
      f'{np.sqrt(np.mean([r["identity_resid_rms"] ** 2 for r in all_rows])):.4f} Nm')

def corr(x, y):
    return float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float('nan')

def ols_r2(X, y):
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    pred = X1 @ beta
    ss = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(ss), beta.tolist()

print('\n--- pooled within/all dwell samples, per joint: corr(tau, .) ---')
for j in range(6):
    tau, fb, ff, e, edot, td = P['tau'][j], P['tau_fb'][j], P['tau_ff'][j], P['e'][j], P['edot'][j], P['t_in_dwell'][j]
    r2, beta = ols_r2(np.column_stack([e, edot]), tau)
    r2fb, _ = ols_r2(np.column_stack([e, edot]), fb)
    print(f'J{j+1}: corr tau~e {corr(tau, e):+.3f}  tau~edot {corr(tau, edot):+.3f}  '
          f'tau~tau_fb {corr(tau, fb):+.3f}  tau~t_in_dwell {corr(tau, td):+.3f}  '
          f'tau_ff std {ff.std():.3f} Nm | OLS tau~e+edot R2 {r2:.3f} (fb R2 {r2fb:.3f})')

print('\n--- group means (between-dwell): corr of group-mean tau with group-mean e ---')
G = {k: np.array([r[k] for r in all_rows]) for k in
     ['tau_mean', 'e_mean', 'edot_mean', 'tau_fb_mean']}
for j in range(6):
    print(f'J{j+1}: corr(group tau, group e) {corr(G["tau_mean"][:, j], G["e_mean"][:, j]):+.3f}  '
          f'corr(group tau, group tau_fb) {corr(G["tau_mean"][:, j], G["tau_fb_mean"][:, j]):+.3f}  '
          f'group tau std {G["tau_mean"][:, j].std():.2f} Nm, group e std {G["e_mean"][:, j].std():.4f} deg')

print('\n--- last-motion direction: group-mean tau split by approach position sign ---')
for j in range(6):
    sgn = np.array([r['approach_pos_sign'][j] for r in all_rows])
    tm = G['tau_mean'][:, j]
    pos, neg = tm[sgn > 0], tm[sgn < 0]
    if len(pos) and len(neg):
        print(f'J{j+1}: approach+ n={len(pos)} mean {pos.mean():+.2f} Nm | approach- n={len(neg)} '
              f'mean {neg.mean():+.2f} Nm | diff {pos.mean() - neg.mean():+.2f} Nm')
    sgnv = np.array([r['approach_vel_sign'][j] for r in all_rows])
    pos, neg = tm[sgnv > 0], tm[sgnv < 0]
    if len(pos) and len(neg):
        print(f'     vel-sign: + n={len(pos)} {pos.mean():+.2f} | - n={len(neg)} {neg.mean():+.2f} '
              f'| diff {pos.mean() - neg.mean():+.2f} Nm')

print('\n--- within-dwell drift: tau slope (Nm/s), median across groups ---')
S = np.array([r['tau_slope_per_s'] for r in all_rows])
for j in range(6):
    print(f'J{j+1}: median slope {np.median(S[:, j]):+.3f} Nm/s, mean {S[:, j].mean():+.3f}, '
          f'std {S[:, j].std():.3f}')

json.dump({'groups': all_rows}, open(Path(__file__).parent / 'dwell_diagnosis.json', 'w'), indent=1)
print('\nsaved work/dwell_diagnosis.json')
