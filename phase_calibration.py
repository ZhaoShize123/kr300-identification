"""Phase calibration for the KR300 identification pipeline.

Method (frozen here, before any test-set use):
  * Work at 250 Hz from the raw recordings, per trajectory, with the same
    causal 4th-order Butterworth (2 Hz) filter as filter_causal.py.
  * For candidate lag L in {0,1,2,3,4,5} samples at 250 Hz (= 0..20 ms),
    shift joint j's acceleration by L samples (past only: qdd(t-L) predicts
    tau(t); no future acceleration is ever used), then decimate by 25 to
    10 Hz. All other signals are undecimated in the same way with L=0.
  * For each joint j and lag L, fit a per-row model
        tau_j = Y_j(q, qd, qdd_lagged) @ pi_j  +  friction_j(qd_j)
                + (hydraulic Fourier basis in q2, joint 2 only)
    where Y is pinocchio's joint-torque regressor (6x60) for the KR300 MDH
    model, pi_j is row-specific (this absorbs the unknown per-row gear
    scaling 1/u_j, which is exactly why the final joint model needs the
    alternating fit in identify.py), and friction is
        f0 + fv*qd + fc*tanh(20*qd) + fa*tanh(2*qd)   (qd in meas. rad/s).
    Fitting is ridge least squares (lambda = 1e-3 * mean diagonal of X^T X).
  * Cross-validation is by trajectory: train trajectories are split into
    5 folds by trajectory index mod 5 (never by sample, so no trajectory
    leaks across a fold boundary). The lag minimising mean CV RMSE per
    joint is selected. Test data are not touched.
  * For speed, phase CV uses every 4th 10-Hz sample (~2.5 Hz effective
    sampling of the CV, still ~10k train samples).

Output: work/phase_calibration.json (per-joint CV RMSE table + best lag).
"""
import glob
import json
import os
import re
import sys

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfilt_zi

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kr300_model import build_model, meas_to_dh  # noqa: E402

import pinocchio as pin  # noqa: E402

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_data")
OUT = os.path.dirname(os.path.abspath(__file__))
DEG = np.pi / 180.0
FS = 250.0
DS = 25


def chrono(path):
    m = re.search(r"(\d+)H_(\d+)M", os.path.basename(path))
    return int(m.group(1)) * 60 + int(m.group(2))


def segment_boundaries(q_ref, qd_ref):
    still = (np.max(np.abs(qd_ref), axis=0) < 0.1) & (
        np.max(np.abs(q_ref), axis=0) < 0.01
    )
    idx = np.where(still)[0]
    groups = []
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            groups.append((s, p))
            s = i
        p = i
    groups.append((s, p))
    groups = [(a, b) for a, b in groups if b - a > 250]
    cuts = [0]
    for a, b in groups[1:-1]:
        cuts.append((a + b) // 2)
    cuts.append(q_ref.shape[1])
    return cuts


def filt_causal(x, sos):
    out = np.empty_like(x, dtype=np.float64)
    for c in range(x.shape[0]):
        zi = sosfilt_zi(sos) * x[c, 0]
        out[c], _ = sosfilt(sos, x[c], zi=zi)
    return out


def build_train_10hz():
    """Return (q, qd, qdd_lagged[L], tau, seg_id) at 10 Hz, train only.

    qdd_lagged[L] has shape (6 lags, 6 joints, N): lag L applied to that
    joint's 250-Hz acceleration before decimation.
    """
    files = sorted(glob.glob(os.path.join(BASE, "*.mat")), key=chrono)[:11]
    sos = butter(4, 2.0, btype="lowpass", fs=FS, output="sos")
    qs, qds, taus, segs = [], [], [], []
    qdd_by_lag = [[] for _ in range(6)]
    seg = 0
    for f in files:
        d = loadmat(f)
        q_deg = np.vstack([d["q_se_meas"][:3], d["q_mot_meas"][3:]])
        qd_deg = np.vstack([d["qd_se_meas"][:3], d["qd_mot_meas"][3:]])
        tau = d["tau_meas"].astype(np.float64)
        cuts = segment_boundaries(d["q_ref"], d["qd_ref"])
        for si in range(6):
            a, b = cuts[si], cuts[si + 1]
            q_f = filt_causal(q_deg[:, a:b] * DEG, sos)
            qd_f = filt_causal(qd_deg[:, a:b] * DEG, sos)
            tau_f = filt_causal(tau[:, a:b], sos)
            qdd_f = np.zeros_like(qd_f)
            qdd_f[:, 1:] = (qd_f[:, 1:] - qd_f[:, :-1]) / (1.0 / FS)
            qs.append(q_f[:, ::DS])
            qds.append(qd_f[:, ::DS])
            taus.append(tau_f[:, ::DS])
            segs.append(np.full(q_f[:, ::DS].shape[1], seg, dtype=np.int32))
            for lag in range(6):
                shifted = np.empty_like(qdd_f)
                if lag == 0:
                    shifted[:] = qdd_f
                else:
                    shifted[:, :lag] = qdd_f[:, :1]
                    shifted[:, lag:] = qdd_f[:, :-lag]
                qdd_by_lag[lag].append(shifted[:, ::DS])
            seg += 1
    q = np.concatenate(qs, axis=1)
    qd = np.concatenate(qds, axis=1)
    tau = np.concatenate(taus, axis=1)
    seg_id = np.concatenate(segs)
    qdd_lag = np.stack(
        [np.concatenate(qdd_by_lag[lag], axis=1) for lag in range(6)]
    )  # (lag, joint, N)
    return q, qd, qdd_lag, tau, seg_id


def friction_basis(qd_j):
    return np.stack(
        [np.ones_like(qd_j), qd_j, np.tanh(20.0 * qd_j), np.tanh(2.0 * qd_j)],
        axis=1,
    )


def hyd_basis(q2):
    return np.stack(
        [np.sin(q2), np.cos(q2), np.sin(2 * q2), np.cos(2 * q2)], axis=1
    )


def main():
    q, qd, qdd_lag, tau, seg_id = build_train_10hz()
    print("train 10Hz shapes", q.shape, qdd_lag.shape)
    # Subsample for CV speed: every 4th sample.
    sub = np.arange(0, q.shape[1], 4)
    q_s, qd_s, tau_s, seg_s = q[:, sub], qd[:, sub], tau[:, sub], seg_id[sub]
    qdd_s = qdd_lag[:, :, sub]  # (lag, joint, Nsub)
    folds = seg_s % 5

    model = build_model()
    data = model.createData()
    q_dh = meas_to_dh(q_s.T).T
    qd_dh = (qd_s.T * np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])).T
    # qdd_dh for lag 0, per joint (sign flips as for qd)
    sign = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])

    results = {}
    for j in range(6):
        results[f"joint{j+1}"] = {}
        for lag in range(6):
            # Build regressor rows for this (joint, lag): qdd of joint j is
            # lagged, all other joints' qdd are lag-0.
            qdd_dh = np.empty_like(q_s)
            for k in range(6):
                use_lag = lag if k == j else 0
                qdd_dh[k] = sign[k] * qdd_s[use_lag, k]
            # Pinocchio regressor, sample by sample (C++ call, fast).
            Y = np.empty((q_s.shape[1], 6, 60))
            for n in range(q_s.shape[1]):
                Y[n] = pin.computeJointTorqueRegressor(
                    model, data, q_dh[:, n], qd_dh[:, n], qdd_dh[:, n]
                )
            Yj = Y[:, j, :]  # (N, 60)
            X = np.concatenate(
                [Yj, friction_basis(qd_s[j]),
                 hyd_basis(q_s[1]) if j == 1 else np.zeros((q_s.shape[1], 0))],
                axis=1,
            )
            y = tau_s[j]
            rmses = []
            for fold in range(5):
                tr = folds != fold
                te = folds == fold
                lam = 1e-3 * np.mean(np.sum(X[tr] ** 2, axis=0))
                A = X[tr].T @ X[tr] + lam * np.eye(X.shape[1])
                coef = np.linalg.solve(A, X[tr].T @ y[tr])
                pred = X[te] @ coef
                rmses.append(float(np.sqrt(np.mean((y[te] - pred) ** 2))))
            results[f"joint{j+1}"][f"lag_{lag * 4}ms"] = {
                "cv_rmse_Nm": float(np.mean(rmses)),
                "fold_rmse_Nm": rmses,
            }
            print(f"joint{j+1} lag {lag * 4}ms CV RMSE {np.mean(rmses):.4f} Nm",
                  flush=True)
    best = {}
    for j in range(6):
        key = f"joint{j+1}"
        best_lag = min(results[key], key=lambda k: results[key][k]["cv_rmse_Nm"])
        best[key] = best_lag
    out = {"best_lag": best, "cv": results,
           "note": "Lag is applied to that joint's acceleration only, past-only, "
                   "at 250 Hz before decimation. CV folds are by trajectory."}
    with open(os.path.join(OUT, "phase_calibration.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    print("best lags:", best)


if __name__ == "__main__":
    main()
