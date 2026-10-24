"""Causal per-trajectory filtering for the KUKA KR300 identification benchmark.

Spec implemented here (user instruction, 2026-10-24):
  * Split: first 11 recordings (chronological) = train, last recording = test.
    Each recording holds 6 trajectories of ~60.6 s, separated by homing dwells.
    Segmentation cuts at the midpoints of the homing dwell intervals, so each
    trajectory is filtered independently (no cross-trajectory filter state).
  * Position channel: secondary encoder for joints 1-3, motor resolver for 4-6
    (same choice as the official MASTER_filter_and_combine_datasets.m).
    Velocity channel: same joint selection from qd_se_meas / qd_mot_meas.
  * Units: deg -> rad, deg/s -> rad/s, deg/s^2 -> rad/s^2 (factor pi/180).
    Torque stays in Nm (motor torque, as measured).
  * Filter: 4th-order causal Butterworth low-pass, fc = 2 Hz, fs = 250 Hz,
    applied with sosfilt and steady-state initial conditions (sosfilt_zi * x0)
    per trajectory and per channel. This deliberately replaces the official
    filtfilt (zero-phase, non-causal) so no future samples leak in.
  * Acceleration: causal backward difference of the filtered velocity at
    250 Hz (uses only the current and previous sample).
  * Downsampling: every 25th sample -> 10 Hz, per trajectory, after filtering.

Output: work/filtered_causal.npz + work/filtered_causal_meta.json
"""
import glob
import json
import os
import re

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, sosfilt, sosfilt_zi

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw_data")
OUT = os.path.dirname(os.path.abspath(__file__))
DEG = np.pi / 180.0
FS = 250.0
FC = 2.0
DS = 25  # 250 Hz -> 10 Hz


def chrono(path):
    m = re.search(r"(\d+)H_(\d+)M", os.path.basename(path))
    return int(m.group(1)) * 60 + int(m.group(2))


def segment_boundaries(q_ref, qd_ref):
    """Return cut indices splitting one recording into 6 trajectories.

    A dwell is a stretch where every joint is (numerically) at home and
    (numerically) still. Cuts are placed at dwell midpoints, so every segment
    starts and ends inside a dwell -- the filter then starts from a constant
    signal, which is exactly the steady-state initialisation used below.
    """
    still = (np.max(np.abs(qd_ref), axis=0) < 0.1) & (
        np.max(np.abs(q_ref), axis=0) < 0.01
    )
    groups = []
    idx = np.where(still)[0]
    if len(idx) == 0:
        return [0, q_ref.shape[1]]
    s = p = idx[0]
    for i in idx[1:]:
        if i - p > 1:
            groups.append((s, p))
            s = i
        p = i
    groups.append((s, p))
    # Keep only dwells longer than 1 s (250 samples); there should be 7:
    # start dwell, 5 internal dwells, end dwell.
    groups = [(a, b) for a, b in groups if b - a > 250]
    cuts = [0]
    for a, b in groups[1:-1]:
        cuts.append((a + b) // 2)
    cuts.append(q_ref.shape[1])
    return cuts


def main():
    files = sorted(glob.glob(os.path.join(BASE, "*.mat")), key=chrono)
    assert len(files) == 12, f"expected 12 recordings, got {len(files)}"
    sos = butter(4, FC, btype="lowpass", fs=FS, output="sos")

    def filt_causal(x):
        # x: (n_chan, n_time) at 250 Hz
        out = np.empty_like(x, dtype=np.float64)
        for c in range(x.shape[0]):
            zi = sosfilt_zi(sos) * x[c, 0]
            out[c], _ = sosfilt(sos, x[c], zi=zi)
        return out

    store = {"train": [], "test": []}  # list of dicts, one per trajectory
    for fi, f in enumerate(files):
        split = "train" if fi < 11 else "test"
        d = loadmat(f)
        assert int(d["recording_ok"].ravel()[0]) == 1, f
        q_deg = np.vstack([d["q_se_meas"][:3], d["q_mot_meas"][3:]])
        qd_deg = np.vstack([d["qd_se_meas"][:3], d["qd_mot_meas"][3:]])
        tau = d["tau_meas"].astype(np.float64)
        cuts = segment_boundaries(d["q_ref"], d["qd_ref"])
        assert len(cuts) == 7, (os.path.basename(f), cuts)
        for si in range(6):
            a, b = cuts[si], cuts[si + 1]
            q_f = filt_causal(q_deg[:, a:b] * DEG)
            qd_f = filt_causal(qd_deg[:, a:b] * DEG)
            # Causal backward difference at 250 Hz, then downsample together.
            qdd_f = np.zeros_like(qd_f)
            qdd_f[:, 1:] = (qd_f[:, 1:] - qd_f[:, :-1]) / (1.0 / FS)
            tau_f = filt_causal(tau[:, a:b])
            seg = {
                "recording": os.path.basename(f),
                "traj_in_recording": si,
                "n_250": b - a,
                "q": q_f[:, ::DS],
                "qd": qd_f[:, ::DS],
                "qdd": qdd_f[:, ::DS],
                "tau": tau_f[:, ::DS],
            }
            store[split].append(seg)
            print(
                f"{split} {os.path.basename(f)} traj{si} "
                f"250Hz={b - a} 10Hz={seg['q'].shape[1]}"
            )

    os.makedirs(OUT, exist_ok=True)
    out = {}
    meta = {}
    for split in ("train", "test"):
        segs = store[split]
        for key in ("q", "qd", "qdd", "tau"):
            out[f"{split}_{key}"] = np.concatenate([s[key] for s in segs], axis=1)
        out[f"{split}_seg_id"] = np.concatenate(
            [
                np.full(s["q"].shape[1], i, dtype=np.int32)
                for i, s in enumerate(segs)
            ]
        )
        meta[split] = {
            "n_trajectories": len(segs),
            "n_samples_10hz": int(out[f"{split}_q"].shape[1]),
            "recordings": [s["recording"] for s in segs[::6]],
            "samples_per_trajectory": [int(s["q"].shape[1]) for s in segs],
        }
        print(split, meta[split]["n_trajectories"], "trajectories,",
              meta[split]["n_samples_10hz"], "samples @10Hz")
    np.savez_compressed(os.path.join(OUT, "filtered_causal.npz"), **out)
    with open(os.path.join(OUT, "filtered_causal_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print("wrote", os.path.join(OUT, "filtered_causal.npz"))


if __name__ == "__main__":
    main()
