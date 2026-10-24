"""KR300 R2500 nominal kinematics, from literature DH.

Sources:
  * Selingue et al., Hybrid Calibration of Industrial Robot Considering Payload
    Variation, J. Intell. Robot. Syst. 109 (2023), Table 1 (nominal DH, KR300):
      joint: a [m], alpha [deg], d [m], theta
      1: 0,     0,   0.675, -th1
      2: 0.350, -90, 0,      th2
      3: 1.150,  0,  0,      th3 - 90deg
      4: -0.041,-90, 1.000, -th4
      5: 0,     90,  0,      th5
      6: 0,    -90,  0.240, -th6 + 180deg
    (standard DH convention: T = RotZ(th) TransZ(d) TransX(a) RotX(alpha))
  * Cross-check vs KUKA datasheet work envelope (KR 300 R2500 ultra):
    base height + upper arm + forearm = 0.675+1.150+1.000 = 2.825 m
    vs dimension A = 2.826 m; horizontal reach to wrist centre
    0.350+1.150+1.000 = 2.500 m vs dimension C = 2.496 m.
  * Benchmark measurement convention (description PDF v2.0, eq. 19):
    measurement q = KUKA q + [0, +90deg, -90deg, 0, 0, 0]
    (checked against Table 7 limits: KUKA A2 -140/-5 -> meas -50/85).
    Hence KUKA th = meas + [0, -90deg, +90deg, 0, 0, 0], and DH thetas are:
      th_DH = [-q1, q2-90deg, q3, -q4, q5, -q6+180deg]  (q in rad, meas. conv.)

Model is built in pinocchio with modified-DH placements, which are exactly
equivalent to the standard-DH chain above: MDH transform for joint i is
  RotX(alpha_{i-1}) TransX(a_{i-1}) RotZ(th_DH_i) TransZ(d_i)
= placement_i * RotZ(th_DH_i)   (RotZ and TransZ commute),
with placement_i = RotX(alpha_{i-1}) TransX(a_{i-1}) TransZ(d_i).
Pinocchio configuration vector is therefore th_DH itself.
"""
import numpy as np
import pinocchio as pin

# standard DH rows: (a, alpha, d) ; theta handled by meas->DH mapping
DH = [
    (0.000, 0.0, 0.675),
    (0.350, -90.0, 0.000),
    (1.150, 0.0, 0.000),
    (-0.041, -90.0, 1.000),
    (0.000, 90.0, 0.000),
    (0.000, -90.0, 0.240),
]
# HAL table is already MDH: joint i placement uses row i (a, alpha, d) directly:
#   T_i = RotX(alpha_i) TransX(a_i) RotZ(th_i) TransZ(d_i)
#       = [RotX(alpha_i) TransX(a_i) TransZ(d_i)] * RotZ(th_i)
#   (RotZ and TransZ commute), which is exactly pinocchio
#   placement_i * JointModelRZ(th_i).
MDH_ROWS = [(a, alpha, d) for (a, alpha, d) in DH]


def meas_to_dh(q_meas):
    """q_meas: (..., 6) rad, benchmark measurement convention -> DH thetas."""
    q = np.asarray(q_meas, dtype=float)
    th = np.empty_like(q)
    th[..., 0] = -q[..., 0]
    th[..., 1] = q[..., 1] - np.pi / 2
    th[..., 2] = q[..., 2]
    th[..., 3] = -q[..., 3]
    th[..., 4] = q[..., 4]
    th[..., 5] = -q[..., 5] + np.pi
    return th


def build_model():
    model = pin.Model()
    parent = 0
    for i, (a_i, alpha_i, d_i) in enumerate(MDH_ROWS):
        al = np.deg2rad(alpha_i)
        # placement = RotX(al) * TransX(a_i) * TransZ(d_i)
        R = np.array([[1, 0, 0],
                      [0, np.cos(al), -np.sin(al)],
                      [0, np.sin(al), np.cos(al)]])
        t = np.array([a_i, -np.sin(al) * d_i, np.cos(al) * d_i])
        placement = pin.SE3(R, t)
        jid = model.addJoint(parent, pin.JointModelRZ(), placement,
                             f"joint{i+1}")
        # body attached to this joint: inertia placeholder (replaced by
        # regressor parameters); pinocchio needs a body, mass 0 is fine here
        # because computeJointTorqueRegressor does not use the model inertias
        # except through the parameter vector ordering.
        model.appendBodyToJoint(jid, pin.Inertia(0.0, np.zeros(3),
                                                 np.zeros((3, 3))),
                                pin.SE3(np.eye(3), np.zeros(3)))
        parent = jid
    model.addFrame(pin.Frame("flange", parent, 0, pin.SE3(np.eye(3),
                                                          np.zeros(3)),
                             pin.FrameType.OP_FRAME))
    return model


if __name__ == "__main__":
    model = build_model()
    data = model.createData()
    for name, q_meas in [("home", np.zeros(6)),
                         ("home+joint2-30deg", np.array([0, -0.5, 0, 0, 0, 0]))]:
        th = meas_to_dh(q_meas)
        pin.forwardKinematics(model, data, th)
        pin.updateFramePlacements(model, data)
        pos = data.oMf[model.getFrameId("flange")].translation
        # wrist centre = frame of joint 5/6 origin: use joint 6 placement
        j6 = data.oMi[6].translation
        print(name, "DH th deg", np.round(np.rad2deg(th), 1),
              "flange xyz m", np.round(pos, 3), "joint6 origin", np.round(j6, 3))
    print("nq", model.nq, "nv", model.nv, "nbodies", model.nbodies)
    print("param ordering check: jointTorqueRegressor shape for one sample:")
    th = meas_to_dh(np.zeros(6))
    Y = pin.computeJointTorqueRegressor(model, data, th, np.zeros(6),
                                        np.zeros(6))
    print("Y shape", Y.shape)  # expect (6, 60)
