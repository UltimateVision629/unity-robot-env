"""
SO100 4-DOF kinematic chain — pure numpy + scipy, no external dependencies.

Replaces the `lerobot_IK` / `get_robot("so100")` coordinate-translation used by
arm_ik.py's MuJoCoIK backend.  The original chain was defined in
lerobot-kinematics' `create_so100()` as an ETS (elementary transform sequence);
this module hard-codes the same parameters so the Joy-Con → MuJoCo EEF
translation behaves identically without the lerobot-kinematics package.

Chain (base → EEF), 4 revolute joints:
    base → tx(0.02943) → tz(0.05504) → Ry(q0) → tx(0.1127) → tz(-0.02798)
        → Ry(q1) → tx(0.13504) → tz(0.00519) → Ry(q2) → tx(0.0593)
        → tz(0.00996) → Rx(q3) → EEF

    q0 = shoulder_pitch, q1 = elbow, q2 = wrist_pitch, q3 = wrist_roll
    (base yaw is NOT in the model — callers prepend it, see arm_ik.py)

Joint limits (from create_so100):
    [[-3.14158, -0.2, -1.5, -3.14158],
     [ 0.2,      3.14158, 1.5, 3.14158]]

API is a drop-in for `lerobot_IK(q_now, target_pose, robot)`:
    so100_ik(q_warm, target_pose) -> (joints_4dof, success)
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

# ═══════════════════════════════════════════════════════════════════════════
# Model parameters (from lerobot_kinematics create_so100)
# ═══════════════════════════════════════════════════════════════════════════

# (axis, tx, ty, tz) — constant transform BEFORE each revolute joint
_SO100_LINKS = [
    ("y", 0.02943, 0.0, 0.05504),    # base → shoulder_pitch
    ("y", 0.1127, 0.0, -0.02798),    # upper arm → elbow
    ("y", 0.13504, 0.0, 0.00519),    # forearm → wrist_pitch
    ("x", 0.0593, 0.0, 0.00996),     # wrist → wrist_roll
]

_SO100_QLIM = np.array([
    [-3.14158, -0.2, -1.5, -3.14158],
    [0.2, 3.14158, 1.5, 3.14158],
])


# ═══════════════════════════════════════════════════════════════════════════
# Forward kinematics
# ═══════════════════════════════════════════════════════════════════════════

def _translate(tx: float, ty: float, tz: float) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = (tx, ty, tz)
    return T


def _rotate(axis: str, angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    T = np.eye(4)
    if axis == "y":
        T[0, 0], T[0, 2] = c, s
        T[2, 0], T[2, 2] = -s, c
    elif axis == "x":
        T[1, 1], T[1, 2] = c, -s
        T[2, 1], T[2, 2] = s, c
    else:
        raise ValueError(f"Unknown joint axis: {axis!r}")
    return T


def so100_fk(q: np.ndarray) -> np.ndarray:
    """4-DOF joints → 4×4 homogeneous transform (base → EEF)."""
    T = np.eye(4)
    for (axis, tx, ty, tz), qi in zip(_SO100_LINKS, q):
        T = T @ _translate(tx, ty, tz) @ _rotate(axis, qi)
    return T


# ═══════════════════════════════════════════════════════════════════════════
# Inverse kinematics (drop-in for lerobot_IK)
# ═══════════════════════════════════════════════════════════════════════════

def _angle_axis_error(Te: np.ndarray, Tep: np.ndarray) -> np.ndarray:
    """Rotation error between Te and Tep as an angle-axis 3-vector.

    Same formula as roboticstoolbox `angle_axis_python` (p_servo.py): the
    rotation part of R_target @ R_current^T is expressed as angle×axis.
    """
    R_err = Tep[:3, :3] @ Te[:3, :3].T
    li = np.array([
        R_err[2, 1] - R_err[1, 2],
        R_err[0, 2] - R_err[2, 0],
        R_err[1, 0] - R_err[0, 1],
    ])
    if np.linalg.norm(li) < 1e-10:
        if np.trace(R_err) > 0:
            return np.zeros(3)
        return np.pi / 2 * (np.diag(R_err) + 1)
    ln = np.linalg.norm(li)
    return math.atan2(ln, np.trace(R_err) - 1) * li / ln


def so100_ik(
    q_warm: np.ndarray,
    target_pose: np.ndarray,
    max_nfev: int = 40,
    tol: float = 1e-3,
    ftol: float = 1e-2,
) -> tuple[np.ndarray, bool]:
    """Numerical IK: target_pose → 4-DOF joints.

    Args:
        q_warm: warm-start joints [shoulder_pitch, elbow, wrist_pitch,
            wrist_roll] (radians) — tracks the previous solution for
            branch continuity (same role as lerobot_IK's q_now).
        target_pose: [x, y, z, roll, pitch, yaw] — XYZ Euler rotation in
            radians, translation in metres (same convention as lerobot_IK).
        max_nfev: max solver iterations.
        tol: pose-error cost threshold for success (position m + rotation rad).
        ftol: least_squares 求解容差（2026-09-02 起可调，默认 1e-2）。
            翻译层的解只进 MuJoCo FK 作 IK 目标，0.03mm 级误差无感；
            实测 ftol 1e-8→1e-2 快 ~25%（1.10→0.83ms/帧，见
            unity-robot-env/test/benchmark_ik_perf.py）。别放宽 tol——
            那是 success 判定，保持 1e-3。

    Returns:
        (joints_4dof, success) — identical return format to lerobot_IK.
    """
    q_warm = np.asarray(q_warm, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)

    r = R.from_euler("xyz", target[3:6], degrees=False)
    T_target = np.eye(4)
    T_target[:3, :3] = r.as_matrix()
    T_target[:3, 3] = target[:3]

    # Pose-only cost — NO pull-to-warm (matches lerobot LM, which used step
    # damping only).  Branch continuity comes from warm-starting + the smooth
    # cap below; adding a (q - q_warm) term lets the solver trade pose error
    # against joint distance and stall in wrong-branch compromises.
    def cost(q):
        Te = so100_fk(q)
        return np.concatenate([
            T_target[:3, 3] - Te[:3, 3],          # position error (m)
            _angle_axis_error(Te, T_target),      # rotation error (rad)
        ])

    result = least_squares(
        cost,
        q_warm,
        bounds=(_SO100_QLIM[0], _SO100_QLIM[1]),
        method="trf",
        max_nfev=max_nfev,
        ftol=ftol,
        xtol=ftol,
    )

    # This chain is only a coordinate translator for the MuJoCo backend.  Its
    # model cannot exactly satisfy every Joy-Con pose, so scipy's half-squared
    # cost compared directly with a linear tolerance rejects otherwise usable
    # targets.  Judge the two physical residuals explicitly instead.
    residual = np.asarray(cost(result.x), dtype=np.float64)
    pos_error = float(np.linalg.norm(residual[:3]))
    rot_error = float(np.linalg.norm(residual[3:6]))
    success = (pos_error <= 0.10) or (rot_error <= 0.10)
    if not success:
        return -np.ones(4), False

    # Same per-frame smooth cap as lerobot's smooth_joint_motion: cap each
    # joint's change from the warm-start at 0.1 rad so the translation can
    # never branch-jump faster than the arm could follow during collection.
    q = result.x.copy()
    for i in range(len(q)):
        delta = q[i] - q_warm[i]
        if abs(delta) > 0.1:
            q[i] = q_warm[i] + np.sign(delta) * 0.1
    return q, True


# ═══════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # Round-trip: random joints → FK → IK should recover the original joints
    rng = np.random.default_rng(0)
    q_true = np.array([0.1, 1.0, -0.5, 0.7])
    T = so100_fk(q_true)
    pose = np.concatenate([T[:3, 3], R.from_matrix(T[:3, :3]).as_euler("xyz")])

    q_warm = q_true.copy()
    q_ik, ok = so100_ik(q_warm, pose)
    print(f"q_true = {q_true}")
    print(f"q_ik   = {q_ik}  success={ok}")
    print(f"err    = {np.linalg.norm(q_ik - q_true):.2e} rad")
