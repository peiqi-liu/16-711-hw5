"""
16-703 Robotic Manipulation — Forward / Inverse Kinematics
==========================================================
This module provides:
    * ``forward_kinematics(q)`` — end-effector pose        [PROVIDED]
    * ``jacobian(q)``          — geometric Jacobian         [PROVIDED]
    * ``inverse_kinematics()`` — Cartesian IK solver        [TODO 2.1]

The forward kinematics are derived from the manipulator's MJCF kinematic
chain.  All transforms follow the MuJoCo body-tree convention:

    T_child = T_parent  @  Trans(body.pos)  @  Rot(body.quat)  @  Rot_z(q_i)

where ``body.pos`` and ``body.quat`` are the fixed offsets read from the XML
and ``Rot_z(q_i)`` is the hinge-joint rotation (all joint axes are local z).
"""

from __future__ import annotations
import numpy as np
from config import JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER

# ======================================================================
#  Helpers
# ======================================================================

def _quat_to_rot(q: tuple | list | np.ndarray) -> np.ndarray:
    """Quaternion (w, x, y, z) -> 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - w*z),       2*(x*z + w*y)],
        [2*(x*y + w*z),       1 - 2*(x*x + z*z),   2*(y*z - w*x)],
        [2*(x*z - w*y),       2*(y*z + w*x),       1 - 2*(x*x + y*y)],
    ])


def _rot_z(theta: float) -> np.ndarray:
    """3x3 rotation about the z axis."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def _hom(R: np.ndarray, t) -> np.ndarray:
    """Build a 4x4 homogeneous transformation matrix."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(t, dtype=float)
    return T


# ======================================================================
#  Kinematic chain — data taken directly from the Barrett WAM MJCF
# ======================================================================
#
# Each entry is (translation, quaternion-wxyz) of the body frame relative
# to its parent *before* the joint rotation is applied.
#
# Index 0 is the fixed base link (no joint).
# Indices 1–7 correspond to joints 0–6.

_CHAIN = [
    # 0: base_link  (fixed in world)
    ([0.0,    0.0,   0.6  ], [1.0,       0.0,       0.0,      0.0     ]),
    # 1: shoulder_yaw_link   — joint 0  (base_yaw)
    ([0.0,    0.0,   0.346], [1.0,       0.0,       0.0,      0.0     ]),
    # 2: shoulder_pitch_link — joint 1  (shoulder_pitch)
    ([0.0,    0.0,   0.0  ], [0.707107, -0.707107,  0.0,      0.0     ]),
    # 3: upper_arm_link      — joint 2  (shoulder_yaw)
    ([0.0,    0.0,   0.0  ], [0.707107,  0.707107,  0.0,      0.0     ]),
    # 4: forearm_link        — joint 3  (elbow_pitch)
    ([0.045,  0.0,   0.55 ], [0.707107, -0.707107,  0.0,      0.0     ]),
    # 5: wrist_yaw_link      — joint 4  (wrist_yaw)
    ([-0.045,-0.3,   0.0  ], [0.707107,  0.707107,  0.0,      0.0     ]),
    # 6: wrist_pitch_link    — joint 5  (wrist_pitch)
    ([0.0,    0.0,   0.0  ], [0.707107, -0.707107,  0.0,      0.0     ]),
    # 7: wrist_palm_link     — joint 6  (palm_yaw)
    ([0.0,    0.0,   0.0  ], [0.707107,  0.707107,  0.0,      0.0     ]),
]

# Fixed offset from wrist_palm_link to bhand_palm_link (end-effector)
_EE_POS  = [0.0, 0.0, 0.06]
_EE_QUAT = [0.0, 0.0, 0.0, 1.0]   # 180 deg about z


# ======================================================================
#  Forward Kinematics  [PROVIDED — do not modify]
# ======================================================================

def forward_kinematics(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute end-effector pose for joint configuration *q*.

    Args:
        q: (7,) joint angles in radians.

    Returns:
        position:  (3,) end-effector position in the world frame [m].
        rotation:  (3, 3) end-effector orientation matrix (world frame).
    """
    q = np.asarray(q, dtype=float)
    assert q.shape == (7,), f"Expected (7,) joint vector, got {q.shape}"

    # Base link (fixed — no joint)
    pos0, quat0 = _CHAIN[0]
    T = _hom(_quat_to_rot(quat0), pos0)

    # Links 1–7 each carry one hinge joint
    for i in range(7):
        pos_i, quat_i = _CHAIN[i + 1]
        T_body  = _hom(_quat_to_rot(quat_i), pos_i)
        T_joint = _hom(_rot_z(q[i]), [0, 0, 0])
        T = T @ T_body @ T_joint

    # End-effector offset
    T = T @ _hom(_quat_to_rot(_EE_QUAT), _EE_POS)

    return T[:3, 3].copy(), T[:3, :3].copy()


# ======================================================================
#  Geometric Jacobian  [PROVIDED — do not modify]
# ======================================================================

def jacobian(q: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    """Numerical geometric Jacobian  J(q) in R^{6 x 7}.

    Rows 0-2: translational Jacobian  (dp/dq).
    Rows 3-5: rotational Jacobian     (domega/dq).

    Uses central finite differences on ``forward_kinematics``.

    Args:
        q: (7,) joint angles.
        delta: finite-difference step size.

    Returns:
        J: (6, 7) Jacobian matrix.
    """
    q = np.asarray(q, dtype=float)
    J = np.zeros((6, 7))
    for i in range(7):
        q_plus = q.copy();  q_plus[i]  += delta
        q_minus = q.copy(); q_minus[i] -= delta

        p_plus,  R_plus  = forward_kinematics(q_plus)
        p_minus, R_minus = forward_kinematics(q_minus)

        # Translational part
        J[:3, i] = (p_plus - p_minus) / (2 * delta)

        # Rotational part — extract angular velocity from dR
        dR = (R_plus - R_minus) / (2 * delta)
        R_mid = (R_plus + R_minus) / 2.0
        skew = dR @ R_mid.T
        J[3, i] = skew[2, 1]
        J[4, i] = skew[0, 2]
        J[5, i] = skew[1, 0]

    return J


# ======================================================================
#  Inverse Kinematics  [TODO 2.1 — YOUR IMPLEMENTATION]
# ======================================================================

def inverse_kinematics(
    target_pos: np.ndarray,
    q_init: np.ndarray,
    target_rot: np.ndarray | None = None,
    max_iter: int = 200,
    tol: float = 1e-3,
) -> np.ndarray:
    """Solve inverse kinematics for a desired end-effector pose.

    Given a target Cartesian position (and optionally a target orientation),
    find a joint configuration *q* such that ``forward_kinematics(q)`` places
    the end-effector within *tol* of the target.

    Two modes:
        * ``target_rot is None`` — position-only IK (3-D error).
        * ``target_rot`` provided — full 6-DOF pose IK (position + orientation).
          Strongly recommended for grasping, where palm orientation matters.

    Suggested approach — Damped Least-Squares (Levenberg–Marquardt):
        1. Compute current EE pose via ``forward_kinematics(q)``.
        2. Compute the 3-D position error   e_p = target_pos - current_pos.
        3. If ``target_rot`` is given, compute the 3-D orientation error via
           the skew-symmetric vee-map of ``0.5 * (R_des R^T - R R_des^T)``.
           Stack into a 6-D error  e = [e_p; e_o].
        4. Take the matching Jacobian slice:
               position-only:   J = jacobian(q)[:3, :]   (3x7)
               full pose:       J = jacobian(q)          (6x7)
        5. Solve for  dq  using the damped pseudo-inverse:
               dq = J^T (J J^T + lambda^2 I)^{-1} e
        6. Update  q <- q + alpha * dq   (with a suitable step size alpha).
        7. Clip q to joint limits (``JOINT_LIMITS_LOWER/UPPER``).
        8. Repeat until ||e|| < tol or max iterations reached.

    Args:
        target_pos: (3,) desired end-effector position in world frame [m].
        q_init:     (7,) initial joint guess (e.g. current joint state).
        target_rot: (3, 3) desired end-effector rotation matrix (world frame),
                    or ``None`` for position-only IK.
        max_iter:   maximum number of iterations.
        tol:        convergence tolerance on error norm
                    ([m] for position-only, mixed [m, rad] for full-pose).

    Returns:
        q_solution: (7,) joint configuration that reaches the target,
                    or the best iterate if convergence was not achieved.
    """
    # ===== TODO 2.1 ==========================================================
    # Implement the damped least-squares IK solver described above.
    # Support BOTH the position-only branch and the full 6-DOF branch.
    #
    # Orientation-error hint (vee map):
    #     skew = 0.5 * (R_des @ R.T - R @ R_des.T)
    #     e_o  = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
    #
    # Hints:
    #   - A damping coefficient lambda ~ 0.01-0.1 works well.
    #   - Step size alpha ~ 0.5 provides a good speed/stability trade-off.
    #   - Use np.clip(q, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER).
    #   - Print convergence info for debugging: iteration, ||e||.
    # ==========================================================================
    target_pos = np.asarray(target_pos, dtype=float)
    q = np.clip(np.asarray(q_init, dtype=float).copy(),
                JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    if target_rot is not None:
        target_rot = np.asarray(target_rot, dtype=float)

    # Warm-start: position-only pre-solve when far from target
    if target_rot is not None:
        current_pos, _ = forward_kinematics(q)
        if np.linalg.norm(target_pos - current_pos) > 0.15:
            q = inverse_kinematics(
                target_pos, q,
                target_rot=None,
                max_iter=100,
                tol=5e-3,
            )

    # Hyperparameters
    lam       = 0.05
    step_max  = 0.25
    ori_w     = 0.5
    q_mid     = 0.5 * (JOINT_LIMITS_LOWER + JOINT_LIMITS_UPPER)
    null_gain = 0.005

    best_q        = q.copy()
    best_err_norm = np.inf
    J_full        = None
    J_age         = 999  # force recompute on first iteration

    for i in range(max_iter):
        current_pos, current_rot = forward_kinematics(q)
        pos_error    = target_pos - current_pos
        pos_err_norm = np.linalg.norm(pos_error)

        # Recompute Jacobian only every 5 iterations
        if J_age >= 5:
            J_full = jacobian(q)
            J_age  = 0
        J_age += 1

        if target_rot is None:
            error = pos_error
            J     = J_full[:3, :]
            if pos_err_norm < tol:
                return q
        else:
            skew      = 0.5 * (target_rot @ current_rot.T - current_rot @ target_rot.T)
            ori_error = np.array([skew[2, 1], skew[0, 2], skew[1, 0]])
            error     = np.concatenate([pos_error, ori_w * ori_error])
            J         = J_full.copy()
            J[3:, :] *= ori_w
            if pos_err_norm < tol and np.linalg.norm(ori_error) < tol * 5:
                return q

        err_norm = np.linalg.norm(error)
        if err_norm < best_err_norm:
            best_err_norm = err_norm
            best_q        = q.copy()
        if err_norm < tol:
            return q

        # Adaptive step size — larger when far, smaller when close
        alpha = min(0.8, 0.3 + 0.5 * (err_norm / (err_norm + 0.1)))

        # Damped least-squares step
        n   = J.shape[0]
        lhs = J @ J.T + (lam ** 2) * np.eye(n)
        dq  = J.T @ np.linalg.solve(lhs, error)

        # Null-space joint centering
        JpJ = J.T @ np.linalg.solve(lhs, J)
        N   = np.eye(7) - JpJ
        dq += null_gain * (N @ (q_mid - q))

        # Step-size limiting
        dq_norm = np.linalg.norm(dq)
        if dq_norm > step_max:
            dq *= step_max / dq_norm

        q = np.clip(q + alpha * dq, JOINT_LIMITS_LOWER, JOINT_LIMITS_UPPER)

    return best_q
    # ===== END TODO 2.1 =======================================================
