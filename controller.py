"""
16-711 Kinematics Dynamics and Control — Controllers
==========================================
Base class and student implementations for Questions 1a and 1b.

Controllers receive the current and desired joint state and return
a 7-D torque vector to be sent to the robot via ``arm.set_trq()``.
"""

from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
from config import MAX_TORQUES
from kinematics import _CHAIN, _quat_to_rot, _rot_z, _hom


# Joint damping values come directly from ``robot_manual.md``.
NOMINAL_JOINT_DAMPING = np.array([1.98, 0.55, 1.65, 0.88, 0.55, 0.11, 0.11])

# Coarse reflected inertias for a diagonal nominal joint-space model.
# These follow the manual's inertia trend: heavy proximal joints, light distal ones.
NOMINAL_REFLECTED_INERTIA = np.array([8.0, 10.0, 4.5, 2.5, 0.35, 0.20, 0.08])

# Link masses / COMs come from ``robot_manual.md`` and are used for a
# lightweight nominal gravity-compensation term.
NOMINAL_LINK_MASSES = np.array([5.0, 3.87494, 2.20228, 0.500168, 1.05376, 0.517974, 0.0828613])
NOMINAL_LINK_COMS = (
    np.array([-0.00443422, -0.00066489, -0.128904]),
    np.array([-0.00236981, -0.0154211, 0.0310561]),
    np.array([0.00683259, 3.309e-05, 0.392492]),
    np.array([-0.0400149, -0.142717, -0.00022942]),
    np.array([8.921e-05, 0.00435824, -0.00511217]),
    np.array([-0.00012262, -0.0246834, -0.0170319]),
    np.array([0.0, 0.0, 0.055]),
)
GRAVITY_VECTOR = np.array([0.0, 0.0, -9.81])


def nominal_gravity_torque(q: np.ndarray) -> np.ndarray:
    """Approximate gravity torques from the nominal link masses and COMs."""
    q = np.asarray(q, dtype=float)

    T = _hom(_quat_to_rot(_CHAIN[0][1]), _CHAIN[0][0])
    joint_origins: list[np.ndarray] = []
    joint_axes: list[np.ndarray] = []
    com_positions: list[np.ndarray] = []

    for i in range(7):
        pos_i, quat_i = _CHAIN[i + 1]
        T_body = T @ _hom(_quat_to_rot(quat_i), pos_i)

        joint_origins.append(T_body[:3, 3].copy())
        joint_axes.append((T_body[:3, :3] @ np.array([0.0, 0.0, 1.0])).copy())

        T = T_body @ _hom(_rot_z(q[i]), [0.0, 0.0, 0.0])
        com_positions.append((T[:3, :3] @ NOMINAL_LINK_COMS[i] + T[:3, 3]).copy())

    tau = np.zeros(7)
    for i in range(7):
        force_i = NOMINAL_LINK_MASSES[i] * GRAVITY_VECTOR
        p_com_i = com_positions[i]
        for j in range(i + 1):
            # Negative sign matches the generalized gravity torques needed
            # in tau = M qdd + c + g.
            tau[j] -= np.dot(
                np.cross(joint_axes[j], p_com_i - joint_origins[j]),
                force_i,
            )

    return tau


@runtime_checkable
class BaseController(Protocol):
    """Protocol that every controller must satisfy.

    Any object with a ``compute_torque`` method of the correct signature is
    accepted — explicit inheritance is *not* required.
    """

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
    ) -> np.ndarray:
        """Compute the 7-D joint torque command.

        Args:
            q:       (7,) measured joint positions  [rad].
            dq:      (7,) measured joint velocities [rad/s].
            q_des:   (7,) desired joint positions   [rad].
            dq_des:  (7,) desired joint velocities  [rad/s].
            ddq_des: (7,) desired joint accels      [rad/s^2].

        Returns:
            tau: (7,) commanded joint torques [Nm].
                 You are responsible for clipping to ``MAX_TORQUES``.
        """
        ...


# ======================================================================
#  Question 1a — Setpoint Controller
# ======================================================================

class SetpointController:
    """Joint-space setpoint controller.

    The simplest viable approach is a PD (or PID) law:

        tau = Kp (q_des - q)  -  Kd dq  +  Ki integral(q_des - q)

    For improved performance you may add gravity / Coriolis compensation
    using the nominal link inertials listed in ``robot_manual.md``.  A
    model-based (computed-torque) controller is not required for this
    sub-question but will be rewarded with higher marks.

    Attributes:
        kp: (7,) proportional gains.
        kd: (7,) derivative gains.
        ki: (7,) integral gains  (optional, default zeros).
    """

    def __init__(
        self,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        ki: np.ndarray | None = None,
    ):
        # Default gains — these keep Q1a as a direct joint-space PID law,
        # which is closer to the intent of the skeleton comments and avoids
        # the overly aggressive model-based behavior we saw in the simulator.
        self.kp = np.asarray(kp) if kp is not None else np.array(
            [120.0, 200.0, 60.0, 150.0, 30.0, 50.0, 10.0]
        )
        self.kd = np.asarray(kd) if kd is not None else np.array(
            [10.0, 25.0, 10.0, 3.0, 0.5, 0.2, 0.05]
        )
        self.ki = np.asarray(ki) if ki is not None else np.array(
            [50.0, 100.0, 50.0, 50.0, 20.0, 15.0, 5.0]
        )

        # Integral error accumulator
        self._int_error = np.zeros(7)
        self._dt = 0.002  # control timestep
        self._int_limit = np.array([0.4, 0.35, 0.35, 0.35, 0.2, 0.2, 0.2])

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray = None,
        ddq_des: np.ndarray = None,
    ) -> np.ndarray:
        """Compute setpoint-tracking torques.

        Args:
            q:       (7,) current joint positions.
            dq:      (7,) current joint velocities.
            q_des:   (7,) target joint positions.
            dq_des:  unused for setpoint control (can be None).
            ddq_des: unused for setpoint control (can be None).

        Returns:
            tau: (7,) torque command.
        """
        # ===== TODO 1.1 ======================================================
        # Implement a joint-space setpoint controller.
        #
        # For better performance consider adding:
        #   - Integral action with anti-windup.
        #   - Gravity compensation using the nominal model.
        #   - Full computed-torque (inverse dynamics) control.
        #
        # Be sure to:
        #   - Choose and set reasonable Kp, Kd (and Ki) gains.
        #   - Clip torques to MAX_TORQUES to respect actuator limits.
        #
        # =====================================================================
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        dq_des = np.zeros_like(q) if dq_des is None else np.asarray(dq_des, dtype=float)

        pos_error = q_des - q
        vel_error = dq_des - dq
        self._int_error += pos_error * self._dt
        self._int_error = np.clip(self._int_error, -self._int_limit, self._int_limit)

        tau = (
            self.kp * pos_error
            + self.kd * vel_error
            + self.ki * self._int_error
        )
        tau = np.clip(tau, -MAX_TORQUES, MAX_TORQUES)

        return tau
        # ===== END TODO 1.1 ==================================================

    def reset_state(self) -> None:
        """Clear any accumulated controller state."""
        self._int_error[:] = 0.0


# ======================================================================
#  Question 1b — Trajectory Tracking Controller
# ======================================================================

class TrajectoryTrackingController:
    """Joint-space trajectory tracking controller.

    Unlike setpoint control, the desired state is **time-varying**: the
    reference includes velocities ``dq_des`` and accelerations ``ddq_des``.
    Your controller can exploit these feedforward terms to achieve
    significantly better tracking than a PD law alone.

    Attributes:
        kp: (7,) proportional gains.
        kd: (7,) derivative gains.
    """

    def __init__(
        self,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
    ):
        # For Q1b we keep a conservative trajectory-tracking controller that
        # still uses the reference velocity/acceleration terms explicitly.
        # The nominal damping / reflected inertia come from ``robot_manual.md``
        # and are used as a lightweight model-based feedforward.
        self.kp = np.asarray(kp) if kp is not None else np.array([100, 150, 80, 100, 10, 8, 5])
        self.kd = np.asarray(kd) if kd is not None else np.array([10, 14, 8, 8, 3.0, 2.5, 1.5])
        # self.ki = np.array([0.5, 8.0, 1.0, 6.0, 0.5, 1.0, 0.1])
        self.ki = np.zeros(7)
        self._nominal_inertia = 0.8 * NOMINAL_REFLECTED_INERTIA
        self._nominal_damping = 0.6 * NOMINAL_JOINT_DAMPING
        self._gravity_ff_scale = 1.0
        self._dt = 0.002
        self._int_error = np.zeros(7)
        self._int_limit = np.zeros(7)
        # self._int_limit = np.array([0.04, 0.10, 0.04, 0.08, 0.04, 0.05, 0.03])
        self._pos_limit = np.array([0.35, 0.35, 0.30, 0.30, 0.25, 0.25, 0.20])
        self._vel_limit = np.array([1.5, 1.5, 1.2, 1.2, 1.0, 1.0, 0.8])

        # Torque slew limiting is useful as a final smoothing pass, but it can
        # also delay gravity support and recovery from large tracking errors.
        # Keep it available for experiments, but default to the unclipped raw
        # controller torque before actuator saturation.
        self._use_torque_slew_limit = False
        self._prev_tau = np.zeros(7)
        self._tau_rate_limit = np.array([12.0, 14.0, 6.0, 5.0, 2.0, 2.0, 0.5])
        self._reset_torque_memory_at_next_step = True

    def compute_torque(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        q_des: np.ndarray,
        dq_des: np.ndarray,
        ddq_des: np.ndarray,
    ) -> np.ndarray:
        """Compute trajectory-tracking torques.

        Args:
            q:       (7,) current joint positions.
            dq:      (7,) current joint velocities.
            q_des:   (7,) desired joint positions.
            dq_des:  (7,) desired joint velocities.
            ddq_des: (7,) desired joint accelerations.

        Returns:
            tau: (7,) torque command.
        """
        # ===== TODO 1.4 ======================================================
        # Implement a trajectory tracking controller.
        #
        #
        #   Build M and c from the nominal inertials in robot_manual.md
        #   if you are implementing a dynamics based controller.
        #
        # Clip torques to MAX_TORQUES.
        # =====================================================================
        q = np.asarray(q, dtype=float)
        dq = np.asarray(dq, dtype=float)
        q_des = np.asarray(q_des, dtype=float)
        dq_des = np.asarray(dq_des, dtype=float)
        ddq_des = np.asarray(ddq_des, dtype=float)

        pos_error = q_des - q
        vel_error = dq_des - dq
        pos_error = np.clip(pos_error, -self._pos_limit, self._pos_limit)
        vel_error = np.clip(vel_error, -self._vel_limit, self._vel_limit)

        # Use one consistent tracking law for the full motion rather than
        # switching into a separate hold-phase controller near the endpoints.
        self._int_error = np.clip(
            0.995 * self._int_error + pos_error * self._dt,
            -self._int_limit,
            self._int_limit,
        )

        tau_fb = (
            self.kp * pos_error
            + self.kd * vel_error
            + self.ki * self._int_error
        )

        # Lightweight model-based feedforward using the nominal manual values:
        #   tau_ff = M_hat * ddq_des + c_hat(dq_des) + g_hat(q)
        # Keep the inertial / damping feedforward tied to the desired motion,
        # but evaluate gravity at the measured pose so the arm always gets
        # support against its *actual* configuration instead of a potentially
        # distant future reference.
        tau_ff = (
            self._nominal_inertia * ddq_des
            + self._nominal_damping * dq_des
            + self._gravity_ff_scale * nominal_gravity_torque(q)
        )
        tau_raw = tau_fb + tau_ff

        if self._reset_torque_memory_at_next_step:
            self._prev_tau = np.clip(
                self._gravity_ff_scale * nominal_gravity_torque(q),
                -MAX_TORQUES,
                MAX_TORQUES,
            )
            self._reset_torque_memory_at_next_step = False

        if self._use_torque_slew_limit:
            tau_slew_limited = np.clip(
                tau_raw,
                self._prev_tau - self._tau_rate_limit,
                self._prev_tau + self._tau_rate_limit,
            )
        else:
            tau_slew_limited = tau_raw.copy()

        tau = np.clip(tau_slew_limited, -MAX_TORQUES, MAX_TORQUES)
        self._prev_tau = tau
        return tau
        # ===== END TODO 1.4 ==================================================

    def reset_state(
        self,
        reset_torque_memory: bool = True,
        q_init: np.ndarray | None = None,
    ) -> None:
        """Clear accumulated error, optionally resetting torque slew memory."""
        self._int_error[:] = 0.0
        if reset_torque_memory:
            if q_init is None:
                self._reset_torque_memory_at_next_step = True
            else:
                self._prev_tau = np.clip(
                    self._gravity_ff_scale * nominal_gravity_torque(q_init),
                    -MAX_TORQUES,
                    MAX_TORQUES,
                )
                self._reset_torque_memory_at_next_step = False
