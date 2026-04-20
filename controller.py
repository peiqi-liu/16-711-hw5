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
        # Default gains — students should tune these
        self.kp = np.asarray(kp) if kp is not None else np.zeros(7)
        self.kd = np.asarray(kd) if kd is not None else np.zeros(7)
        self.ki = np.asarray(ki) if ki is not None else np.zeros(7)

        # Integral error accumulator
        self._int_error = np.zeros(7)
        self._dt = 0.002  # control timestep

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
        raise NotImplementedError("TODO 1.1: Implement SetpointController.compute_torque()")
        # ===== END TODO 1.1 ==================================================


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
        self.kp = np.asarray(kp) if kp is not None else np.zeros(7)
        self.kd = np.asarray(kd) if kd is not None else np.zeros(7)

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
        raise NotImplementedError("TODO 1.4: Implement TrajectoryTrackingController.compute_torque()")
        # ===== END TODO 1.4 ==================================================
