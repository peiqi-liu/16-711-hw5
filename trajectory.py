"""
16-711 Kinematics Dynamics and Control — Trajectory Generation
====================================================
Provides ``QuinticTrajectory`` for generating smooth joint-space motions
with continuous position, velocity, and acceleration profiles.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TrajectoryState:
    """Desired kinematic state at a single time instant."""
    q: np.ndarray       # (n,) desired joint positions   [rad]
    dq: np.ndarray      # (n,) desired joint velocities  [rad/s]
    ddq: np.ndarray     # (n,) desired joint accels      [rad/s^2]


class QuinticTrajectory:
    """Fifth-order polynomial trajectory with zero-velocity boundary conditions.

    The trajectory satisfies the boundary conditions:
        q(0)  = q_start,   dq(0)  = 0,   ddq(0)  = 0
        q(T)  = q_end,     dq(T)  = 0,   ddq(T)  = 0

    The normalised time variable is  s = t / T,  s in [0, 1].

    The **unique** quintic that satisfies these six constraints is:

        h(s) = 10 s^3  -  15 s^4  +  6 s^5

    so that   q(s) = q_start + (q_end - q_start) * h(s).

    The first and second derivatives w.r.t. *real* time are obtained via
    the chain rule using ds/dt = 1/T.

    Attributes:
        q_start:  (n,) start configuration.
        q_end:    (n,) end configuration.
        duration: trajectory duration T [s].
        t_start:  global clock time at which the trajectory begins [s].
    """

    def __init__(
        self,
        q_start: np.ndarray,
        q_end: np.ndarray,
        duration: float,
        t_start: float = 0.0,
    ):
        assert duration > 0, "Duration must be positive."
        self.q_start = np.asarray(q_start, dtype=float)
        self.q_end = np.asarray(q_end, dtype=float)
        self.duration = float(duration)
        self.t_start = float(t_start)
        self._delta = self.q_end - self.q_start

    @property
    def t_end(self) -> float:
        return self.t_start + self.duration

    def evaluate(self, t: float) -> TrajectoryState:
        """Return the desired state at global time *t*.

        Outside the interval ``[t_start, t_end]`` the trajectory is clamped:
        before ``t_start`` it returns the start state, after ``t_end`` the
        end state — both with zero velocity and acceleration.

        Args:
            t: current time [s].

        Returns:
            ``TrajectoryState`` with fields ``q``, ``dq``, ``ddq``.
        """
        # ===== TODO 1.3 ======================================================
        # Implement the quintic polynomial trajectory evaluation.
        #
        # Steps:
        #   1. Compute the normalised time  s = (t - t_start) / duration.
        #   2. Clamp s to [0, 1].
        #   3. Evaluate h(s), h'(s), h''(s)  where
        #          h(s)   = 10 s^3  - 15 s^4  + 6 s^5
        #          h'(s)  = 30 s^2  - 60 s^3  + 30 s^4
        #          h''(s) = 60 s    - 180 s^2 + 120 s^3
        #   4. Map back to joint space:
        #          q_des   = q_start + delta * h(s)
        #          dq_des  = delta * h'(s) / T
        #          ddq_des = delta * h''(s) / T^2
        #   5. If s <= 0 or s >= 1, velocities and accelerations must be zero.
        #
        # Return:
        #     TrajectoryState(q=q_des, dq=dq_des, ddq=ddq_des)
        # =====================================================================
        if t <= self.t_start:
            return TrajectoryState(
                q=self.q_start.copy(),
                dq=np.zeros_like(self.q_start),
                ddq=np.zeros_like(self.q_start),
            )

        if t >= self.t_end:
            return TrajectoryState(
                q=self.q_end.copy(),
                dq=np.zeros_like(self.q_end),
                ddq=np.zeros_like(self.q_end),
            )

        s = np.clip((t - self.t_start) / self.duration, 0.0, 1.0)

        h = 10.0 * s**3 - 15.0 * s**4 + 6.0 * s**5
        h_dot = 30.0 * s**2 - 60.0 * s**3 + 30.0 * s**4
        h_ddot = 60.0 * s - 180.0 * s**2 + 120.0 * s**3

        q_des = self.q_start + self._delta * h
        dq_des = self._delta * h_dot / self.duration
        ddq_des = self._delta * h_ddot / (self.duration**2)

        return TrajectoryState(q=q_des, dq=dq_des, ddq=ddq_des)
        # ===== END TODO 1.3 ==================================================
