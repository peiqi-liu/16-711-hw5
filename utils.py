"""
16-711 Kinematics Dynamics and Control — Utilities
========================================
Data logging and plotting helpers.  You may extend this file freely.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


# ======================================================================
#  Data Logger
# ======================================================================

@dataclass
class Logger:
    """Accumulates time-series data during a control loop for later analysis.

    Usage::

        from utils import Logger
        from kinematics import forward_kinematics

        log = Logger()

        # Inside your 500 Hz control loop:
        #   q   = arm.get_pos()
        #   dq  = arm.get_vel()
        #   tau = controller.compute_torque(q, dq, q_des, dq_des, ddq_des)
        #   ee_pos, _ = forward_kinematics(q)
        #
        #   log.record(
        #       t=elapsed,
        #       q=q,
        #       dq=dq,
        #       q_des=q_des,
        #       dq_des=dq_des,
        #       tau=tau,
        #       ee_pos=ee_pos,
        #   )

        # After the loop, plot results:
        # plot_joint_tracking(log, title="My Experiment")
        # plt.show()
    """
    time:   list[float]      = field(default_factory=list)
    q:      list[np.ndarray] = field(default_factory=list)
    dq:     list[np.ndarray] = field(default_factory=list)
    q_des:  list[np.ndarray] = field(default_factory=list)
    dq_des: list[np.ndarray] = field(default_factory=list)
    tau:    list[np.ndarray] = field(default_factory=list)
    ee_pos: list[np.ndarray] = field(default_factory=list)

    def record(
        self,
        t: float,
        q: np.ndarray,
        dq: np.ndarray | None = None,
        q_des: np.ndarray | None = None,
        dq_des: np.ndarray | None = None,
        tau: np.ndarray | None = None,
        ee_pos: np.ndarray | None = None,
    ):
        self.time.append(t)
        self.q.append(q.copy())
        if dq is not None:
            self.dq.append(dq.copy())
        if q_des is not None:
            self.q_des.append(q_des.copy())
        if dq_des is not None:
            self.dq_des.append(dq_des.copy())
        if tau is not None:
            self.tau.append(tau.copy())
        if ee_pos is not None:
            self.ee_pos.append(ee_pos.copy())


# ======================================================================
#  Plotting Functions
# ======================================================================

JOINT_NAMES = [
    "Base Yaw", "Shoulder Pitch", "Shoulder Yaw", "Elbow Pitch",
    "Wrist Yaw", "Wrist Pitch", "Palm Yaw",
]


def plot_joint_tracking(log: Logger, title: str = "Joint Tracking") -> plt.Figure:
    """Plot measured vs. desired joint positions for all 7 joints.

    Args:
        log:   a ``Logger`` instance with at least ``time``, ``q``, ``q_des``.
        title: figure title.

    Returns:
        matplotlib Figure.
    """
    t = np.array(log.time)
    q = np.array(log.q)
    q_des = np.array(log.q_des) if log.q_des else None

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.plot(t, q[:, j], label="measured", linewidth=0.8)
        if q_des is not None:
            ax.plot(t, q_des[:, j], "--", label="desired", linewidth=0.8)
        ax.set_ylabel(f"{JOINT_NAMES[j]} [rad]")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    axes[7].set_visible(False)

    axes[6].set_xlabel("Time [s]")
    axes[5].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_joint_errors(log: Logger, title: str = "Joint Tracking Error") -> plt.Figure:
    """Plot per-joint tracking error norm over time.

    Args:
        log:   a ``Logger`` instance.
        title: figure title.

    Returns:
        matplotlib Figure.
    """
    t = np.array(log.time)
    q = np.array(log.q)
    q_des = np.array(log.q_des)
    err = q_des - q

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for j in range(7):
        ax1.plot(t, err[:, j], label=JOINT_NAMES[j], linewidth=0.7)
    ax1.set_ylabel("Joint Error [rad]")
    ax1.legend(fontsize=7, ncol=4)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Per-Joint Error")

    ax2.plot(t, np.linalg.norm(err, axis=1), "k-", linewidth=0.9)
    ax2.set_ylabel("||e|| [rad]")
    ax2.set_xlabel("Time [s]")
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Error Norm")

    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_torques(log: Logger, title: str = "Applied Torques") -> plt.Figure:
    """Plot applied joint torques over time.

    Args:
        log:   a ``Logger`` instance with ``tau`` data.
        title: figure title.

    Returns:
        matplotlib Figure.
    """
    t = np.array(log.time)
    tau = np.array(log.tau)

    fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
    axes = axes.flatten()

    for j in range(7):
        ax = axes[j]
        ax.plot(t, tau[:, j], linewidth=0.7)
        ax.set_ylabel(f"{JOINT_NAMES[j]} [Nm]")
        ax.grid(True, alpha=0.3)

    axes[7].set_visible(False)
    axes[6].set_xlabel("Time [s]")
    axes[5].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_cartesian_trajectory(log: Logger, title: str = "EE Trajectory") -> plt.Figure:
    """3-D plot of the end-effector path.

    Args:
        log:   a ``Logger`` instance with ``ee_pos`` data.
        title: figure title.

    Returns:
        matplotlib Figure.
    """
    pos = np.array(log.ee_pos)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], linewidth=1.0)
    ax.scatter(*pos[0], color="green", s=60, label="start")
    ax.scatter(*pos[-1], color="red", s=60, label="end")
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_zlabel("Z [m]")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
