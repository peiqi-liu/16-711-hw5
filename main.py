#!/usr/bin/env python3
"""
16-711 Kinematics Dynamics and Control — Assignment Entry Point
=====================================================

Usage
-----
Start the simulator in a **separate terminal** first::

    python server.pyc

Then run a specific question::

    python main.py 1a          # Question 1a: Setpoint Control
    python main.py 1b          # Question 1b: Trajectory Tracking
    python main.py 2           # Question 2:  Pick and Place
    python main.py 3           # Question 3:  Ordered Stacking
    python main.py bonus       # Bonus:       Barrett Hand Grasping
"""

from __future__ import annotations
import sys
from time import perf_counter, sleep
import numpy as np
import matplotlib.pyplot as plt

from interface import RemoteRobotArm
from config import CONFIG
from controller import SetpointController, TrajectoryTrackingController
from trajectory import QuinticTrajectory
from kinematics import forward_kinematics
from tasks import PickAndPlaceTask, StackingTask, BarrettHandController
from utils import (
    Logger,
    plot_joint_tracking,
    plot_joint_errors,
    plot_torques,
    plot_cartesian_trajectory,
)


# ======================================================================
#  Global constants
# ======================================================================

DT = CONFIG.physics_dt              # 0.002 s  (500 Hz)
SETTLE_TIME = 2.0                   # seconds to wait after arm.reset()


def _make_task_tracking_controller() -> TrajectoryTrackingController:
    """Use milder gains for contact-heavy manipulation tasks."""
    controller = TrajectoryTrackingController()
    controller.reset_state(reset_torque_memory=True)
    return controller


# ======================================================================
#  Question 1a — Setpoint Control
# ======================================================================

# Target joint configuration (a clearly different pose from home)
Q_TARGET_1A = np.array([0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])


def run_setpoint_control(duration: float = 10.0) -> Logger:
    """Drive the arm from its home configuration to ``Q_TARGET_1A``.

    Args:
        duration: total experiment time [s].

    Returns:
        Populated ``Logger`` for analysis.
    """
    print("[Q1a] Setpoint control — connecting to simulator...")
    arm = RemoteRobotArm()
    arm.reset()
    sleep(SETTLE_TIME)

    controller = SetpointController(
        # ---- set your gains here (or inside the constructor) ----
    )
    log = Logger()

    # ===== TODO 1.2 ==========================================================
    # Implement the 500 Hz control loop for setpoint regulation.
    #
    # Pseudocode:
    #     start_time = clock = perf_counter()
    #     while elapsed < duration:
    #         # 1. Send torques computed on the previous iteration
    #         arm.set_trq(tau)
    #         # 2. Advance simulation and retrieve fresh state
    #         arm.step()
    #         q  = arm.get_pos()
    #         dq = arm.get_vel()
    #         # 3. Compute control for the NEXT step
    #         tau = controller.compute_torque(q, dq, Q_TARGET_1A)
    #         # 4. Log data
    #         log.record(t=elapsed, q=q, dq=dq, q_des=Q_TARGET_1A, tau=tau)
    #         # 5. Real-time pacing
    #         clock += DT
    #         sleep(max(0, clock - perf_counter()))
    #
    # IMPORTANT:
    #   - Initialise tau = np.zeros(7) before the loop.
    #   - Wrap in try/except KeyboardInterrupt for clean exit.
    # =========================================================================
    tau = np.zeros(7)
    start_time = clock = perf_counter()

    try:
        while (elapsed := perf_counter() - start_time) < duration:
            arm.set_trq(tau)
            arm.step()

            q = arm.get_pos()
            dq = arm.get_vel()

            tau = controller.compute_torque(q, dq, Q_TARGET_1A)
            ee_pos, _ = forward_kinematics(q)
            log.record(
                t=elapsed,
                q=q,
                dq=dq,
                q_des=Q_TARGET_1A,
                tau=tau,
                ee_pos=ee_pos,
            )

            clock += DT
            sleep(max(0.0, clock - perf_counter()))
    except KeyboardInterrupt:
        print("\n[Q1a] Interrupted by user.")
    finally:
        try:
            arm.set_trq(np.zeros(7))
        except Exception:
            pass
    # ===== END TODO 1.2 ======================================================

    return log


# ======================================================================
#  Question 1b — Trajectory Tracking
# ======================================================================

# Two-segment trajectory: home -> target -> home
Q_HOME = CONFIG.robot.home_pos
# Q_TARGET_1B = np.array([-0.5, 1.2, 0.3, 1.8, -0.2, 0.5, 0.0])
Q_TARGET_1B = np.array([0.3, 0.5, 0.0, 1.0, 0.0, 0.0, 0.0])
TRAJ_DURATION = 3.0   # seconds per segment


def run_trajectory_tracking(duration: float = 10.0) -> Logger:
    """Track a quintic polynomial trajectory: home -> target -> home.

    Args:
        duration: total experiment time [s].

    Returns:
        Populated ``Logger``.
    """
    print("[Q1b] Trajectory tracking — connecting to simulator...")
    arm = RemoteRobotArm()
    arm.reset()
    sleep(SETTLE_TIME)

    controller = TrajectoryTrackingController(
        # ---- set your gains here ----
        kp = np.array([330.0, 660.0, 210.0, 390.0, 54.0, 66.0, 15.0]),
        kd = np.array([14.0, 24.0, 12.0, 15.0, 4.0, 6.0, 1.0])
    )

    # Build the two-segment trajectory
    seg1 = QuinticTrajectory(Q_HOME, Q_TARGET_1B, TRAJ_DURATION, t_start=0.0)
    seg2 = QuinticTrajectory(Q_TARGET_1B, Q_HOME, TRAJ_DURATION, t_start=TRAJ_DURATION)

    log = Logger()

    # ===== TODO 1.5 ==========================================================
    # Implement the trajectory-tracking control loop.
    #
    # At each timestep:
    #   1. Compute elapsed time t_sim.
    #   2. Evaluate the correct trajectory segment:
    #        if t_sim < TRAJ_DURATION:  state = seg1.evaluate(t_sim)
    #        else:                      state = seg2.evaluate(t_sim)
    #   3. Read joint state.
    #   4. Compute torques using the tracking controller.
    #   5. Send torques, step, log, pace.
    #
    # The loop structure is the same as TODO 1.2.
    # =========================================================================
    tau = np.zeros(7)
    arm.set_trq(tau)
    arm.step()
    start_time = clock = perf_counter()

    try:
        while (t_sim := perf_counter() - start_time) < duration:
            q = arm.get_pos()
            dq = arm.get_vel()
            state = seg1.evaluate(t_sim) if t_sim < TRAJ_DURATION else seg2.evaluate(t_sim)

            tau = controller.compute_torque(q, dq, state.q, state.dq, state.ddq)
            ee_pos, _ = forward_kinematics(q)
            log.record(
                t=t_sim,
                q=q,
                dq=dq,
                q_des=state.q,
                dq_des=state.dq,
                tau=tau,
                ee_pos=ee_pos,
            )

            arm.set_trq(tau)
            arm.step()

            clock += DT
            sleep(max(0.0, clock - perf_counter()))
    except KeyboardInterrupt:
        print("\n[Q1b] Interrupted by user.")
    finally:
        try:
            arm.set_trq(np.zeros(7))
        except Exception:
            pass
    # ===== END TODO 1.5 ======================================================

    return log


# ======================================================================
#  Question 2 — Pick and Place
# ======================================================================

def run_pick_and_place() -> Logger:
    """Execute the pick-and-place task for all three objects.

    Returns:
        Populated ``Logger``.
    """
    print("[Q2] Pick and Place — connecting to simulator...")
    arm = RemoteRobotArm()
    arm.reset()
    sleep(SETTLE_TIME)
    controller = _make_task_tracking_controller()

    log = Logger()
    task = PickAndPlaceTask()
    success = task.execute(arm, controller, logger=log)
    print(f"[Q2] Task {'succeeded' if success else 'FAILED'}.")
    return log


# ======================================================================
#  Question 3 — Ordered Stacking
# ======================================================================

def run_stacking() -> Logger:
    """Execute the ordered stacking task.

    Returns:
        Populated ``Logger``.
    """
    print("[Q3] Ordered Stacking — connecting to simulator...")
    arm = RemoteRobotArm()
    arm.reset()
    sleep(SETTLE_TIME)
    controller = _make_task_tracking_controller()

    log = Logger()
    task = StackingTask()
    success = task.execute(arm, controller, logger=log)
    print(f"[Q3] Task {'succeeded' if success else 'FAILED'}.")
    return log


# ======================================================================
#  Bonus — Barrett Hand Grasping
# ======================================================================

def run_bonus() -> Logger:
    """Re-execute Questions 2 and 3 using finger-based grasping.

    This replaces ``arm.attach()`` / ``arm.detach()`` with the
    ``BarrettHandController`` for physically-simulated grasping.
    The same task classes are used — passing ``hand`` switches them
    from attach/detach to finger-based grasp/release automatically.

    Returns:
        Populated ``Logger``.
    """
    print("[Bonus] Barrett Hand Grasping — connecting to simulator...")
    arm = RemoteRobotArm()
    arm.reset()
    sleep(SETTLE_TIME)
    controller = _make_task_tracking_controller()

    hand = BarrettHandController()
    hand.reset_state()
    log = Logger()

    # --- Redo Q2 with finger grasping ---
    print("[Bonus] Running pick-and-place with Barrett Hand...")
    pnp_task = PickAndPlaceTask()
    pnp_task.execute(arm, controller, logger=log, hand=hand)

    # --- Redo Q3 with finger grasping ---
    print("[Bonus] Running stacking with Barrett Hand...")
    arm.reset()
    sleep(SETTLE_TIME)
    controller = _make_task_tracking_controller()
    hand = BarrettHandController()
    hand.reset_state()
    stack_task = StackingTask()
    stack_task.execute(arm, controller, logger=log, hand=hand)

    return log


# ======================================================================
#  CLI
# ======================================================================

QUESTIONS = {
    "1a":    ("Setpoint Control",      run_setpoint_control),
    "1b":    ("Trajectory Tracking",   run_trajectory_tracking),
    "2":     ("Pick and Place",        run_pick_and_place),
    "3":     ("Ordered Stacking",      run_stacking),
    "bonus": ("Barrett Hand Grasping", run_bonus),
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in QUESTIONS:
        print("Usage:  python main.py <question>")
        print()
        for key, (desc, _) in QUESTIONS.items():
            print(f"    {key:8s}  {desc}")
        sys.exit(1)

    qid = sys.argv[1]
    desc, fn = QUESTIONS[qid]
    print(f"\n{'='*60}")
    print(f"  16-711 Assignment — {desc}")
    print(f"{'='*60}\n")

    try:
        log = fn()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(0)

    # ---- Plot results ----
    if log.q_des:
        plot_joint_tracking(log, title=f"Q{qid}: {desc} — Joint Tracking")
        plot_joint_errors(log,  title=f"Q{qid}: {desc} — Tracking Error")
    if log.tau:
        plot_torques(log, title=f"Q{qid}: {desc} — Torques")
    if log.ee_pos:
        plot_cartesian_trajectory(log, title=f"Q{qid}: {desc} — EE Trajectory")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
