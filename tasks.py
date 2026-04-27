"""
16-711 Kinematics Dynamics and Control — Task Implementations
====================================================
Scaffolding for Questions 2, 3, and the Bonus.

Each task class encapsulates the planning and execution logic for a
manipulation primitive.  Students implement the TODO methods; the
``main.py`` entry point calls ``task.execute(arm, controller)``.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from time import perf_counter, sleep
import numpy as np

from interface import RemoteRobotArm
from controller import BaseController, NOMINAL_JOINT_DAMPING, nominal_gravity_torque
from kinematics import forward_kinematics, jacobian, inverse_kinematics
from trajectory import QuinticTrajectory
from config import (
    CONFIG,
    BARRIER_TOP_Z,
    DROP_BOX_CENTER,
    DROP_BOX_FLOOR_Z,
    ITEM_POSITIONS,
    ITEM_DIMS,
    MAX_TORQUES,
)
from utils import Logger


# ======================================================================
#  Constants
# ======================================================================

CLEARANCE_Z = BARRIER_TOP_Z + 0.15   # safe height above barrier [m]
APPROACH_OFFSET_Z = 0.22            # height above object for approach [m]
PICK_OFSET_Z = 0.1
SEGMENT_DURATION = 2.0               # default time per trajectory segment [s]
GRASP_SETTLE_TIME = 0.35             # hold time after grasp / release [s]
PLACE_HOLD_TIME = 0.20               # hold time before release [s]
HAND_ACTUATION_TIME = 2.25           # time to close/open Barrett hand [s]
HAND_MAX_SPREAD_RATE = 0.55          # max spread command rate [rad/s]
HAND_MAX_CURL_RATE = 0.80            # max finger curl command rate [rad/s]
HAND_ARM_DAMPING_SCALE = 0.35        # passive arm damping during finger motion
HAND_HOLD_KP = np.array([30.0, 40.0, 20.0, 40.0, 5.0, 5.0, 3.0])
HAND_HOLD_KD = np.array([3.0, 5.0, 4.0, 5.0, 2.0, 2.0, 1.0])
STARTUP_STAGE_Y = 0.55               # safer y position before first pick [m]
STARTUP_LIFT_EXTRA = 0.10            # extra initial lift above current ee z [m]

# Palm-down end-effector orientation (world frame).
# Use as ``target_rot`` in ``inverse_kinematics`` so the palm faces
# the table during approach/grasp/place.
PALM_DOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])


def _object_half_height(object_name: str) -> float:
    dims = ITEM_DIMS[object_name]
    shape = dims["shape"]
    if shape == "cuboid":
        return float(dims["half_size"][2])
    if shape == "cylinder":
        return float(dims["half_height"])
    if shape == "sphere":
        return float(dims["radius"])
    raise ValueError(f"Unsupported object shape: {shape}")


# ======================================================================
#  Abstract base
# ======================================================================

class BaseTask(ABC):
    """Interface for all manipulation tasks."""

    @abstractmethod
    def execute(
        self,
        arm: RemoteRobotArm,
        controller: BaseController,
        logger: Logger | None = None,
        hand: BarrettHandController | None = None,
    ) -> bool:
        """Run the task to completion.

        Args:
            arm:        robot interface.
            controller: joint-space tracking controller.
            logger:     optional data logger.
            hand:       optional Barrett Hand controller.  When provided,
                        grasping uses ``hand.execute_grasp()`` /
                        ``hand.execute_release()`` instead of the
                        simplified ``arm.attach()`` / ``arm.detach()``.

        Returns:
            ``True`` if the task completed successfully.
        """
        ...


# ======================================================================
#  Shared helper — trajectory execution
# ======================================================================

def execute_trajectory(
    arm: RemoteRobotArm,
    controller: BaseController,
    traj: QuinticTrajectory,
    logger: Logger | None = None,
) -> None:
    """Drive the arm along a ``QuinticTrajectory`` in real time.

    This is a convenience wrapper that implements the 500 Hz control loop
    for a single trajectory segment.  You may call it from within your
    ``execute()`` methods.

    Args:
        arm:        robot interface.
        controller: any object satisfying ``BaseController``.
        traj:       trajectory to follow.
        logger:     optional data logger.
    """
    dt = 0.002
    start = clock = perf_counter()

    while perf_counter() - start <= traj.duration + 0.1:
        t_now = perf_counter() - start + traj.t_start
        state = traj.evaluate(t_now)

        q = arm.get_pos()
        dq = arm.get_vel()

        tau = controller.compute_torque(q, dq, state.q, state.dq, state.ddq)
        arm.set_trq(tau)
        arm.step()

        if logger is not None:
            ee_pos, _ = forward_kinematics(q)
            logger.record(
                t=perf_counter() - start,
                q=q, dq=dq, q_des=state.q, dq_des=state.dq,
                tau=tau, ee_pos=ee_pos,
            )

        clock += dt
        idle = clock - perf_counter()
        if idle > 0:
            sleep(idle)


def _prime_arm_state(arm: RemoteRobotArm, attempts: int = 3) -> np.ndarray:
    """Advance one control step so the state cache reflects the simulator."""
    for _ in range(attempts):
        arm.set_trq(np.zeros(7))
        arm.step()
    return arm.get_pos()


def _plan_palm_waypoint_path(
    q_current: np.ndarray,
    waypoints: list[np.ndarray],
    slow_indices: tuple[int, ...] = (),
    target_rots: list[np.ndarray | None] | None = None,
) -> list[QuinticTrajectory]:
    """Plan joint trajectories through palm-center Cartesian waypoints."""
    q_current = np.asarray(q_current, dtype=float)
    cart_waypoints = [forward_kinematics(q_current)[0]]
    joint_waypoints = [q_current.copy()]
    q_seed = q_current.copy()

    if target_rots is None:
        target_rots = [PALM_DOWN] * len(waypoints)

    for waypoint, target_rot in zip(waypoints, target_rots):
        waypoint = np.asarray(waypoint, dtype=float)
        q_seed = inverse_kinematics(waypoint, q_seed, target_rot=target_rot)
        joint_waypoints.append(q_seed.copy())
        cart_waypoints.append(waypoint.copy())

    trajectories: list[QuinticTrajectory] = []
    slow_set = set(slow_indices)
    for i in range(len(joint_waypoints) - 1):
        q_start = joint_waypoints[i]
        q_end = joint_waypoints[i + 1]
        cart_dist = np.linalg.norm(cart_waypoints[i + 1] - cart_waypoints[i])
        joint_dist = np.linalg.norm(q_end - q_start)
        duration = SEGMENT_DURATION * max(cart_dist / 0.4, joint_dist / 1.5, 0.5)
        if i in slow_set:
            duration *= 1.8
        duration = float(np.clip(duration, 1.0, 5.0))
        trajectories.append(QuinticTrajectory(q_start, q_end, duration, t_start=0.0))

    return trajectories


def _reset_controller_state(controller: BaseController) -> None:
    """Reset controller memory when entering a new motion phase, if supported."""
    reset_fn = getattr(controller, "reset_state", None)
    if callable(reset_fn):
        reset_fn()


def _plan_startup_egress(q_start: np.ndarray) -> list[QuinticTrajectory]:
    """Move up first, then retreat in +y before the first pick.

    The nominal home pose sits close to the object cluster in this setup.
    A direct trajectory from home toward the first approach pose can sweep
    through the sphere, so we first lift vertically and then move to a
    safer staging pose on the positive-y side of the workspace.
    """
    q_start = np.asarray(q_start, dtype=float)
    ee_start, _ = forward_kinematics(q_start)

    lift_z = max(CLEARANCE_Z, float(ee_start[2] + STARTUP_LIFT_EXTRA))
    lift_pos = np.array([ee_start[0], ee_start[1], lift_z])
    stage_pos = np.array([ee_start[0], STARTUP_STAGE_Y, CLEARANCE_Z])

    q_lift = inverse_kinematics(lift_pos, q_start, target_rot=PALM_DOWN)
    q_stage = inverse_kinematics(stage_pos, q_lift, target_rot=PALM_DOWN)

    return [
        QuinticTrajectory(q_start, q_lift, 2.0, t_start=0.0),
        # QuinticTrajectory(q_lift, q_stage, 2.5, t_start=0.0),
    ]


def _hold_configuration(
    arm: RemoteRobotArm,
    controller: BaseController,
    q_hold: np.ndarray,
    duration: float,
) -> np.ndarray:
    """Hold the current configuration for a short settling period."""
    q_hold = np.asarray(q_hold, dtype=float)
    dq_des = np.zeros(7)
    ddq_des = np.zeros(7)
    tau = np.zeros(7)
    start = clock = perf_counter()

    while perf_counter() - start < duration:
        q = arm.get_pos()
        dq = arm.get_vel()
        tau = controller.compute_torque(q, dq, q_hold, dq_des, ddq_des)
        arm.set_trq(tau)
        arm.step()

        clock += 0.002
        idle = clock - perf_counter()
        if idle > 0:
            sleep(idle)

    return arm.get_pos()


def _execute_pick_place_segments(
    arm: RemoteRobotArm,
    controller: BaseController,
    trajectories: list[QuinticTrajectory],
    object_name: str,
    logger: Logger | None = None,
    hand: BarrettHandController | None = None,
    pre_release_hold: float = PLACE_HOLD_TIME,
    post_release_hold: float = GRASP_SETTLE_TIME,
) -> np.ndarray:
    """Execute the six-segment pick/place motion with grasp and release."""
    # if len(trajectories) != 6:
    #     raise ValueError(f"Expected 6 trajectory segments, got {len(trajectories)}")

    shape = ITEM_DIMS[object_name]["shape"]

    for idx, traj in enumerate(trajectories):
        # Re-anchor every segment at the measured current configuration.
        # This avoids reference jumps after grasp/release contact events and
        # makes the start of each new segment dynamically consistent.
        q_start = arm.get_pos().copy()
        exec_traj = QuinticTrajectory(q_start, traj.q_end, traj.duration, t_start=0.0)
        _reset_controller_state(controller)
        execute_trajectory(arm, controller, exec_traj, logger=logger)
        q_current = arm.get_pos()

        if idx == 1:
            if hand is None:
                arm.attach(object_name)
            else:
                q_current = hand.execute_grasp(arm, hand.plan_grasp(shape))
            _reset_controller_state(controller)
            q_current = _hold_configuration(arm, controller, q_current, GRASP_SETTLE_TIME)
        elif idx == 4:
            q_current = _hold_configuration(arm, controller, q_current, pre_release_hold)
            if hand is None:
                arm.detach()
            else:
                q_current = hand.execute_release(arm)
            _reset_controller_state(controller)
            q_current = _hold_configuration(arm, controller, q_current, post_release_hold)

    return arm.get_pos()


def _execute_stacking_cycle_closed_loop(
    arm: RemoteRobotArm,
    controller: BaseController,
    object_name: str,
    object_center: np.ndarray,
    stack_center: np.ndarray,
    logger: Logger | None = None,
    hand: BarrettHandController | None = None,
) -> np.ndarray:
    """Execute one stacking cycle, correcting the place pose after grasp."""
    object_center = np.asarray(object_center, dtype=float)
    stack_center = np.asarray(stack_center, dtype=float)
    shape = ITEM_DIMS[object_name]["shape"]

    palm_pick = object_center.copy()
    palm_pick[2] += _object_half_height(object_name)

    pick_target_rots = [None, None] if object_center[0] > 0.85 else None
    pick_trajs = _plan_palm_waypoint_path(arm.get_pos(), [
        palm_pick + np.array([0.0, 0.0, APPROACH_OFFSET_Z]),
        palm_pick,
    ], target_rots=pick_target_rots)

    q_current = arm.get_pos()
    for traj in pick_trajs:
        q_start = arm.get_pos().copy()
        exec_traj = QuinticTrajectory(q_start, traj.q_end, traj.duration, t_start=0.0)
        _reset_controller_state(controller)
        execute_trajectory(arm, controller, exec_traj, logger=logger)
        q_current = arm.get_pos()

    if hand is None:
        arm.attach(object_name)
    else:
        hand.execute_grasp(arm, hand.plan_grasp(shape))

    actual_palm_pos, _ = forward_kinematics(q_current)
    planned_palm_pos = palm_pick
    palm_tracking_offset = actual_palm_pos - planned_palm_pos

    _reset_controller_state(controller)
    q_current = _hold_configuration(arm, controller, q_current, GRASP_SETTLE_TIME)

    place_trajs = _plan_palm_waypoint_path(
        q_current,
        [
            np.array([0.3, 0.25, CLEARANCE_Z]),
            np.array([0.3, -0.25, CLEARANCE_Z]),
            stack_center + palm_tracking_offset,
            np.array([0.3, 0.25, CLEARANCE_Z]),
        ],
        slow_indices=(0, 1),
    )

    for idx, traj in enumerate(place_trajs):
        q_start = arm.get_pos().copy()
        exec_traj = QuinticTrajectory(q_start, traj.q_end, traj.duration, t_start=0.0)
        _reset_controller_state(controller)
        execute_trajectory(arm, controller, exec_traj, logger=logger)
        q_current = arm.get_pos()

        if idx == 2:
            q_current = _hold_configuration(arm, controller, q_current, PLACE_HOLD_TIME)
            if hand is None:
                arm.detach()
            else:
                hand.execute_release(arm)
            _reset_controller_state(controller)
            q_current = _hold_configuration(arm, controller, q_current, GRASP_SETTLE_TIME)

    return arm.get_pos()


# ======================================================================
#  Question 2 — Pick and Place
# ======================================================================

class PickAndPlaceTask(BaseTask):
    """Move all three items from the left side of the barrier (y > 0)
    to the drop box on the right side (y < 0).

    Workflow for each object:

        1. Plan a trajectory from the current configuration to an
           **approach pose** directly above the object.
        2. Lower the end-effector to the **grasp pose** at the object.
        3. Call ``arm.attach(object_name)`` to rigidly grasp the object.
        4. Lift to the **clearance height** (above the barrier).
        5. Translate across the barrier to above the **drop position**.
        6. Lower to the drop position and call ``arm.detach()``.
        7. Retract upward before moving to the next object.

    The order in which the three items are transferred is up to you.

    If a ``BarrettHandController`` is passed to ``execute()``, the task
    uses finger-based grasping instead of ``attach`` / ``detach``.
    """

    def plan_cartesian_path(
        self,
        pick_pos: np.ndarray,
        place_pos: np.ndarray,
        q_current: np.ndarray,
    ) -> list[QuinticTrajectory]:
        """Plan a sequence of trajectory segments for one pick-and-place cycle.

        The method should return an ordered list of ``QuinticTrajectory``
        objects.  Each segment connects two joint configurations obtained
        via inverse kinematics from Cartesian waypoints.

        Suggested waypoints (all in world frame):
            W0: current configuration                        (no IK needed)
            W1: above the object at pick_pos + [0, 0, APPROACH_OFFSET_Z]
            W2: at the object (grasp pose)   pick_pos
            W3: lift to clearance height     [pick_pos[0], pick_pos[1], CLEARANCE_Z]
            W4: above drop position          [place_pos[0], place_pos[1], CLEARANCE_Z]
            W5: at the drop position         place_pos
            W6: retract upward               [place_pos[0], place_pos[1], CLEARANCE_Z]

        Between waypoints, use ``inverse_kinematics()`` to convert
        Cartesian targets to joint configurations, then create a
        ``QuinticTrajectory`` for each consecutive pair.

        Args:
            pick_pos:   (3,) Cartesian position of the object centre.
            place_pos:  (3,) Cartesian position of the target location.
            q_current:  (7,) current joint configuration (start of path).

        Returns:
            List of ``QuinticTrajectory`` segments in execution order.
        """
        # ===== TODO 2.2 ======================================================
        # Implement the Cartesian path planner described above.
        #
        # Steps:
        #   1. Convert each waypoint to a joint config using inverse_kinematics().
        #      Pass ``target_rot=PALM_DOWN`` so the palm stays oriented toward
        #      the table throughout the motion (essential for stable grasps).
        #      Use the previous waypoint's joint config as q_init for the next.
        #   2. Create QuinticTrajectory segments between consecutive configs.
        #   3. Set segment durations appropriately (SEGMENT_DURATION is a good
        #      default; increase it for longer motions).
        #   4. Return the list of trajectory segments.
        #
        # Hint: be careful with segment t_start values — they should be
        # relative to each segment (starting at 0), since execute_trajectory()
        # resets its own clock for each segment.
        # =====================================================================
        pick_pos = np.asarray(pick_pos, dtype=float)
        place_pos = np.asarray(place_pos, dtype=float)
        q_current = np.asarray(q_current, dtype=float)

        waypoints = [
            pick_pos + np.array([0.0, 0.0, APPROACH_OFFSET_Z]),
            # np.array([0.35, 0.25, CLEARANCE_Z - 0.2]),
            pick_pos,
            # np.array([pick_pos[0], pick_pos[1], CLEARANCE_Z]),
            np.array([0.3, 0.25, CLEARANCE_Z]),
            # np.array([place_pos[0], place_pos[1], CLEARANCE_Z]),
            np.array([0.3, -0.25, CLEARANCE_Z]),
            place_pos + np.array([0, 0, 0.15]),
            np.array([place_pos[0], place_pos[1], CLEARANCE_Z]),
            np.array([0.3, 0.25, CLEARANCE_Z]),
        ]

        joint_waypoints = [q_current.copy()]
        # joint_waypoints = []
        q_seed = q_current.copy()
        relax_pick_orientation = pick_pos[0] > 0.85
        for idx, waypoint in enumerate(waypoints):
            target_rot = None if relax_pick_orientation and idx in (0, 1) else PALM_DOWN
            q_seed = inverse_kinematics(waypoint, q_seed, target_rot=target_rot)
            joint_waypoints.append(q_seed.copy())

        cart_waypoints = [forward_kinematics(q_current)[0], *waypoints]
        trajectories: list[QuinticTrajectory] = []

        for i in range(len(joint_waypoints) - 1):
            q_start = joint_waypoints[i]
            q_end = joint_waypoints[i + 1]
            cart_dist = np.linalg.norm(cart_waypoints[i + 1] - cart_waypoints[i])
            joint_dist = np.linalg.norm(q_end - q_start)
            duration = SEGMENT_DURATION * max(cart_dist / 0.4, joint_dist / 1.5, 0.5)
            # Give the vertical lift and over-barrier transfer more time so
            # modest tracking lag still leaves geometric clearance.
            if i in (2, 3):
                duration *= 1.8
            duration = float(np.clip(duration, 1.0, 5.0))
            trajectories.append(QuinticTrajectory(q_start, q_end, duration, t_start=0.0))

        return trajectories
        # ===== END TODO 2.2 ==================================================

    def execute(
        self,
        arm: RemoteRobotArm,
        controller: BaseController,
        logger: Logger | None = None,
        hand: BarrettHandController | None = None,
    ) -> bool:
        """Execute the full pick-and-place task for all three objects.

        Args:
            arm:        robot interface (already connected and reset).
            controller: a trajectory-tracking controller.
            logger:     optional data logger.
            hand:       optional Barrett Hand controller.  When ``None``,
                        grasping uses ``arm.attach()`` / ``arm.detach()``.
                        When provided, uses ``hand.execute_grasp()`` /
                        ``hand.execute_release()`` for finger-based grasping.

        Returns:
            ``True`` on success.
        """
        # ===== TODO 2.3 ======================================================
        # Orchestrate the complete pick-and-place sequence.
        #
        # For each item ("item1", "item2", "item3"):
        #   1. Determine pick position from ITEM_POSITIONS.
        #   2. Determine place position (e.g. DROP_BOX_CENTER with appropriate z).
        #   3. Call self.plan_cartesian_path(pick, place, q_current).
        #   4. Execute trajectory segments using execute_trajectory().
        #   5. Grasp / release the object:
        #        if hand is None:
        #            arm.attach(object_name)   /  arm.detach()
        #        else:
        #            shape = ITEM_DIMS[object_name]["shape"]
        #            grasp_cfg = hand.plan_grasp(shape)
        #            hand.execute_grasp(arm, grasp_cfg)
        #            ...
        #            hand.execute_release(arm)
        #   6. Update q_current from arm.get_pos() after each segment.
        #
        # Remember:
        #   - Grasp after lowering to the object.
        #   - Release after lowering to the place position.
        #   - A short sleep (0.3-0.5 s) after grasp/release helps the
        #     simulator settle.
        # =====================================================================
        # Match the more stable ordered-stacking startup: plan from the known
        # reset pose, then execute a short egress before the first pick.
        q_plan = CONFIG.robot.home_pos.copy()
        startup_trajs = _plan_startup_egress(q_plan)
        q_plan = startup_trajs[-1].q_end.copy()

        place_targets = {
            "item1": np.array([
                DROP_BOX_CENTER[0],
                DROP_BOX_CENTER[1],
                DROP_BOX_FLOOR_Z + _object_half_height("item1"),
            ]),
            "item2": np.array([
                DROP_BOX_CENTER[0],
                DROP_BOX_CENTER[1],
                DROP_BOX_FLOOR_Z + _object_half_height("item2"),
            ]),
            "item3": np.array([
                DROP_BOX_CENTER[0],
                DROP_BOX_CENTER[1],
                DROP_BOX_FLOOR_Z + _object_half_height("item3"),
            ]),
        }

        planned_cycles: list[tuple[str, list[QuinticTrajectory]]] = []
        for object_name in ("item1", "item2", "item3"):
            pick_pos = ITEM_POSITIONS[object_name].copy()
            pick_pos[-1] += (_object_half_height(object_name) * 2)
            place_pos = place_targets[object_name]
            trajectories = self.plan_cartesian_path(pick_pos, place_pos, q_plan)
            planned_cycles.append((object_name, trajectories))
            q_plan = trajectories[-1].q_end.copy()

        q_current = _prime_arm_state(arm)
        for traj in startup_trajs:
            _reset_controller_state(controller)
            execute_trajectory(arm, controller, traj, logger=logger)
            q_current = arm.get_pos()
        for object_name, trajectories in planned_cycles:
            q_current = _execute_pick_place_segments(
                arm,
                controller,
                trajectories,
                object_name,
                logger=logger,
                hand=hand,
            )

        return True
        # ===== END TODO 2.3 ==================================================


# ======================================================================
#  Question 3 — Ordered Stacking
# ======================================================================

class StackingTask(BaseTask):
    """Stack the three items in the drop box in a specific order:

        Bottom:  cuboid   (item1)
        Middle:  cylinder (item2)
        Top:     sphere   (item3)

    This extends the pick-and-place logic with precise **placement poses**
    that account for the geometry of each object to create a stable stack.
    """

    def compute_stack_poses(self) -> dict[str, np.ndarray]:
        """Compute the world-frame placement positions for each item.

        The stack is built inside the drop box.  You must account for the
        height of each object so that each item rests on the one below.

        Geometry reminders (from ``config.ITEM_DIMS``):
            - item1 (cuboid):   half-extents [0.02, 0.02, 0.02] → height 0.04 m
            - item2 (cylinder): radius 0.02 m, half-height 0.02 m → height 0.04 m
            - item3 (sphere):   radius 0.02 m → diameter 0.04 m

        The drop-box floor is at z = DROP_BOX_FLOOR_Z (0.92 m).

        Returns:
            Dictionary mapping item name to (3,) placement position::

                {"item1": np.array([x, y, z]),
                 "item2": np.array([x, y, z]),
                 "item3": np.array([x, y, z])}
        """
        # ===== TODO 3.1 ======================================================
        # Compute the centre position of each item when it is placed in the
        # stack.  All three items should be centred at the drop-box xy and
        # stacked along z.
        #
        # Example calculation:
        #   item1 (cuboid) centre z  = DROP_BOX_FLOOR_Z + cuboid_half_z
        #   item2 (cylinder) centre z = item1_top + cylinder_half_height
        #   item3 (sphere) centre z   = item2_top + sphere_radius
        # =====================================================================
        stack_xy = DROP_BOX_CENTER[:2].copy()

        item1_half = _object_half_height("item1")
        item2_half = _object_half_height("item2")
        item3_half = _object_half_height("item3")

        item1_z = DROP_BOX_FLOOR_Z + item1_half
        item1_top = item1_z + item1_half

        item2_z = item1_top + item2_half
        item2_top = item2_z + item2_half

        item3_z = item2_top + item3_half

        return {
            "item1": np.array([stack_xy[0], stack_xy[1], item1_z + 0.05]),
            "item2": np.array([stack_xy[0], stack_xy[1], item2_z + 0.05]),
            "item3": np.array([stack_xy[0], stack_xy[1], item3_z + 0.05]),
        }
        # ===== END TODO 3.1 ==================================================

    def execute(
        self,
        arm: RemoteRobotArm,
        controller: BaseController,
        logger: Logger | None = None,
        hand: BarrettHandController | None = None,
    ) -> bool:
        """Execute the ordered stacking sequence.

        The items must be placed **in order** — cuboid first, then cylinder,
        then sphere — so that each item has a stable resting surface.

        You may reuse ``PickAndPlaceTask.plan_cartesian_path`` and
        ``execute_trajectory`` from Question 2.

        Args:
            arm:        robot interface.
            controller: trajectory-tracking controller.
            logger:     optional data logger.
            hand:       optional Barrett Hand controller.  When ``None``,
                        grasping uses ``arm.attach()`` / ``arm.detach()``.
                        When provided, uses finger-based grasping.

        Returns:
            ``True`` on success.
        """
        # ===== TODO 3.2 ======================================================
        # Orchestrate the stacking sequence.
        #
        # Steps:
        #   1. Compute placement poses via self.compute_stack_poses().
        #   2. For each item in order ["item1", "item2", "item3"]:
        #      a. Pick the item from its initial position (ITEM_POSITIONS).
        #      b. Place it at the computed stack position.
        #      c. Use the pick-and-place trajectory planning from Q2.
        #      d. Grasp / release:
        #           if hand is None:
        #               arm.attach(name) / arm.detach()
        #           else:
        #               hand.execute_grasp(arm, hand.plan_grasp(shape))
        #               ...
        #               hand.execute_release(arm)
        #   3. After placing each item, briefly hold position to let the
        #      stack settle before releasing and retracting.
        #
        # Tip:  You can instantiate PickAndPlaceTask and call its
        #       plan_cartesian_path() method for reuse.
        # =====================================================================
        planner = PickAndPlaceTask()
        stack_targets = self.compute_stack_poses()

        q_plan = CONFIG.robot.home_pos.copy()
        startup_trajs = _plan_startup_egress(q_plan)
        q_plan = startup_trajs[-1].q_end.copy()
        planned_cycles: list[tuple[str, list[QuinticTrajectory]]] = []
        for object_name in ("item1", "item2", "item3"):
            pick_pos = ITEM_POSITIONS[object_name].copy()
            pick_pos[-1] += (_object_half_height(object_name) * 2)
            place_pos = stack_targets[object_name]
            base_trajectories = planner.plan_cartesian_path(pick_pos, place_pos, q_plan)
            trajectories = []
            for idx, traj in enumerate(base_trajectories):
                duration_scale = 1.0
                if idx in (4, 5):
                    duration_scale = 1.6
                elif idx in (2, 3):
                    duration_scale = 1.2
                trajectories.append(
                    QuinticTrajectory(
                        traj.q_start,
                        traj.q_end,
                        traj.duration * duration_scale,
                        t_start=0.0,
                    )
                )
            planned_cycles.append((object_name, trajectories))
            q_plan = trajectories[-1].q_end.copy()

        q_current = _prime_arm_state(arm)

        for object_name in ("item1", "item2", "item3"):
            # Plan each item only after the previous one has actually finished.
            # Use the measured palm pose after grasp to compensate tracking
            # error in the stack placement target.
            object_center = ITEM_POSITIONS[object_name].copy()
            q_current = _execute_stacking_cycle_closed_loop(
                arm,
                controller,
                object_name,
                object_center,
                stack_targets[object_name],
                logger=logger,
                hand=hand,
            )

        return True
        # ===== END TODO 3.2 ==================================================


# ======================================================================
#  Bonus (Question 4) — Barrett Hand Grasping
# ======================================================================

class BarrettHandController:
    """Controller for the Barrett Hand's three-finger mechanism.

    The Barrett Hand has 8 joints across 3 fingers:

        Finger 1:  prox_joint (spread), med_joint, dist_joint
        Finger 2:  prox_joint (spread), med_joint, dist_joint
        Finger 3:  med_joint, dist_joint  (no spread — fixed)

    Equality constraints enforced by the simulator:
        - finger_1/prox == finger_2/prox     (spread is symmetric)
        - dist = 0.33 * med                  (for each finger)

    This reduces the hand to **4 independent DOF**:
        1. Spread angle        (finger 1 & 2 proximal)
        2. Finger 1 curl       (med joint)
        3. Finger 2 curl       (med joint)
        4. Finger 3 curl       (med joint)

    Finger positions are commanded via ``arm.set_finger_pos(spread,
    curl_1, curl_2, curl_3)``.

    Joint ranges:
        prox_joint:  [0, pi]       rad
        med_joint:   [0, 2.443]    rad
        dist_joint:  [0, 0.838]    rad   (coupled — do not command directly)
    """

    def __init__(self):
        self._current_cmd = {
            "spread": 0.0,
            "curl_1": 0.0,
            "curl_2": 0.0,
            "curl_3": 0.0,
        }

    def reset_state(self) -> None:
        """Reset the hand controller's internal command / hold state."""
        self._current_cmd = {
            "spread": 0.0,
            "curl_1": 0.0,
            "curl_2": 0.0,
            "curl_3": 0.0,
        }

    @staticmethod
    def _sanitize_cmd(cmd: dict) -> dict:
        return {
            "spread": float(np.clip(cmd["spread"], 0.0, np.pi)),
            "curl_1": float(np.clip(cmd["curl_1"], 0.0, 2.443)),
            "curl_2": float(np.clip(cmd["curl_2"], 0.0, 2.443)),
            "curl_3": float(np.clip(cmd["curl_3"], 0.0, 2.443)),
        }

    def _move_fingers(self, arm: RemoteRobotArm, target_cmd: dict, duration: float = HAND_ACTUATION_TIME) -> np.ndarray:
        """Interpolate finger targets using a slow, rate-limited motion."""
        target_cmd = self._sanitize_cmd(target_cmd)
        start_cmd = self._sanitize_cmd(self._current_cmd)
        start = clock = perf_counter()
        dt = 0.002
        cmd = start_cmd.copy()
        q_hold = arm.get_pos().copy()

        while True:
            elapsed = perf_counter() - start
            alpha = min(1.0, elapsed / max(duration, dt))
            # Quintic time-scaling gives zero velocity and acceleration at the
            # endpoints, which makes finger closure/opening much gentler.
            smooth_alpha = 10.0 * alpha**3 - 15.0 * alpha**4 + 6.0 * alpha**5

            desired_cmd = {
                key: start_cmd[key] + smooth_alpha * (target_cmd[key] - start_cmd[key])
                for key in start_cmd
            }
            # Apply an additional per-step rate limit so the fingers cannot
            # snap closed from a large target update or timing jitter.
            rate_limits = {
                "spread": HAND_MAX_SPREAD_RATE,
                "curl_1": HAND_MAX_CURL_RATE,
                "curl_2": HAND_MAX_CURL_RATE,
                "curl_3": HAND_MAX_CURL_RATE,
            }
            for key, limit in rate_limits.items():
                max_step = limit * dt
                delta = desired_cmd[key] - cmd[key]
                cmd[key] += float(np.clip(delta, -max_step, max_step))

            arm.set_finger_pos(cmd["spread"], cmd["curl_1"], cmd["curl_2"], cmd["curl_3"])
            q = arm.get_pos()
            dq = arm.get_vel()
            q_error = q_hold - q
            q_error = np.clip(q_error, -0.02, 0.02)
            dq = np.clip(dq, -0.03, 0.03)
            tau_support = (
                nominal_gravity_torque(q)
                + HAND_HOLD_KP * q_error
                - HAND_HOLD_KD * dq
            )
            arm.set_trq(np.clip(tau_support, -MAX_TORQUES, MAX_TORQUES))
            arm.step()

            if alpha >= 1.0:
                break

            clock += dt
            idle = clock - perf_counter()
            if idle > 0:
                sleep(idle)

        self._current_cmd = target_cmd
        return arm.get_pos()

    def plan_grasp(self, object_shape: str) -> dict:
        """Plan finger joint targets for grasping an object of given shape.

        Different object shapes require different grasp strategies:
            - "cuboid":   power grasp — all three fingers wrap uniformly.
            - "cylinder": precision grasp — spread fingers wider, curl evenly.
            - "sphere":   fingertip grasp — moderate spread, light curl.

        Args:
            object_shape: one of ``"cuboid"``, ``"cylinder"``, ``"sphere"``.

        Returns:
            Dictionary with keys:
                "spread":  float — proximal joint angle [rad].
                "curl_1":  float — finger 1 medial joint angle [rad].
                "curl_2":  float — finger 2 medial joint angle [rad].
                "curl_3":  float — finger 3 medial joint angle [rad].
        """
        # ===== TODO 4.1 ======================================================
        # Design grasp configurations for each object shape.
        #
        # Consider:
        #   - Object size (all objects are ~4 cm).
        #   - Contact stability (how many contact points?).
        #   - The spread angle determines the aperture between fingers 1 & 2.
        #   - Wider spread for larger objects, narrower for small ones.
        # =====================================================================
        grasps = {
            "cuboid": {
                "spread": 0.5,
                "curl_1": 1.5,
                "curl_2": 1.5,
                "curl_3": 1.5,
            },
            "cylinder": {
                "spread": 1.2,
                "curl_1": 1.0,
                "curl_2": 1.0,
                "curl_3": 1.05,
            },
            "sphere": {
                "spread": 1.2,
                "curl_1": 1.0,
                "curl_2": 1.0,
                "curl_3": 1.05,
            },
        }
        if object_shape not in grasps:
            raise ValueError(f"Unsupported object shape: {object_shape}")
        return self._sanitize_cmd(grasps[object_shape])
        # ===== END TODO 4.1 ===================================================

    def execute_grasp(self, arm: RemoteRobotArm, grasp_config: dict) -> np.ndarray:
        """Close the fingers to the planned grasp configuration.

        Use ``arm.set_finger_pos(spread, curl_1, curl_2, curl_3)`` to
        command the Barrett Hand's 4 independent DOF.

        Args:
            arm:          robot interface.
            grasp_config: output of ``plan_grasp()`` with keys
                          ``"spread"``, ``"curl_1"``, ``"curl_2"``, ``"curl_3"``.
        """
        # ===== TODO 4.2 ======================================================
        # Close the fingers gradually for a stable grasp.
        #
        # Suggested approach:
        #   - Linearly interpolate from open (all zeros) to grasp_config
        #     over 0.5–1.0 seconds at 500 Hz.
        #   - At each step call:
        #       arm.set_finger_pos(spread, curl_1, curl_2, curl_3)
        #     along with arm.set_trq(hold_torques) + arm.step() to keep
        #     the arm stationary while the fingers close.
        #   - Use your setpoint or tracking controller to compute
        #     hold_torques for the current arm configuration.
        # =====================================================================
        return self._move_fingers(arm, grasp_config, duration=HAND_ACTUATION_TIME)
        # ===== END TODO 4.2 ===================================================

    def execute_release(self, arm: RemoteRobotArm) -> np.ndarray:
        """Open the fingers to release the grasped object.

        Use ``arm.set_finger_pos(0, 0, 0, 0)`` as the fully open target.

        Args:
            arm: robot interface.
        """
        # ===== TODO 4.2 (continued) ==========================================
        # Open the fingers by interpolating all DOF toward 0.
        # Same gradual approach as execute_grasp().
        # =====================================================================
        return self._move_fingers(
            arm,
            {"spread": 0.0, "curl_1": 0.0, "curl_2": 0.0, "curl_3": 0.0},
            duration=HAND_ACTUATION_TIME,
        )
        # ===== END TODO 4.2 ===================================================
