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
from controller import BaseController
from kinematics import forward_kinematics, jacobian, inverse_kinematics
from trajectory import QuinticTrajectory
from config import (
    BARRIER_TOP_Z,
    DROP_BOX_CENTER,
    DROP_BOX_FLOOR_Z,
    ITEM_POSITIONS,
    ITEM_DIMS,
)
from utils import Logger


# ======================================================================
#  Constants
# ======================================================================

CLEARANCE_Z = BARRIER_TOP_Z + 0.10  # safe height above barrier [m]
APPROACH_OFFSET_Z = 0.08             # height above object for approach [m]
SEGMENT_DURATION = 2.0               # default time per trajectory segment [s]

# Palm-down end-effector orientation (world frame).
# Use as ``target_rot`` in ``inverse_kinematics`` so the palm faces
# the table during approach/grasp/place.
PALM_DOWN = np.array([
    [1.0,  0.0,  0.0],
    [0.0, -1.0,  0.0],
    [0.0,  0.0, -1.0],
])


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
        raise NotImplementedError("TODO 2.2: Implement PickAndPlaceTask.plan_cartesian_path()")
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
        raise NotImplementedError("TODO 2.3: Implement PickAndPlaceTask.execute()")
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
        raise NotImplementedError("TODO 3.1: Implement StackingTask.compute_stack_poses()")
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
        raise NotImplementedError("TODO 3.2: Implement StackingTask.execute()")
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
        raise NotImplementedError("TODO 4.1: Implement BarrettHandController.plan_grasp()")
        # ===== END TODO 4.1 ===================================================

    def execute_grasp(self, arm: RemoteRobotArm, grasp_config: dict) -> None:
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
        raise NotImplementedError("TODO 4.2: Implement BarrettHandController.execute_grasp()")
        # ===== END TODO 4.2 ===================================================

    def execute_release(self, arm: RemoteRobotArm) -> None:
        """Open the fingers to release the grasped object.

        Use ``arm.set_finger_pos(0, 0, 0, 0)`` as the fully open target.

        Args:
            arm: robot interface.
        """
        # ===== TODO 4.2 (continued) ==========================================
        # Open the fingers by interpolating all DOF toward 0.
        # Same gradual approach as execute_grasp().
        # =====================================================================
        raise NotImplementedError("TODO 4.2: Implement BarrettHandController.execute_release()")
        # ===== END TODO 4.2 ===================================================
