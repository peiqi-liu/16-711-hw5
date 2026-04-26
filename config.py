"""
16-711 Kinematics Dynamics and Control — Assignment Configuration
======================================================
Simulation parameters and communication protocol constants.
Do not modify this file.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class RobotConfig:
    """Nominal robot hardware specification."""
    joint_names: list[str] = field(default_factory=list)
    actuator_names: list[str] = field(default_factory=list)
    home_pos: np.ndarray = field(default_factory=lambda: np.zeros(7))


@dataclass
class UdpConfig:
    """Network communication settings."""
    ip: str = "127.0.0.1"
    robot_control_port: int = 5005
    robot_aux_port: int = 5006
    timeout_s: float = 0.1


@dataclass
class SimConfig:
    """Top-level simulation configuration."""
    physics_dt: float = 0.002
    robot: RobotConfig = field(default_factory=RobotConfig)
    udp: UdpConfig = field(default_factory=UdpConfig)


# ---------------------------------------------------------------------------
# Public hardware specification
# ---------------------------------------------------------------------------
CONFIG = SimConfig(
    physics_dt=0.002,
    robot=RobotConfig(
        joint_names=[
            "base_yaw_joint",
            "shoulder_pitch_joint",
            "shoulder_yaw_joint",
            "elbow_pitch_joint",
            "wrist_yaw_joint",
            "wrist_pitch_joint",
            "palm_yaw_joint",
        ],
        actuator_names=[
            "base_yaw_motor",
            "shoulder_pitch_motor",
            "shoulder_yaw_motor",
            "elbow_pitch_motor",
            "wrist_yaw_motor",
            "wrist_pitch_motor",
            "palm_yaw_motor",
        ],
        home_pos=np.array([
            0.348568, 0.921571, 0.190382, 1.735326, 0.000439, 0.036126, 0.0
        ]),
    ),
    udp=UdpConfig(),
)

# ---------------------------------------------------------------------------
# Auxiliary-channel command strings
# ---------------------------------------------------------------------------
CMD_RESET  = "RESET"
CMD_ESTOP  = "ESTOP"
CMD_ATTACH = "ATTACH"
CMD_DETACH = "DETACH"
CMD_FINGER = "FINGER"

# ---------------------------------------------------------------------------
# Workspace geometry (metres, world frame)
# ---------------------------------------------------------------------------
TABLE_CENTER  = np.array([0.6, 0.0, 0.45])
TABLE_HALF    = np.array([0.4, 0.8, 0.45])
TABLE_TOP_Z   = 0.9

BARRIER_CENTER = np.array([0.8, 0.0, 1.05])
BARRIER_HALF   = np.array([0.2, 0.02, 0.15])
BARRIER_TOP_Z  = 1.20

DROP_BOX_CENTER = np.array([0.5, -0.2, 0.9])
DROP_BOX_FLOOR_Z = 0.92  # top surface of box bottom

# Item initial positions (left side, y > 0)
ITEM_POSITIONS = {
    "item1": np.array([0.45, 0.25, 0.925]),   # cuboid
    "item2": np.array([0.25, 0.50, 0.925]),   # cylinder
    "item3": np.array([0.925, 0.25, 0.925]),  # sphere 
}
# ITEM_POSITIONS = {
#     "item1": np.array([0.45, 0.25, 0.925]),   # cuboid
#     "item2": np.array([0.56, 0.49, 0.925]),   # cylinder
#     "item3": np.array([0.925, 0.25, 0.925]),  # sphere 
# }

# Item geometric dimensions (half-extents or radii)
ITEM_DIMS = {
    "item1": {"shape": "cuboid",   "half_size": np.array([0.02, 0.02, 0.02])},
    "item2": {"shape": "cylinder", "radius": 0.02, "half_height": 0.02},
    "item3": {"shape": "sphere",   "radius": 0.02},
}

# Joint limits (rad)
JOINT_LIMITS_LOWER = np.array([-2.6, -1.985, -2.8, -0.9, -4.55, -1.5707, -3.0])
JOINT_LIMITS_UPPER = np.array([ 2.6,  1.985,  2.8,  3.14159, 1.25, 1.5707,  3.0])

# Maximum joint torques (Nm)
MAX_TORQUES = np.array([150.0, 125.0, 60.0, 60.0, 30.0, 30.0, 10.0])