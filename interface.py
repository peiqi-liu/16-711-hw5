"""
16-711 Kinematics Dynamics and Control — Robot Arm Interface
==================================================
Provides ``RemoteRobotArm``, the sole communication channel between your
controller code and the physics simulator running in ``server.pyc``.

Do not modify this file (except for the Bonus question, Question 4).
"""

import select
import socket
import struct
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional
from config import SimConfig, CMD_RESET, CMD_ESTOP, CMD_ATTACH, CMD_DETACH, CMD_FINGER, CONFIG


class BaseRobotArm(ABC):
    """Abstract interface that every robot arm implementation must satisfy."""

    def __init__(self, control_dt: Optional[float] = None):
        self.control_dt = control_dt

    @property
    @abstractmethod
    def num_joints(self) -> int:
        """Dimensionality of the joint space."""
        ...

    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Ordered names of the joints."""
        ...

    @property
    @abstractmethod
    def actuator_names(self) -> tuple[str, ...]:
        """Ordered names of the actuators."""
        ...

    @abstractmethod
    def set_trq(self, tau: np.ndarray) -> None:
        """Send joint torques to the robot (Zero-Order Hold)."""
        ...

    @abstractmethod
    def step(self) -> None:
        """Advance one simulation step; blocks until new state is available."""
        ...

    @abstractmethod
    def get_pos(self) -> np.ndarray:
        """Read current joint positions q  (n_joints,)."""
        ...

    @abstractmethod
    def get_vel(self) -> np.ndarray:
        """Read current joint velocities dq (n_joints,)."""
        ...

    @abstractmethod
    def get_frc(self) -> float:
        """Read scalar contact-force magnitude at the force sensor."""
        ...

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger an emergency stop — zeros all torques on the server."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset the simulation to the home configuration."""
        ...

    @abstractmethod
    def attach(self, object_name: str) -> None:
        """Rigidly attach *object_name* to the end-effector."""
        ...

    @abstractmethod
    def detach(self) -> None:
        """Release any object currently attached to the end-effector."""
        ...

    @abstractmethod
    def set_finger_pos(self, spread: float, curl_1: float, curl_2: float, curl_3: float) -> None:
        """Command Barrett Hand finger positions (Bonus question).

        Args:
            spread: proximal joint angle for fingers 1 & 2 [rad], range [0, pi].
            curl_1: finger 1 medial joint angle [rad], range [0, 2.443].
            curl_2: finger 2 medial joint angle [rad], range [0, 2.443].
            curl_3: finger 3 medial joint angle [rad], range [0, 2.443].
        """
        ...


class RemoteRobotArm(BaseRobotArm):
    """Networked robot arm using dual-channel UDP communication.

    Control channel (isochronous, 500 Hz):
        Client  -->  server:  7 doubles (torques)
        Server  -->  client:  15 doubles (q[7], dq[7], frc[1])

    Auxiliary channel (asynchronous):
        Client  -->  server:  UTF-8 command string
    """

    def __init__(self, config: SimConfig = CONFIG, control_dt: Optional[float] = None):
        super().__init__(control_dt)
        self.config = config
        self._ip = config.udp.ip
        self._control_addr = (self._ip, config.udp.robot_control_port)
        self._aux_addr = (self._ip, config.udp.robot_aux_port)

        # Control channel
        self.control_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.control_sock.settimeout(config.udp.timeout_s)

        # Auxiliary channel
        self.aux_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        # Internal state cache
        self._n = 7
        self._q = np.zeros(self._n)
        self._dq = np.zeros(self._n)
        self._frc_cached = 0.0

        # Protocol sizes
        self._control_fmt = f"{self._n}d"
        self._state_fmt = f"{2 * self._n + 1}d"
        self._state_size = struct.calcsize(self._state_fmt)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_joints(self) -> int:
        return self._n

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(self.config.robot.joint_names)

    @property
    def actuator_names(self) -> tuple[str, ...]:
        return tuple(self.config.robot.actuator_names)

    # ------------------------------------------------------------------
    # Control channel
    # ------------------------------------------------------------------
    def set_trq(self, tau: np.ndarray) -> None:
        if len(tau) != self._n:
            raise ValueError(f"Expected {self._n}-D torque, got {len(tau)}")
        packet = struct.pack(self._control_fmt, *tau)
        self.control_sock.sendto(packet, self._control_addr)

    def step(self) -> None:
        """Block until the server completes one physics step and returns state."""
        try:
            data, _ = self.control_sock.recvfrom(4096)
            # Drain to the latest packet if multiple are buffered.
            # NOTE: cannot use MSG_DONTWAIT here — Python's timeout-mode
            # socket does not honor it and blocks for ~100 ms per call.
            while select.select([self.control_sock], [], [], 0)[0]:
                data, _ = self.control_sock.recvfrom(4096)
            unpacked = struct.unpack(self._state_fmt, data[: self._state_size])
            self._q = np.array(unpacked[0 : self._n])
            self._dq = np.array(unpacked[self._n : 2 * self._n])
            self._frc_cached = unpacked[2 * self._n]
        except socket.timeout:
            pass

    # ------------------------------------------------------------------
    # Sensor readings (from cache populated by step())
    # ------------------------------------------------------------------
    def get_pos(self) -> np.ndarray:
        return self._q.copy()

    def get_vel(self) -> np.ndarray:
        return self._dq.copy()

    def get_frc(self) -> float:
        return self._frc_cached

    # ------------------------------------------------------------------
    # Auxiliary channel commands
    # ------------------------------------------------------------------
    def emergency_stop(self) -> None:
        self.aux_sock.sendto(CMD_ESTOP.encode(), self._aux_addr)

    def reset(self) -> None:
        self.aux_sock.sendto(CMD_RESET.encode(), self._aux_addr)

    def attach(self, object_name: str) -> None:
        """Attach *object_name* to the end-effector via a rigid constraint.

        The simulator creates a weld between the palm link and the named
        object provided the end-effector is within grasp range.

        Args:
            object_name: One of ``"item1"``, ``"item2"``, ``"item3"``.
        """
        msg = f"{CMD_ATTACH}:{object_name}"
        self.aux_sock.sendto(msg.encode(), self._aux_addr)

    def detach(self) -> None:
        """Release the currently attached object."""
        self.aux_sock.sendto(CMD_DETACH.encode(), self._aux_addr)

    def set_finger_pos(self, spread: float, curl_1: float, curl_2: float, curl_3: float) -> None:
        """Command Barrett Hand finger positions via the auxiliary channel.

        The simulator maps the 4 independent DOF to the 8 finger
        actuators (including coupled distal joints).

        Args:
            spread: proximal joint angle for fingers 1 & 2 [rad].
            curl_1: finger 1 medial joint angle [rad].
            curl_2: finger 2 medial joint angle [rad].
            curl_3: finger 3 medial joint angle [rad].
        """
        msg = f"{CMD_FINGER}:{spread},{curl_1},{curl_2},{curl_3}"
        self.aux_sock.sendto(msg.encode(), self._aux_addr)
