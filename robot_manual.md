# Robot Control Manual
Welcome to the robot arm simulation environment.

## 1. Establishing Communication
Run the server from this directory in a separate terminal:
```bash
python server.pyc
```
The robot simulation operates with two channels:
*   **Control Channel (Isochronous):** `127.0.0.1:5005`
*   **Service Channel (Asynchronous):** `127.0.0.1:5006`

## 2. Global Parameters
*   **Gravity:** (0.0, 0.0, -9.81) m/s^2
*   **Control Frequency:** 500.00 Hz (dt: 0.002s)

## 3. Nominal Robot Model
### Link Inertials
| Link Name | Mass (kg) | COM (x y z) | Diag Inertia (kgm^2) |
| :-- | :-- | :-- | :-- |
| base_link | 1 | 0 0 0 | 0.1 0.1 0.1 |
| shoulder_yaw_link | 5 | -0.00443422 -0.00066489 -0.128904 | 0.135089 0.113095 0.0904426 |
| shoulder_pitch_link | 3.87494 | -0.00236981 -0.0154211 0.0310561 | 0.0214195 0.0167127 0.0126452 |
| upper_arm_link | 2.20228 | 0.00683259 3.309e-005 0.392492 | 0.0592718 0.0592207 0.00313419 |
| forearm_link | 0.500168 | -0.0400149 -0.142717 -0.00022942 | 0.0151047 0.0148285 0.00275805 |
| wrist_yaw_link | 1.05376 | 8.921e-005 0.00435824 -0.00511217 | 0.000555168 0.00046317 0.000234072 |
| wrist_pitch_link | 0.517974 | -0.00012262 -0.0246834 -0.0170319 | 0.000555168 0.00046317 0.000234072 |
| wrist_palm_link | 0.0828613 | 0 0 0.055 | 0.00020683 0.00010859 0.00010851 |

### Joint Properties
| Joint Name | Range (rad) | Damping (Nms/rad) | Max Torque (Nm) |
| :-- | :-- | :-- | :-- |
| base_yaw_joint | -2.6 2.6 | 1.98 | 150.0 |
| shoulder_pitch_joint | -1.985 1.985 | 0.55 | 125.0 |
| shoulder_yaw_joint | -2.8 2.8 | 1.65 | 60.0 |
| elbow_pitch_joint | -0.9 3.14159 | 0.88 | 60.0 |
| wrist_yaw_joint | -4.55 1.25 | 0.55 | 30.0 |
| wrist_pitch_joint | -1.5707 1.5707 | 0.11 | 30.0 |
| palm_yaw_joint | -3 3 | 0.11 | 10.0 |


## 4. Getting Started (Python Example)
```python
from time import perf_counter, sleep
import numpy as np
from interface import RemoteRobotArm

def main():
    arm = RemoteRobotArm()
    arm.reset()
    sleep(2.0)
    TARGET_dt = 0.002
    MAX_TIME = 5.0
    tau = np.zeros(7)
    start_time = clock = perf_counter()
    try:
        while perf_counter() - start_time < MAX_TIME:
            arm.set_trq(tau)
            arm.step()
            q = arm.get_pos()
            clock += TARGET_dt
            idle = clock - perf_counter()
            sleep(max(0.0, idle))
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    main()
```
