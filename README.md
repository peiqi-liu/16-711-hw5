# Simulator Setup

## Environment Setup

This project uses [`uv`](https://docs.astral.sh/uv/) for dependency management. Install `uv` first:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

From the `assignment/` directory, create and populate the virtual environment:

```bash
uv sync
```

**Activate the venv in every terminal you use for this assignment** (both the simulator terminal and your code terminal):

```bash
source .venv/bin/activate
```

You should see `(assignment)` or similar in your prompt once activated. All commands below assume the venv is active.

## Starting the Simulator

The physics server runs as a standalone process. Open a terminal and execute:

```
python server.pyc
```

This launches the simulator with a built-in web-based visualiser. Once the server is running, open a *second* terminal and run your code:

```
python main.py 1a
```

## Communication Protocol

Your controller interacts with the server through the `RemoteRobotArm` class in `interface.py`. A single control cycle is:

1. `arm.set_trq(tau)` — send a 7-D torque vector over UDP.
2. `arm.step()` — block until the server applies the torque, steps physics by $\Delta t = 0.002$ s, and replies with the new state.
3. `arm.get_pos()`, `arm.get_vel()` — read the cached joint positions $q$ and velocities $\dot{q}$.

A minimal real-time control loop looks like:

```python
from time import perf_counter, sleep
import numpy as np
from interface import RemoteRobotArm
from utils import Logger

arm = RemoteRobotArm()
arm.reset()
sleep(2.0)

log = Logger()
DT = 0.002
tau = np.zeros(7)
start = clock = perf_counter()
while (elapsed := perf_counter() - start) < 5.0:
    # 1. Send torques computed on the previous iteration.
    arm.set_trq(tau)
    # 2. Advance the simulator and receive the new state.
    arm.step()
    # 3. Read the fresh state and compute the next torque.
    q  = arm.get_pos()
    dq = arm.get_vel()
    # --- your controller here ---
    # tau = ...
    log.record(t=elapsed, q=q, dq=dq, q_des=q_des, tau=tau)
    # 4. Real-time pacing at 500 Hz.
    clock += DT
    sleep(max(0, clock - perf_counter()))
```


### Attach / Detach

For Questions 2 and 3, the simulator provides a simplified grasping mechanism:

- `arm.attach("item1")` — creates a rigid weld constraint between the end-effector and the named object, provided the end-effector is within grasp range.
- `arm.detach()` — removes the constraint, releasing the object.

Allow ~0.3 s after each call for the constraint to settle in the simulation.

# Robot and Environment Specification

The robot base is fixed at the world-frame origin at height $z = 0.6$ m. Full nominal link inertials (masses, centres of mass, diagonal inertia tensors) are provided in `robot_manual.md`.

| Object   | Centre [m]            | Dimensions                                                  |
|----------|-----------------------|-------------------------------------------------------------|
| Table    | $(0.6,\ 0,\ 0.45)$    | $0.8 \times 1.6 \times 0.9$ m; top surface at $z = 0.9$ m   |
| Barrier  | $(0.8,\ 0,\ 1.05)$    | $0.4 \times 0.04 \times 0.3$ m; spans $z \in [0.9,\ 1.2]$ m |
| Drop box | $(0.6,\ -0.4,\ 0.9)$  | $0.28 \times 0.28$ m inner footprint; wall height $0.1$ m   |

Three objects rest on the *left* side of the barrier ($y > 0$):

| Name    | Shape    | Pos [m]              | Size                      | Mass [kg] |
|---------|----------|----------------------|---------------------------|-----------|
| `item1` | Cuboid   | $(0.5,\ 0.3,\ 0.95)$ | $4 \times 4 \times 4$ cm  | 0.1       |
| `item2` | Cylinder | $(0.6,\ 0.4,\ 0.95)$ | $r = 2$ cm, $h = 4$ cm    | 0.1       |
| `item3` | Sphere   | $(0.7,\ 0.3,\ 0.95)$ | $r = 2$ cm                | 0.1       |
