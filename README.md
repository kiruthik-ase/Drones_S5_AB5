## B-Spline Trajectory Planning with MILP Obstacle Avoidance and Geometric Control
![alt text](AMRIT-removebg-preview_2.png)

## **22AIE448 - Data Driven Control of Drones — Semester 5**  
## TEAM AB5
CB.SC.U4AIE24023 - KIRUTHIKPRANAV\
CB.SC.U4AIE24040 - PULIPATI CHAITHANYA\
CB.SC.U4AIE24060 - VIKRAMENDRAA SHANMUGAVELU THIYAGARAJAN\
CB.SC.U4AIE24141 - NEERAJ T\
CB.SC.U4AIE24153 - THAMMINI VARUN


> Implements and extends the method from:  
> *A. R. Babaei and M. Karimi, "Optimal Trajectory-Planning of UAVs via B-Splines and Disjunctive Programming" 2018.* - https://arxiv.org/pdf/1807.02931

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - 3.1 [B-Spline Trajectory Representation](#31-b-spline-trajectory-representation)
   - 3.2 [MILP Obstacle Avoidance (from Paper)](#32-milp-obstacle-avoidance-from-paper)
   - 3.3 [Receding Horizon Control (RHC)](#33-receding-horizon-control-rhc)
   - 3.4 [6-DOF Quadrotor Dynamics](#34-6-dof-quadrotor-dynamics)
   - 3.5 [Geometric Tracking Controller on SE(3)](#35-geometric-tracking-controller-on-se3)
   - 3.6 [Wind Turbulence Model (Dryden)](#36-wind-turbulence-model-dryden)
4. [What We Implemented from the Paper](#4-what-we-implemented-from-the-paper)
5. [What We Added Beyond the Paper](#5-what-we-added-beyond-the-paper)
6. [Scenario: Mixed-Height Urban Canyon](#6-scenario-mixed-height-urban-canyon)
7. [File Structure](#7-file-structure)
8. [Project Setup & Execution Guide](#8-project-setup--execution-guide)
9. [Output Files](#9-output-files)
10. [Key Parameters](#10-key-parameters)
11. [References](#11-references)

---

## 1. Project Overview

This project simulates an autonomous UAV navigating through a **mixed-height urban canyon** featuring tall skyscrapers (lateral avoidance), wide low hurdles (fly-over maneuvers), and No-Fly Zones. The drone reaches a goal ~6 km away while:

- **Avoiding all static buildings** (using MILP disjunctive constraints)
- **Avoiding a dynamic no-fly zone** (an intruder UAV crossing its path)
- **Rejecting wind turbulence** (Dryden atmospheric disturbance model)
- **Making decisions on-the-fly** using true closed-loop Receding Horizon Control

The core pipeline is:

```
┌──────────────────────────────────────────────────────┐
│           RECEDING HORIZON CONTROL LOOP               │
│                                                       │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────┐ │
│  │ MILP Planner│────▶│  B-Spline   │────▶│6-DOF Sim│ │
│  │ (horizon)   │     │  Reference  │     │+ Wind   │ │
│  └─────────────┘     └─────────────┘     └────┬────┘ │
│         ▲                                      │      │
│         └─────────── ACTUAL STATE ─────────────┘      │
└──────────────────────────────────────────────────────┘
```

The planner never sees pre-computed answers — it always starts from the **real, wind-disturbed position** of the drone.

---

## 2. System Architecture

```
Drones/
├── bspline_utils.py      # Cox-de Boor recursion, B-spline matrices & sampling
├── planner_obstacles.py  # UAVPlannerMILP: MILP formulation (core of paper)
├── controller.py         # Geometric tracking controller on SE(3) (Lee 2010)
├── quadrotor.py          # 6-DOF rigid-body dynamics + RK4 + WindModel
├── dynamic_obstacle.py   # Moving intruder UAV (constant velocity model)
├── scenario.py           # Scene definition: 14 buildings, 4 NFZs, start/goal
├── core.py               # run_closed_loop_rhc(): the TRUE closed-loop engine
├── simulation.py         # Run simulation, save PNG figure + interactive HTML
└── visualizer.py         # Live 3D animated GUI with HUD and telemetry
```

**Data flow at each 2-second horizon:**

1. `scenario.py` defines the static map once at startup
2. `core.py` queries `planner_obstacles.py` with the **actual** drone position
3. `planner_obstacles.py` solves the MILP and returns a 10-second B-spline
4. `core.py` runs the 6-DOF physics (`quadrotor.py`) for 2 seconds, tracking the spline with `controller.py`
5. Wind is injected every physics step. The new **real** position is fed back to step 2.

---

## 3. Mathematical Foundations

### 3.1 B-Spline Trajectory Representation

The planned trajectory is a **B-spline of degree** $k=4$ (cubic-order, $C^3$ continuous).

A B-spline curve is defined as:

$$\mathbf{p}(\tau) = \sum_{i=0}^{n} N_{i,k}(\tau)\, \mathbf{q}_i, \quad \tau \in [0, 1]$$

where:
- $\mathbf{q}_i \in \mathbb{R}^3$ are the **control points** (the MILP decision variables)
- $N_{i,k}(\tau)$ are the **B-spline basis functions** computed by the Cox-de Boor recursion

**Cox-de Boor Recursion** (implemented in `bspline_utils.py`):

$$N_{i,1}(\tau) = \begin{cases} 1 & \text{if } t_i \leq \tau < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,k}(\tau) = \frac{\tau - t_i}{t_{i+k-1} - t_i} N_{i,k-1}(\tau) + \frac{t_{i+k} - \tau}{t_{i+k} - t_{i+1}} N_{i+1,k-1}(\tau)$$

**Velocity and Acceleration** are obtained by differentiating the basis functions:

$$\dot{\mathbf{p}}(\tau) = \frac{1}{T_h}\sum_{i=0}^{n} N'_{i,k}(\tau)\, \mathbf{q}_i, \qquad \ddot{\mathbf{p}}(\tau) = \frac{1}{T_h^2}\sum_{i=0}^{n} N''_{i,k}(\tau)\, \mathbf{q}_i$$

In matrix form (how the MILP uses it):

$$\mathbf{X} = \mathbf{B}\,Q, \quad \mathbf{V} = \frac{1}{T_h}\mathbf{B}'\,Q, \quad \mathbf{A} = \frac{1}{T_h^2}\mathbf{B}''\,Q$$

where $Q \in \mathbb{R}^{n_c \times 3}$ is the matrix of all control points and $\mathbf{B}, \mathbf{B}', \mathbf{B}''$ are pre-computed basis matrices.

---

### 3.2 MILP(Mixed Integer Linear Programming) Obstacle Avoidance (from Paper)

This is the **core contribution of the reference paper**. To avoid a 3D box obstacle (cuboid), the drone must be **outside** at least one face. This is a non-convex constraint because of the "OR" logic — but it can be linearised using **Big-M disjunctive programming**.

For a cuboid obstacle with bounds $[x_l, x_u] \times [y_l, y_u] \times [z_l, z_u]$ and safety margin $\eta$, at each trajectory sample point $i$:

Introduce 6 binary variables $b^{(0)}, \ldots, b^{(5)} \in \{0,1\}$ with $\sum_{j=0}^{5} b_j = 1$ (exactly one must be active), and enforce:

$$\mathbf{X}_i[x] \leq (x_l - \eta) + (1 - b_0) \cdot M$$
$$\mathbf{X}_i[x] \geq (x_u + \eta) - (1 - b_1) \cdot M$$
$$\mathbf{X}_i[y] \leq (y_l - \eta) + (1 - b_2) \cdot M$$
$$\mathbf{X}_i[y] \geq (y_u + \eta) - (1 - b_3) \cdot M$$
$$\mathbf{X}_i[z] \leq (z_l - \eta) + (1 - b_4) \cdot M$$
$$\mathbf{X}_i[z] \geq (z_u + \eta) - (1 - b_5) \cdot M$$

When $b_j = 1$, the corresponding constraint becomes tight (active). When $b_j = 0$, the Big-M constant $M = 10000$ relaxes the constraint so it is trivially satisfied. This forces the planner to be outside **at least one face** of each obstacle.

**2D No-Fly Zones** (NFZ — infinite height, like restricted airspace) use the same logic with only 4 binary variables (no Z-face constraints).

**Total Boolean Variables per Horizon** (this project):

$$14 \text{ buildings} \times 6 \text{ sides} \times h + 4 \text{ NFZs} \times 4 \text{ sides} \times h = (84 + 16) \times 10 = 1000 \text{ binary variables}$$

---

### 3.3 Receding Horizon Control (RHC)

The full MILP over the entire 120-second flight would be computationally intractable. RHC decomposes it into a rolling sequence of short, tractable sub-problems.

**At each horizon $k$:**

1. **Plan** a B-spline over a short window $T_h = 10\,\text{s}$ starting from the **current actual state** $(\mathbf{p}_{actual}, \dot{\mathbf{p}}_{actual})$
2. **Execute** only the first $\Delta t = 2\,\text{s}$ of the plan (the committed portion)
3. **Observe** the new actual state (which differs from the plan due to wind disturbance)
4. **Repeat** from step 1

The key insight of closed-loop RHC vs. open-loop pre-computation:

| | Open-Loop | Closed-Loop (this project) |
|---|---|---|
| Planner input | Ideal state from previous plan | **Actual** wind-disturbed state |
| Wind error | Accumulates unbounded | **Corrected** at every replanning step |
| Infeasibility handling | Crashes | Graceful glide + retry |

The optimisation solved at each horizon is:

$$\min_{Q} \quad w_1 \sum_{i=1}^{h} \|\mathbf{X}_{i+1} - \mathbf{X}_i\|_1 + w_2 \|\mathbf{X}_h - \mathbf{x}_f\|_1 + w_3 \sum_{i=1}^h \mathbf{X}_i[z] + w_4 \sum_{i=1}^h \|\text{Jerk}_i\|_1$$

subject to:
- **Initial conditions:** $\mathbf{X}_0 = \mathbf{p}_{actual},\ \mathbf{V}_0 = \dot{\mathbf{p}}_{actual},\ \mathbf{A}_0 = \ddot{\mathbf{p}}_{plan}$
- **Velocity limits:** $|\mathbf{V}_i| \leq v_{max} = 60\,\text{m/s}$
- **Acceleration limits:** $|\mathbf{A}_i| \leq a_{max} = 14.715\,\text{m/s}^2\ (\approx 1.5g)$
- **Altitude bounds:** $1\,\text{m} \leq \mathbf{X}_i[z] \leq 25\,\text{m}$
- **Big-M obstacle avoidance:** (as described in §3.2)

---

### 3.4 6-DOF Quadrotor Dynamics

The quadrotor is modelled as a rigid body with the **12-state vector**:

$$\mathbf{s} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \boldsymbol{\Phi} \\ \boldsymbol{\omega} \end{bmatrix} = \begin{bmatrix} x, y, z \\ v_x, v_y, v_z \\ \phi, \theta, \psi \\ p, q, r \end{bmatrix}$$

**Translational Dynamics** (Newton's second law + wind disturbance):

$$m\ddot{\mathbf{p}} = \mathbf{R}\begin{bmatrix}0\\0\\T\end{bmatrix} - mg\mathbf{e}_3 + m\mathbf{w}$$

where $T$ is total thrust, $\mathbf{R} \in SO(3)$ is the rotation matrix, and $\mathbf{w}$ is the wind acceleration disturbance.

**Rotation Matrix** (ZYX Euler convention: yaw $\psi$, pitch $\theta$, roll $\phi$):

$$\mathbf{R} = R_z(\psi)\,R_y(\theta)\,R_x(\phi) = \begin{bmatrix} c_\psi c_\theta & c_\psi s_\theta s_\phi - s_\psi c_\phi & c_\psi s_\theta c_\phi + s_\psi s_\phi \\ s_\psi c_\theta & s_\psi s_\theta s_\phi + c_\psi c_\phi & s_\psi s_\theta c_\phi - c_\psi s_\phi \\ -s_\theta & c_\theta s_\phi & c_\theta c_\phi \end{bmatrix}$$

**Rotational Dynamics** (Euler's equation):

$$\mathbf{I}\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})$$

where $\mathbf{I} = \text{diag}(I_{xx}, I_{yy}, I_{zz}) = \text{diag}(0.0196,\, 0.0196,\, 0.0264)\,\text{kg·m}^2$.

**Euler Angle Kinematics:**

$$\dot{\boldsymbol{\Phi}} = \begin{bmatrix} 1 & s_\phi \tan\theta & c_\phi \tan\theta \\ 0 & c_\phi & -s_\phi \\ 0 & s_\phi / c_\theta & c_\phi / c_\theta \end{bmatrix} \boldsymbol{\omega}$$

**Integration:** 4th-order Runge-Kutta (RK4) at $\Delta t_{sim} = 0.002\,\text{s}$ (500 Hz):

$$\mathbf{s}(t + \Delta t) = \mathbf{s}(t) + \frac{\Delta t}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)$$

---

### 3.5 Geometric Tracking Controller on SE(3)

Based on **T. Lee, M. Leok, N.H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010.**

Unlike conventional PID controllers, the geometric controller operates directly on the **Special Euclidean group** SE(3), avoiding the singularities inherent in Euler-angle representations (gimbal lock). The control law is:

**Step 1 — Desired Force:**

$$\mathbf{F}_{des} = -K_p\,\mathbf{e}_p - K_v\,\mathbf{e}_v + mg\mathbf{e}_3 + m\,\ddot{\mathbf{p}}_d$$

where $\mathbf{e}_p = \mathbf{p} - \mathbf{p}_d$ and $\mathbf{e}_v = \dot{\mathbf{p}} - \dot{\mathbf{p}}_d$.

Gains are set for a critically-damped position loop ($\omega_n \approx 4\,\text{rad/s}$):

$$K_p = \text{diag}(16, 16, 20), \quad K_v = \text{diag}(8, 8, 9)$$

**Step 2 — Total Thrust:**

$$T = \mathbf{F}_{des} \cdot \mathbf{b}_3 = \mathbf{F}_{des} \cdot \mathbf{R}\,\mathbf{e}_3$$

**Step 3 — Desired Rotation Matrix $\mathbf{R}_{des}$:**

The desired body-z axis points along the force direction:

$$\mathbf{b}_{3,des} = \frac{\mathbf{F}_{des}}{\|\mathbf{F}_{des}\|}$$

Given a desired heading $\mathbf{b}_{1,c} = [\cos\psi_d,\, \sin\psi_d,\, 0]^\top$:

$$\mathbf{b}_{2,des} = \frac{\mathbf{b}_{3,des} \times \mathbf{b}_{1,c}}{\|\mathbf{b}_{3,des} \times \mathbf{b}_{1,c}\|}, \quad \mathbf{b}_{1,des} = \mathbf{b}_{2,des} \times \mathbf{b}_{3,des}$$

$$\mathbf{R}_{des} = \begin{bmatrix} \mathbf{b}_{1,des} & \mathbf{b}_{2,des} & \mathbf{b}_{3,des} \end{bmatrix}$$

**Step 4 — Attitude Error on SO(3):**

$$\mathbf{e}_R = \frac{1}{2}\,\text{vee}\!\left(\mathbf{R}_{des}^\top\mathbf{R} - \mathbf{R}^\top\mathbf{R}_{des}\right)$$

where the **vee map** $\text{vee}(\cdot) : \mathfrak{so}(3) \to \mathbb{R}^3$ extracts a 3-vector from a skew-symmetric matrix.

**Step 5 — Control Torques:**

$$\boldsymbol{\tau} = -K_R\,\mathbf{e}_R - K_\omega\,\mathbf{e}_\omega + \boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega}$$

Gains are set for a much faster attitude loop ($\omega_n \approx 30\,\text{rad/s}$):

$$K_R = \text{diag}(120, 120, 60), \quad K_\omega = \text{diag}(16, 16, 8)$$

---

### 3.6 Wind Turbulence Model (Dryden)

The atmospheric wind is modelled as a **first-order Markov (auto-regressive) process** inspired by the Dryden Wind Turbulence Model (MIL-HDBK-1797):

$$\mathbf{w}[k] = \rho\,\mathbf{w}[k-1] + \boldsymbol{\sigma} \odot \sqrt{1 - \rho^2}\;\mathbf{n}[k], \quad \mathbf{n}[k] \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

Parameters:
- $\rho = 0.995$ — high autocorrelation gives **slow, persistent gusts** (realistic atmospheric turbulence)
- $\boldsymbol{\sigma} = [1.8,\, 1.8,\, 0.4]\,\text{m/s}^2$ — moderate horizontal turbulence, low vertical

The wind acceleration is injected directly into the quadrotor's translational dynamics at every 2 ms physics step. Because the RHC planner uses the **actual** wind-pushed position at each replanning step, the controller continuously compensates for accumulated wind drift.

---

## 4. What We Implemented from the Paper

All core algorithmic elements from Babaei & Karimi (2018) are implemented:

| Component | Paper Section | Implementation |
|---|---|---|
| **B-spline trajectory** | §II | `bspline_utils.py` — Cox-de Boor recursion, matrix form |
| **MILP formulation** | §III | `planner_obstacles.py` — `UAVPlannerMILP.plan()` |
| **Big-M disjunctive obstacles** | §III-A | Binary variables per building face |
| **No-fly zone constraints** | §III-B | 2D NFZ with 4-face disjunction |
| **Velocity & acceleration bounds** | §III-C | Per-sample linear constraints |
| **Altitude ceiling constraint** | §III | $z \leq z_{max}$ forces lateral avoidance |
| **Trajectory objectives** | §IV | Path length + target approach + altitude + jerk |
| **Receding Horizon structure** | §V | Rolling 2-second execution window |

---

## 5. What We Added Beyond the Paper

The following components are **original extensions** not present in the reference paper:

| Addition | Description |
|---|---|
| **True Closed-Loop RHC** | The paper uses pre-computed open-loop. We feed the **actual** 6-DOF physics state back to the planner at every horizon, correcting for wind drift in real-time. |
| **6-DOF Quadrotor Dynamics** | Full rigid-body simulation with RK4 at 500 Hz. The paper only plans trajectories; we simulate whether a real drone could follow them. |
| **Geometric Controller (SE3)** | Lee et al. (2010) attitude and position controller that operates on the rotation manifold SO(3), avoiding gimbal-lock singularities. |
| **Dryden Wind Turbulence** | First-order Markov wind disturbance model. The drone is persistently pushed off-course and must actively correct. |
| **Dynamic Obstacle (intruder UAV)** | A second UAV flies across the scene at constant velocity. Its predicted position is injected as a temporary NFZ at each horizon. If it causes infeasibility, the planner gracefully retries without it. |
| **Dense Urban Canyon Scenario** | 14 buildings + 4 NFZs in a complex S-curve layout requiring 7 distinct lateral manoeuvres. The paper only shows simple 2–3 building scenarios. |
| **Interactive 3D HTML viewer** | `trajectory_3d.html` — fully rotatable browser-based 3D view using Plotly, useful for inspecting the flight path from any angle. |
| **Result Caching** | Simulation results are compressed and cached to `simulation_cache.npz`. The visualiser loads in 0.06s instead of re-running the 90-second MILP computation. |
| **Live GUI Visualiser** | Real-time animation with tilted drone body, wind vector, HUD telemetry, telemetry strip charts, and spacebar-pause for 3D view rotation. |

---

## 6. Scenario: Mixed-Height Urban Canyon

```
  START [2500, -1400, 12m]
       │
       ├── TALL GATE A (200m)  ─── Thread laterally through gap
       ├── LOW HURDLE 1 (15m)  ─── Climb over (Z >= 20m)
       ├── TALL GATE B (200m)  ─── Thread laterally through gap
       ├── LOW HURDLE 2 (15m)  ─── Climb over (Z >= 20m)
       ├── TALL GATE C (200m)  ─── Thread laterally through gap
       ├── LOW HURDLE 3 (15m)  ─── Climb over (Z >= 20m)
       └── TALL GATE D (200m)  ─── Final gate approach into goal
       │
  GOAL [-2200, 2000, 12m]
```

- **Flight Duration:** ~88 seconds (44 RHC horizons)
- **Tall Buildings (200 m):** Taller than altitude ceiling ($z_{max}=25\,\text{m}$), forcing lateral navigation.
- **Low Hurdles (15 m):** Short wide walls where overflying is mathematically optimal ($z \ge 20\,\text{m}$).
- **No-Fly Zones:** 3 restricted airspace zones strategically blocking trivial bypass routes.

---

## 7. File Structure

| File | Purpose |
|---|---|
| `requirements.txt` | Python dependencies list |
| `bspline_utils.py` | Cox-de Boor recursion; `get_bspline_matrices()`; `sample_trajectory()` |
| `planner_obstacles.py` | `UAVPlannerMILP` class: MILP problem construction and solve |
| `controller.py` | `GeometricController.compute()`: SE(3) tracking law |
| `quadrotor.py` | `Quadrotor` (6-DOF RK4); `WindModel` (Dryden Markov) |
| `dynamic_obstacle.py` | `DynamicObstacle`: constant-velocity intruder + NFZ projection |
| `scenario.py` | `build_scenario()`: registers mixed-height buildings and NFZs |
| `core.py` | `run_closed_loop_rhc()`: the main interleaved plan-simulate loop |
| `simulation.py` | Entry point: runs simulation, saves PNG figure + HTML viewer |
| `visualizer.py` | Live GUI: 3D animation, 2D map, telemetry strips |
| `simulation_cache.npz` | Auto-generated cache for instant visualizer launch |
| `simulation_results.png` | Auto-generated 6-panel publication figure |
| `trajectory_3d.html` | Auto-generated interactive 3D viewer for web browsers |

---

## 8. Project Setup & Execution Guide

Follow these step-by-step instructions to set up and run the project from scratch on any new device.

### 8.1 System Requirements
- **Operating System:** Windows 10/11, macOS, or Linux
- **Python:** Python 3.10 or higher installed ([Download Python](https://www.python.org/downloads/))
- **RAM:** Minimum 4 GB

---

### 8.2 Step-by-Step Installation

#### Step 1: Open Terminal / Command Prompt
Open PowerShell (Windows) or Terminal (macOS/Linux) and navigate to the project directory:
```bash
cd /path/to/Drones
```

#### Step 2: Create a Virtual Environment
Creating a virtual environment ensures dependencies do not conflict with your global Python setup:
```bash
# Windows / macOS / Linux
python -m venv venv
```

#### Step 3: Activate the Virtual Environment
- **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
  *(If you get a script execution policy error, run this first: `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`)*

- **Windows (Command Prompt / CMD):**
  ```cmd
  venv\Scripts\activate.bat
  ```

- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

#### Step 4: Upgrade pip & Install Dependencies
Install all required packages via `requirements.txt`:
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### 8.3 Running the Project

#### Mode A: Run Full Simulation (Generates Graphs & Cache)
Runs the closed-loop RHC simulation, prints per-horizon solver progress in terminal, and outputs both a publication-ready 6-panel chart and an interactive 3D web model:
```bash
python simulation.py
```
**Outputs generated:**
- `simulation_results.png` — 6-panel plot (3D trajectory, top-down map, altitude/wind, error metrics, control signals).
- `trajectory_3d.html` — Standalone 3D interactive model (open in any web browser to rotate, pan, and inspect).
- `simulation_cache.npz` — Cached flight trajectory data for instant animation playback.

#### Mode B: Launch the Real-Time 3D Visualizer GUI
Loads the simulation and launches the live interactive 3D visualization window:
```bash
python visualizer.py
```

**GUI Keyboard & Mouse Controls:**
| Control | Action |
|---|---|
| `SPACE` | **Pause / Resume** animation |
| **Mouse Left-Click Drag** | Rotate 3D camera angle *(while paused)* |
| **Mouse Scroll Wheel** | Zoom in / Zoom out |
| **Close Window** | Exit GUI |

#### Mode C: Force Fresh Recomputation
If you edit building positions or controller gains and want to recompute directly from the GUI without loading old cache:
```bash
python visualizer.py --fresh
```

#### Mode D: Export Video / GIF Animation
To render and save the full flight animation directly to a video/GIF file:
```bash
python visualizer.py --save
```
*(Saves `flight.mp4` if `ffmpeg` is available on the system; otherwise automatically falls back to `flight.gif`)*

---

### 8.4 Troubleshooting Common Issues

1. **"Execution of scripts is disabled on this system" (Windows PowerShell):**
   - Solution: Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process` in PowerShell, then re-run `.\venv\Scripts\Activate.ps1`.
2. **Tkinter / GUI window does not appear on Linux:**
   - Solution: Install python-tk via your package manager: `sudo apt-get install python3-tk`
3. **Solver errors or cvxpy installation issues:**
   - Solution: Ensure you are using Python $\le 3.12$ and run `pip install --upgrade cvxpy scipy`.


---

## 9. Output Files

| File | Description |
|---|---|
| `simulation_results.png` | 6-panel figure: 3D path, top-down map, altitude, speed, tracking error, wind |
| `trajectory_3d.html` | Interactive Plotly 3D scene with buildings, NFZs, and trajectory |
| `flight.gif` / `flight.mp4` | Animation export from `python visualizer.py --save` |
| `simulation_cache.npz` | Compressed numpy archive of all simulation logs |

---

## 10. Key Parameters

| Parameter | Value | Location | Effect |
|---|---|---|---|
| Spline degree $k$ | 4 | `planner_obstacles.py` | $C^3$ smoothness |
| Horizon steps $h$ | 10 | `planner_obstacles.py` | 10 s lookahead |
| Execute time $\Delta t$ | 2.0 s | `core.py` | Replanning frequency |
| Physics timestep $\delta t$ | 0.002 s | `core.py` | 500 Hz simulation |
| Max velocity $v_{max}$ | 60 m/s | `planner_obstacles.py` | Speed limit |
| Max acceleration $a_{max}$ | 14.715 m/s² | `planner_obstacles.py` | 1.5g thrust cap |
| Altitude ceiling $z_{max}$ | 25 m | `planner_obstacles.py` | Below building tops |
| Safety margin $\eta$ | 80 m | `scenario.py` | Building clearance |
| Big-M constant $M$ | 10000 | `planner_obstacles.py` | Disjunctive relaxation |
| Wind autocorrelation $\rho$ | 0.995 | `quadrotor.py` | Persistent gusts |
| Wind std-dev $\sigma$ | [1.8, 1.8, 0.4] m/s² | `quadrotor.py` | Turbulence intensity |
| Position gain $K_p$ | diag(16,16,20) | `controller.py` | $\omega_n \approx 4$ rad/s |
| Attitude gain $K_R$ | diag(120,120,60) | `controller.py` | $\omega_n \approx 30$ rad/s |

---

## 11. References

1. **A. R. Babaei and M. Karimi**, "Optimization of UAV Trajectory in Urban Area," *2018* — Core MILP + B-spline planning method.

2. **T. Lee, M. Leok, N. H. McClamroch**, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," *49th IEEE Conference on Decision and Control (CDC)*, 2010, pp. 5420–5425. — Geometric controller on SE(3).

3. **MIL-HDBK-1797**, "Flying Qualities of Piloted Aircraft," US Department of Defense, 1997. — Dryden wind turbulence model specification.

4. **L. Piegl and W. Tiller**, *The NURBS Book*, 2nd ed., Springer, 1997. — B-spline theory and Cox-de Boor algorithm.

5. **cvxpy** — Diamond & Boyd, "CVXPY: A Python-embedded modeling language for convex optimization," *JMLR*, 2016.