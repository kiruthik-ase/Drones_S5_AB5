# B-Spline Trajectory Planning with MILP Obstacle Avoidance and Geometric Control

![Team Banner](AMRIT-removebg-preview_2.png)

**22AIE448 — Data Driven Control of Drones · Semester 5**  
**Team AB5**

| Roll No. | Name |
|---|---|
| CB.SC.U4AIE24023 | Kiruthikpranav |
| CB.SC.U4AIE24040 | Pulipati Chaithanya |
| CB.SC.U4AIE24060 | Vikramendraa Shanmugavelu Thiyagarajan |
| CB.SC.U4AIE24141 | Neeraj T |
| CB.SC.U4AIE24153 | Thammini Varun |

> **Based on:** *A. R. Babaei and M. Karimi, "Optimal Trajectory-Planning of UAVs via B-Splines and Disjunctive Programming," 2018* · [arxiv.org/pdf/1807.02931](https://arxiv.org/pdf/1807.02931)  
> **Controller:** *T. Lee, M. Leok, N. H. McClamroch, "Geometric Tracking Control of a Quadrotor UAV on SE(3)," CDC 2010*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Mathematical Foundations](#3-mathematical-foundations)
   - [3.1 B-Spline Trajectory](#31-b-spline-trajectory)
   - [3.2 MILP Obstacle Avoidance](#32-milp-obstacle-avoidance)
   - [3.3 Receding Horizon Control](#33-receding-horizon-control)
   - [3.4 6-DOF Quadrotor Dynamics](#34-6-dof-quadrotor-dynamics)
   - [3.5 Geometric Controller on SE(3)](#35-geometric-controller-on-se3)
   - [3.6 Dryden Wind Turbulence Model](#36-dryden-wind-turbulence-model)
4. [What We Implemented from the Paper](#4-what-we-implemented-from-the-paper)
5. [What We Added Beyond the Paper](#5-what-we-added-beyond-the-paper)
6. [Scenario: Urban Canyon](#6-scenario-urban-canyon)
7. [File Structure](#7-file-structure)
8. [Setup and Running](#8-setup-and-running)
9. [Output Files](#9-output-files)
10. [Key Parameters](#10-key-parameters)
11. [References](#11-references)

---

## 1. Project Overview

This project simulates an autonomous UAV navigating ~6 km through a **mixed-height urban canyon** with tall skyscrapers, low wide hurdles, and restricted No-Fly Zones — all while fighting persistent wind turbulence.

The drone handles **three simultaneous challenges:**

| Challenge | How We Handle It |
|---|---|
| 3D buildings in the path | MILP disjunctive constraints (Big-M method) |
| Restricted airspace (NFZs) | 2D disjunctive constraints per zone |
| Persistent wind disturbance | Dryden Markov model + closed-loop RHC correction |

**The core control loop:**

```
┌──────────────────────────────────────────────────────────┐
│                RECEDING HORIZON CONTROL LOOP              │
│                                                          │
│   ┌──────────────┐   ┌─────────────┐   ┌─────────────┐  │
│   │ MILP Planner │──▶│  B-Spline   │──▶│  6-DOF Sim  │  │
│   │  (10s ahead) │   │  Reference  │   │  + Wind     │  │
│   └──────────────┘   └─────────────┘   └──────┬──────┘  │
│           ▲                                    │         │
│           └──────── ACTUAL STATE (with wind) ──┘         │
└──────────────────────────────────────────────────────────┘
```

**Key principle:** The planner always starts from the **real, wind-disturbed position** of the drone — never from a predicted one. This is what makes it truly closed-loop.

---

## 2. System Architecture

### Full System Diagram

![UAV RHC System Architecture — shows all four layers: mission definition (scenario.py), RHC orchestrator (core.py), MILP planner with B-spline matrices (planner_obstacles.py / bspline_utils.py) running in parallel with the 6-DOF physics simulator and geometric controller (quadrotor.py / controller.py), and all output files generated after the run.](uav_rhc_system_architecture.png)

> **Reading the diagram:** Data flows top-to-bottom across four layers. The RHC orchestrator (Layer 1) drives both the MILP planner (Layer 2A, runs every 2 s) and the physics simulator (Layer 2B, runs at 500 Hz). The planner produces B-spline control points Q\*, which are sampled to give reference position/velocity/acceleration to the geometric controller every 2 ms. The actual wind-disturbed drone state feeds back up to Layer 1 to start the next horizon — this feedback loop is what makes the system truly closed-loop.

### Timing Diagram

![UAV RHC Timing Diagram — shows how each 10-second planning horizon overlaps with a 2-second execution window repeating ~44 times over the 88-second flight. Each execution window runs 1000 RK4 physics steps. The actual wind-disturbed drone position at the end of each execute window becomes the starting state for the next MILP solve.](uav_rhc_timing_diagram.png)

> **Reading the diagram:** Each PLAN covers a 10 s lookahead window but only the first 2 s (EXEC) is actually flown before replanning. The drone position `pos₀ → pos₁ → ...` shown at the bottom is the real, wind-disturbed position — not the planned one. This is what gets fed back into the next MILP solve.

### Layer Summary

| Layer | Files | Runs at | Role |
|---|---|---|---|
| 0 — Mission | `scenario.py` | Once at start | Define map, start, goal |
| 1 — RHC Loop | `core.py` | Every 2 s | Orchestrate plan → execute → observe |
| 2A — Planner | `planner_obstacles.py`, `bspline_utils.py` | Every 2 s | Solve MILP, return B-spline Q |
| 2B — Physics | `quadrotor.py`, `controller.py` | Every 0.002 s (500 Hz) | Simulate real drone + wind |
| 3 — Outputs | `simulation.py`, `visualizer.py` | After run | Save figures, HTML, cache, GUI |

---

## 3. Mathematical Foundations

### 3.1 B-Spline Trajectory

**File:** `bspline_utils.py`

The planned trajectory is a **degree-4 B-spline** (quartic), giving $C^3$ continuity — position, velocity, acceleration, and jerk are all smooth. This is critical so the controller never receives discontinuous commands.

**Trajectory curve:**

$$\mathbf{p}(\tau) = \sum_{i=0}^{n_c - 1} N_{i,k}(\tau)\,\mathbf{q}_i, \qquad \tau \in [0,\,1]$$

where $\mathbf{q}_i \in \mathbb{R}^3$ are the **control points** (the MILP decision variables) and $N_{i,k}(\tau)$ are the basis functions from the Cox-de Boor recursion.

**Cox-de Boor Recursion:**

$$N_{i,1}(\tau) = \begin{cases} 1 & t_i \leq \tau < t_{i+1} \\ 0 & \text{otherwise} \end{cases}$$

$$N_{i,k}(\tau) = \frac{\tau - t_i}{t_{i+k-1} - t_i}\,N_{i,k-1}(\tau) \;+\; \frac{t_{i+k} - \tau}{t_{i+k} - t_{i+1}}\,N_{i+1,k-1}(\tau)$$

> Divisions by zero are treated as 0 (standard convention).

**Matrix form** used inside the MILP (pre-computed at $h+1 = 11$ sample points):

$$\mathbf{X} = \mathbf{B}\,Q \qquad \mathbf{V} = \frac{1}{T_h}\mathbf{B}'\,Q \qquad \mathbf{A} = \frac{1}{T_h^2}\mathbf{B}''\,Q$$

where $Q \in \mathbb{R}^{14 \times 3}$ is the full control-point matrix, and $\mathbf{B},\,\mathbf{B}',\,\mathbf{B}''$ are the pre-computed basis, velocity-basis, and acceleration-basis matrices.  
This matrix form is what allows velocity and acceleration to appear as **linear constraints** in the MILP.

---

### 3.2 MILP Obstacle Avoidance

**File:** `planner_obstacles.py`

This is the **core contribution of the reference paper.** The problem: the drone must be *outside* at least one face of each building — this is an OR condition (non-convex). It is linearised using **Big-M disjunctive programming.**

**For a 3D cuboid** $[x_l, x_u] \times [y_l, y_u] \times [z_l, z_u]$ with safety margin $\eta$, at each trajectory sample point $i$:

Introduce 6 binary variables $b_0, \ldots, b_5 \in \{0,1\}$ with:

$$\sum_{j=0}^{5} b_j = 1 \quad \text{(exactly one escape direction is active)}$$

Then enforce the six face constraints:

$$X_i^x \leq (x_l - \eta) + (1-b_0)\,M \qquad \text{[left of building]}$$
$$X_i^x \geq (x_u + \eta) - (1-b_1)\,M \qquad \text{[right of building]}$$
$$X_i^y \leq (y_l - \eta) + (1-b_2)\,M \qquad \text{[in front]}$$
$$X_i^y \geq (y_u + \eta) - (1-b_3)\,M \qquad \text{[behind]}$$
$$X_i^z \leq (z_l - \eta) + (1-b_4)\,M \qquad \text{[below]}$$
$$X_i^z \geq (z_u + \eta) - (1-b_5)\,M \qquad \text{[above]}$$

**How it works:** When $b_j = 1$, that constraint is tight — the drone must be on that safe side. When $b_j = 0$, the term $(1-0) \cdot M = 10000$ makes the constraint trivially satisfied (relaxed). Only one $b_j = 1$ is enforced, so the drone is guaranteed to avoid **at least one face** per building per step.

**2D No-Fly Zones** (infinite height, like restricted airspace) use the same logic with only 4 binary variables — no Z-face constraints.

---

### 3.3 Receding Horizon Control

**File:** `core.py`

Solving the MILP over the entire 88-second flight at once is computationally intractable. RHC breaks it into short, manageable sub-problems.

**Each horizon repeats:**

| Step | Action |
|---|---|
| 1. Plan | Solve MILP for a $T_h = 10\,\text{s}$ horizon from the **actual** drone state |
| 2. Execute | Fly the first $\Delta t = 2\,\text{s}$ of the plan (1000 physics steps at 500 Hz) |
| 3. Observe | Read the new **actual** state — which differs from the plan due to wind |
| 4. Repeat | Go back to step 1 |

**The optimisation problem at each horizon:**

$$\min_{Q} \;\; w_1 \sum_{i=0}^{h-1} \|\mathbf{X}_{i+1} - \mathbf{X}_i\|_1 \;+\; w_2 \|\mathbf{X}_h - \mathbf{x}_f\|_1 \;+\; w_3 \sum_{i=1}^{h} X_i^z \;+\; w_4 \sum_{i=0}^{h-1} \|\Delta\mathbf{J}_i\|_1$$

**Subject to:**

| Constraint | Value | Purpose |
|---|---|---|
| $\mathbf{X}_0 = \mathbf{p}_\text{actual}$ | — | Start from real position |
| $\mathbf{V}_0 = \dot{\mathbf{p}}_\text{actual}$ | — | No velocity jump |
| $\mathbf{A}_0 = \ddot{\mathbf{p}}_\text{plan}$ | — | C² continuity at horizon boundary |
| $-v_{max} \leq V_i^{x,y,z} \leq v_{max}$ | 60 m/s | Speed limit (per axis) |
| $-a_{max} \leq A_i^{x,y,z} \leq a_{max}$ | 14.715 m/s² ≈ 1.5g | Thrust cap (per axis) |
| $1 \leq X_i^z \leq z_{max}$ | 1–25 m | Altitude bounds |
| Big-M obstacle constraints | — | As described in §3.2 |

> **Altitude ceiling insight:** Setting $z_{max} = 25\,\text{m}$ below the tall-building height (200 m) forces the planner to navigate around them laterally. Short hurdles at 15 m are below the ceiling, so the planner flies over them — which is cheaper.

**Open-loop vs. closed-loop:**

| | Open-Loop | Closed-Loop (this project) |
|---|---|---|
| Planner input | Predicted state from previous plan | **Actual** wind-disturbed state |
| Wind error | Accumulates unbounded | Corrected every 2 seconds |
| Infeasibility | Crashes | Graceful glide + retry |

---

### 3.4 6-DOF Quadrotor Dynamics

**File:** `quadrotor.py`

The drone is modelled as a rigid body with a **12-dimensional state vector:**

$$\mathbf{s} = \begin{bmatrix} \mathbf{p} \\ \dot{\mathbf{p}} \\ \boldsymbol{\Phi} \\ \boldsymbol{\omega} \end{bmatrix} = \begin{bmatrix} x,\, y,\, z \\ v_x,\, v_y,\, v_z \\ \phi,\, \theta,\, \psi \\ p,\, q,\, r \end{bmatrix}$$

where $\boldsymbol{\Phi} = (\phi,\theta,\psi)$ are ZYX Euler angles (roll, pitch, yaw) and $\boldsymbol{\omega} = (p,q,r)$ are body angular rates.

**Translational dynamics** (Newton's 2nd law + wind):

$$m\ddot{\mathbf{p}} = \mathbf{R}\begin{bmatrix}0\\0\\T\end{bmatrix} - mg\,\mathbf{e}_3 + m\mathbf{w}$$

where $T$ is total thrust, $\mathbf{R} \in SO(3)$ is the rotation matrix, and $\mathbf{w}$ is the Dryden wind acceleration.

**Rotation matrix** (ZYX convention — yaw first, then pitch, then roll):

$$\mathbf{R} = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$$

**Rotational dynamics** (Euler's equation):

$$\mathbf{I}\dot{\boldsymbol{\omega}} = \boldsymbol{\tau} - \boldsymbol{\omega} \times (\mathbf{I}\boldsymbol{\omega})$$

where $\mathbf{I} = \text{diag}(0.0196,\; 0.0196,\; 0.0264)\;\text{kg m}^2$ and $\boldsymbol{\tau}$ is the control torque vector.

**Numerical integration** — 4th-order Runge-Kutta at $\delta t = 0.002\,\text{s}$ (500 Hz):

$$\mathbf{s}(t+\delta t) = \mathbf{s}(t) + \frac{\delta t}{6}\left(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4\right)$$

RK4 is used (rather than Euler) because at speeds up to 60 m/s and high angular rates, Euler integration accumulates significant errors unless the timestep is impractically small.

---

### 3.5 Geometric Controller on SE(3)

**File:** `controller.py`  
**Reference:** Lee, Leok & McClamroch, CDC 2010

Unlike PID controllers that use Euler angles and suffer from **gimbal lock** (a singularity at pitch = ±90°), this controller operates directly on SO(3) — the rotation manifold — with no singularities.

**The control law is computed in five steps:**

**① Desired force** (position + velocity feedback + feed-forward):

$$\mathbf{F}_{des} = -K_p\,\mathbf{e}_p - K_v\,\mathbf{e}_v + mg\,\mathbf{e}_3 + m\,\ddot{\mathbf{p}}_d$$

where $\mathbf{e}_p = \mathbf{p} - \mathbf{p}_d$ (position error) and $\mathbf{e}_v = \dot{\mathbf{p}} - \dot{\mathbf{p}}_d$ (velocity error).  
Gains: $K_p = \text{diag}(16,16,20)$, $K_v = \text{diag}(8,8,9)$ → critically damped at $\omega_n \approx 4\,\text{rad/s}$.

**② Total thrust** (projection of desired force onto current body-z):

$$T = \mathbf{F}_{des} \cdot \mathbf{b}_3, \qquad \mathbf{b}_3 = \mathbf{R}\,\mathbf{e}_3$$

**③ Desired rotation matrix** from the force direction and desired heading:

$$\mathbf{b}_{3,d} = \frac{\mathbf{F}_{des}}{\|\mathbf{F}_{des}\|}, \quad \mathbf{b}_{2,d} = \frac{\mathbf{b}_{3,d} \times \mathbf{b}_{1,c}}{\|\mathbf{b}_{3,d} \times \mathbf{b}_{1,c}\|}, \quad \mathbf{b}_{1,d} = \mathbf{b}_{2,d} \times \mathbf{b}_{3,d}$$

$$\mathbf{R}_{des} = \begin{bmatrix} \mathbf{b}_{1,d} & \mathbf{b}_{2,d} & \mathbf{b}_{3,d} \end{bmatrix}$$

**④ Attitude error on SO(3)** via the vee map:

$$\mathbf{e}_R = \frac{1}{2}\,\text{vee}\!\left(\mathbf{R}_{des}^\top\mathbf{R} - \mathbf{R}^\top\mathbf{R}_{des}\right)$$

The **vee map** extracts a 3-vector from a skew-symmetric matrix: $\text{vee}(S) = [S_{32},\, S_{13},\, S_{21}]^\top$.

**⑤ Control torques:**

$$\boldsymbol{\tau} = -K_R\,\mathbf{e}_R - K_\omega\,\mathbf{e}_\omega + \boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega}$$

Gains: $K_R = \text{diag}(120,120,60)$, $K_\omega = \text{diag}(16,16,8)$ → $\omega_n \approx 30\,\text{rad/s}$ (attitude loop is ~7.5× faster than position loop). The $\boldsymbol{\omega} \times \mathbf{I}\boldsymbol{\omega}$ term cancels gyroscopic effects.

---

### 3.6 Dryden Wind Turbulence Model

**File:** `quadrotor.py → WindModel`

Wind is modelled as a **first-order Markov (auto-regressive) process**, inspired by MIL-HDBK-1797:

$$\mathbf{w}[k] = \rho\,\mathbf{w}[k-1] + \boldsymbol{\sigma} \odot \sqrt{1-\rho^2}\;\mathbf{n}[k], \qquad \mathbf{n}[k] \sim \mathcal{N}(\mathbf{0},\mathbf{I})$$

| Parameter | Value | Effect |
|---|---|---|
| $\rho$ | 0.995 | High autocorrelation → slow, persistent gusts |
| $\boldsymbol{\sigma}$ | $[1.8,\; 1.8,\; 0.4]\;\text{m/s}^2$ | Horizontal turbulence >> vertical |
| Seed | 42 | Fully reproducible runs |

The wind acceleration $\mathbf{w}$ is injected directly into the translational dynamics at every 2 ms physics step. Because the RHC planner uses the **actual** wind-pushed position at each replanning step, accumulated drift is corrected every 2 seconds automatically.

---

## 4. What We Implemented from the Paper

| Component | Paper Section | File |
|---|---|---|
| B-spline trajectory representation | §II | `bspline_utils.py` |
| MILP problem formulation (Big-M) | §III-A | `planner_obstacles.py` |
| 2D No-Fly Zone constraints | §III-B | `planner_obstacles.py` |
| Velocity & acceleration bounds | §III-C | `planner_obstacles.py` |
| Altitude ceiling constraint | §III | `planner_obstacles.py` |
| Multi-objective optimisation (J₁, J₂, J₃) | §IV | `planner_obstacles.py` |
| Receding Horizon Control structure | §V | `core.py` |

---

## 5. What We Added Beyond the Paper

| Addition | Description |
|---|---|
| **True Closed-Loop RHC** | The paper uses open-loop pre-computation. We re-plan from the **actual** wind-disturbed 6-DOF state at every horizon. |
| **6-DOF Quadrotor Physics** | Full rigid-body simulation with RK4 at 500 Hz. The paper only plans; we simulate whether a real drone could follow. |
| **Geometric Controller on SE(3)** | Lee et al. (2010) — operates on the rotation manifold SO(3), no gimbal-lock singularities. |
| **Dryden Wind Turbulence** | First-order Markov gust model; drone is persistently pushed off-course and must actively correct. |
| **Dynamic Obstacle Infrastructure** | `DynamicObstacle` class and RHC injection logic are fully implemented — currently disabled in the scenario (`create_dynamic_obstacle()` returns `None`). Ready to enable. |
| **Free Open-Source Solver** | Paper uses commercial Gurobi; we use SciPy MILP via CVXPY — no licence needed. |
| **Infeasibility Handling** | Graceful deceleration glide + retry if the MILP fails, rather than crashing. |
| **Interactive Outputs** | 6-panel publication figure, standalone Plotly 3D HTML viewer, live animated GUI. |

---

## 6. Scenario: Urban Canyon

```
  START  [2500, -1400, 12 m]
     │
     ├── TALL GATE  (200 m)  ─── Fly laterally through the gap   ─── z_max = 25 m enforces this
     │
     ├── LOW HURDLE (15 m)   ─── Fly over                        ─── cheaper than going around
     │
     ├── TALL GATE  (200 m)  ─── Fly laterally through the gap
     │
     ├── LOW HURDLE (15 m)   ─── Fly over
     │
     └── [ 4 No-Fly Zones block trivial bypass routes ]
     │
  GOAL   [-2200, 2000, 12 m]
```

| Stat | Value |
|---|---|
| Straight-line distance | ~5.4 km |
| Actual flight path | ~6 km |
| Flight duration | ~88 seconds |
| RHC horizons solved | ~44 |
| Tall building height | 200 m (above $z_{max}=25$ m → lateral avoidance) |
| Short hurdle height | 15 m (below $z_{max}=25$ m → overfly) |
| No-Fly Zones | 4 zones blocking bypass routes |

---

## 7. File Structure

| File | Key Functions / Classes | Role |
|---|---|---|
| `bspline_utils.py` | `cox_de_boor()`, `get_bspline_matrices()`, `sample_trajectory()` | Pure B-spline maths — no framework imports |
| `planner_obstacles.py` | `UAVPlannerMILP.plan()` | MILP formulation and solve |
| `controller.py` | `GeometricController.compute()` | SE(3) tracking law |
| `quadrotor.py` | `Quadrotor.step()`, `WindModel.step()` | 6-DOF RK4 physics + Dryden wind |
| `dynamic_obstacle.py` | `DynamicObstacle.as_nfz()` | Constant-velocity intruder → NFZ *(built, currently disabled)* |
| `scenario.py` | `build_scenario()` | Defines map: buildings, NFZs, start, goal |
| `core.py` | `run_closed_loop_rhc()` | Interleaved plan-simulate RHC loop |
| `simulation.py` | `main()`, `create_figure()`, `save_interactive_html()` | Entry point; generates all outputs |
| `visualizer.py` | `FuncAnimation` loop | Live 3D GUI with HUD and telemetry |
| `simulation_cache.npz` | — | Compressed log for instant GUI re-launch |
| `simulation_results.png` | — | 6-panel publication figure |
| `trajectory_3d.html` | — | Interactive Plotly 3D viewer (open in browser) |
| `flight.mp4` | — | Rendered animation export |

---

## 8. Setup and Running

### Requirements
- Python 3.10 or higher
- Windows 10/11, macOS, or Linux
- 4 GB RAM minimum

### Installation

```bash
# 1. Create virtual environment
python -m venv venv

# 2. Activate
#    Windows PowerShell:
.\venv\Scripts\Activate.ps1
#    Windows CMD:
venv\Scripts\activate.bat
#    macOS / Linux:
source venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows PowerShell script error?** Run this first:  
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process`

### Running

| Mode | Command | Output |
|---|---|---|
| **Full simulation** | `python simulation.py` | `simulation_results.png`, `trajectory_3d.html`, `simulation_cache.npz` |
| **Live 3D GUI** | `python visualizer.py` | Animated window (loads cache instantly) |
| **Force fresh run** | `python visualizer.py --fresh` | Re-runs MILP + physics, then opens GUI |
| **Export video** | `python visualizer.py --save` | `flight.mp4` (ffmpeg) or `flight.gif` (fallback) |

### GUI Controls

| Key / Action | Effect |
|---|---|
| `SPACE` | Pause / Resume animation |
| Left-click drag *(while paused)* | Rotate 3D camera |
| Scroll wheel | Zoom in / out |

### Troubleshooting

| Problem | Fix |
|---|---|
| GUI window doesn't appear (Linux) | `sudo apt-get install python3-tk` |
| CVXPY / solver errors | `pip install --upgrade cvxpy scipy` (use Python ≤ 3.12) |
| Plotly not found | `pip install plotly` |

---

## 9. Output Files

| File | Description |
|---|---|
| `simulation_results.png` | 6-panel dark-theme figure: 3D path, top-down map, altitude + wind, X/Y tracking, tracking error, thrust + energy |
| `trajectory_3d.html` | Standalone interactive Plotly 3D viewer — rotate, zoom, pan in any browser |
| `simulation_cache.npz` | Compressed NumPy archive of all simulation logs — enables instant GUI re-launch |
| `flight.mp4` / `flight.gif` | Rendered flight animation from `python visualizer.py --save` |

---

## 10. Key Parameters

| Parameter | Symbol | Value | File | Effect |
|---|---|---|---|---|
| Spline degree | $k$ | 4 | `planner_obstacles.py` | $C^3$ continuity |
| Horizon steps | $h$ | 10 | `planner_obstacles.py` | 10 s lookahead, 14 control points |
| Execute window | $\Delta t$ | 2.0 s | `simulation.py` | Replanning frequency |
| Physics timestep | $\delta t$ | 0.002 s | `simulation.py` | 500 Hz RK4 |
| Max velocity | $v_{max}$ | 60 m/s | `scenario.py` | Per-axis speed limit |
| Max acceleration | $a_{max}$ | 14.715 m/s² | `planner_obstacles.py` | ≈ 1.5g thrust cap |
| Altitude ceiling | $z_{max}$ | 25 m | `planner_obstacles.py` | Forces lateral avoidance of tall buildings |
| Safety margin (tall building) | $\eta$ | 80 m | `scenario.py` | Clearance from building face |
| Safety margin (NFZ) | $\gamma$ | 30 m | `scenario.py` | Clearance from NFZ boundary |
| Big-M constant | $M$ | 10 000 | `planner_obstacles.py` | Disjunctive constraint relaxation |
| Wind autocorrelation | $\rho$ | 0.995 | `quadrotor.py` | Slow, persistent gusts |
| Wind std-dev | $\boldsymbol{\sigma}$ | [1.8, 1.8, 0.4] m/s² | `quadrotor.py` | Turbulence intensity |
| Position gain | $K_p$ | diag(16, 16, 20) | `controller.py` | $\omega_n \approx 4$ rad/s |
| Velocity gain | $K_v$ | diag(8, 8, 9) | `controller.py` | Critical damping |
| Attitude gain | $K_R$ | diag(120, 120, 60) | `controller.py` | $\omega_n \approx 30$ rad/s |
| Angular rate gain | $K_\omega$ | diag(16, 16, 8) | `controller.py` | Attitude damping |
| Goal radius | — | 50 m | `core.py` | Termination threshold |

---

## 11. References

1. **A. R. Babaei and M. Karimi** — *"Optimal Trajectory-Planning of UAVs via B-Splines and Disjunctive Programming,"* 2018. Core MILP + B-spline planning method. [[arXiv]](https://arxiv.org/pdf/1807.02931)

2. **T. Lee, M. Leok, N. H. McClamroch** — *"Geometric Tracking Control of a Quadrotor UAV on SE(3),"* 49th IEEE CDC, 2010, pp. 5420–5425. Geometric controller on SE(3).

3. **MIL-HDBK-1797** — *"Flying Qualities of Piloted Aircraft,"* US Department of Defense, 1997. Dryden wind turbulence model specification.

4. **L. Piegl and W. Tiller** — *The NURBS Book,* 2nd ed., Springer, 1997. B-spline theory and Cox-de Boor algorithm.

5. **S. Diamond and S. Boyd** — *"CVXPY: A Python-Embedded Modeling Language for Convex Optimization,"* JMLR, 2016. Optimisation framework used for MILP formulation.