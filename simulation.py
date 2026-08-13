"""
UAV B-Spline Trajectory Planning & Dynamic Simulation
=====================================================
Based on: "Optimal Trajectory-Planning of UAVs via B-Splines
           and Disjunctive Programming" (Babaei & Karimi, 2018)

Pipeline
--------
1. Pre-compute full Receding-Horizon trajectory  (MILP planner)
2. Simulate 6-DOF quadrotor dynamics tracking it (Geometric controller)
3. Generate publication-quality plots             (matplotlib)
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')  # Windows console fix

import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive: save to file only
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

from planner_obstacles import UAVPlannerMILP
from controller import GeometricController
from quadrotor import Quadrotor
from bspline_utils import sample_trajectory

# ====================================================================== #
#  1.  PRE-COMPUTE FULL RHC TRAJECTORY                                    #
# ====================================================================== #

def precompute_rhc_trajectory(planner, x_s, v_s, a_s, x_f,
                               max_iter=80, execute_time=2.0, dt_ref=0.01):
    """
    Runs the MILP planner in a Receding-Horizon loop and stitches
    together the reference trajectory at `dt_ref` resolution.

    Returns
    -------
    t_ref       : (N,)     time stamps
    ref_pos     : (N, 3)   reference position
    ref_vel     : (N, 3)   reference velocity
    ref_acc     : (N, 3)   reference acceleration
    plan_segments : list of (Q, t_start, t_end) for each successful plan
    """
    ref_t   = [0.0]
    ref_pos = [x_s.copy()]
    ref_vel = [v_s.copy()]
    ref_acc = [a_s.copy()]
    plan_segments = []

    pos = x_s.copy()
    vel = v_s.copy()
    acc = a_s.copy()
    t_global = 0.0

    k = planner.k
    h = planner.h
    t_h = planner.t_h

    for iteration in range(max_iter):
        Q, X, V, A = planner.plan(pos, vel, acc, x_f)

        if Q is None:
            # Infeasible -- glide for execute_time with deceleration
            print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  INFEASIBLE  "
                  f"pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}]")
            n_hold = int(execute_time / dt_ref)
            decel = 0.8   # gentle braking factor
            for j in range(1, n_hold + 1):
                t_global += dt_ref
                vel *= decel
                pos_new = pos + vel * dt_ref
                pos_new[2] = max(pos_new[2], 1.0)  # terrain floor
                ref_t.append(t_global)
                ref_pos.append(pos_new.copy())
                ref_vel.append(vel.copy())
                ref_acc.append(np.zeros(3))
                pos = pos_new
            acc = np.zeros(3)
            continue

        plan_segments.append((Q.copy(), t_global, t_global + execute_time))

        # Sample the B-spline for `execute_time` seconds
        n_steps = int(execute_time / dt_ref)
        for j in range(1, n_steps + 1):
            t_local = j * dt_ref
            p, v, a = sample_trajectory(t_local, Q, k=k, h=h, t_h=t_h)
            t_global += dt_ref
            ref_t.append(t_global)
            ref_pos.append(p.copy())
            ref_vel.append(v.copy())
            ref_acc.append(a.copy())

        # State at end of executed portion -> initial state of next horizon
        pos, vel, acc = sample_trajectory(execute_time, Q, k=k, h=h, t_h=t_h)

        dist = np.linalg.norm(pos - x_f)
        print(f"  RHC iter {iteration:3d}  t={t_global:7.1f}s  OPTIMAL  "
              f"pos=[{pos[0]:.0f},{pos[1]:.0f},{pos[2]:.1f}]  "
              f"dist={dist:.0f}m")

        if dist < 50:
            print("  >>> Close enough -- stopping RHC.")
            break

    return (np.array(ref_t),
            np.array(ref_pos),
            np.array(ref_vel),
            np.array(ref_acc),
            plan_segments)


# ====================================================================== #
#  2.  SIMULATE QUADROTOR DYNAMICS                                         #
# ====================================================================== #

def simulate_dynamics(ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002):
    """
    Run the 6-DOF quadrotor model tracking the reference.

    Parameters
    ----------
    dt_sim : simulation time-step (500 Hz default -- stable for RK4)
    """
    quad = Quadrotor(mass=1.0)
    ctrl = GeometricController(mass=1.0)

    quad.set_state(pos=ref_pos[0], vel=ref_vel[0])

    T_total = ref_t[-1]
    n_sim   = int(T_total / dt_sim) + 1
    dt_ref  = ref_t[1] - ref_t[0]            # reference sample period

    # Pre-allocate logs
    log_t   = np.zeros(n_sim)
    log_pos = np.zeros((n_sim, 3))
    log_vel = np.zeros((n_sim, 3))
    log_eul = np.zeros((n_sim, 3))
    log_u   = np.zeros((n_sim, 4))

    log_pos[0] = quad.pos
    log_vel[0] = quad.vel

    for i in range(1, n_sim):
        t = i * dt_sim

        # Find reference index (nearest)
        ref_idx = min(int(t / dt_ref), len(ref_t) - 1)
        p_d = ref_pos[ref_idx]
        v_d = ref_vel[ref_idx]
        a_d = ref_acc[ref_idx]

        u = ctrl.compute(quad.state, p_d, v_d, a_d)
        quad.step(u, dt_sim)

        log_t[i]   = t
        log_pos[i] = quad.pos
        log_vel[i] = quad.vel
        log_eul[i] = quad.euler
        log_u[i]   = u

    return log_t, log_pos, log_vel, log_eul, log_u


# ====================================================================== #
#  3.  VISUALISATION                                                       #
# ====================================================================== #

def draw_cuboid(ax, xl, xu, yl, yu, zl, zu, color='royalblue', alpha=0.35, label=None):
    """Draw a semi-transparent cuboid on a 3-D axis."""
    verts = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],   # bottom
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],   # top
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],   # left
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],   # right
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],   # front
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]],   # back
    ]
    pc = Poly3DCollection(verts, facecolors=color, linewidths=0.5,
                          edgecolors='k', alpha=alpha)
    ax.add_collection3d(pc)


def create_figure(ref_t, ref_pos, ref_vel, ref_acc,
                  log_t, log_pos, log_vel, log_eul, log_u,
                  obstacles, no_fly_zones, x_s, x_f):
    """Generate a 4-panel publication figure."""

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0e1117')

    # colour palette
    C_REF  = '#00d4ff'   # cyan  -- reference
    C_ACT  = '#ff6b6b'   # coral -- actual
    C_OBS  = '#4169E1'   # blue  -- obstacle
    C_NFZ  = '#ff4444'   # red   -- no-fly zone
    C_TXT  = '#e0e0e0'

    # ---- Panel 1:  3-D trajectory ---- #
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.set_facecolor('#1a1a2e')
    ax3d.plot(ref_pos[:, 0], ref_pos[:, 1], ref_pos[:, 2],
              '-', color=C_REF, lw=1.5, alpha=0.6, label='Planned (B-spline)')
    ax3d.plot(log_pos[:, 0], log_pos[:, 1], log_pos[:, 2],
              '-', color=C_ACT, lw=1.2, alpha=0.9, label='Actual (6-DOF)')
    ax3d.scatter(*x_s, color='lime', s=80, zorder=5, label='Start')
    ax3d.scatter(*x_f, color='gold', s=80, zorder=5, marker='*', label='Goal')

    for obs in obstacles:
        draw_cuboid(ax3d, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                    obs['zl'], obs['zu'], color=C_OBS, alpha=0.4)
    for nfz in no_fly_zones:
        draw_cuboid(ax3d, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                    0, 60, color=C_NFZ, alpha=0.15)

    ax3d.set_xlabel('X (m)', color=C_TXT)
    ax3d.set_ylabel('Y (m)', color=C_TXT)
    ax3d.set_zlabel('Z (m)', color=C_TXT)
    ax3d.set_title('3-D Trajectory', color=C_TXT, fontsize=13, fontweight='bold')
    ax3d.legend(fontsize=8, loc='upper left')
    ax3d.view_init(elev=25, azim=-60)

    # ---- Panel 2:  X-Y projection ---- #
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.set_facecolor('#1a1a2e')
    ax_xy.plot(ref_pos[:, 0], ref_pos[:, 1], '-', color=C_REF, lw=1.5, alpha=0.6, label='Planned')
    ax_xy.plot(log_pos[:, 0], log_pos[:, 1], '-', color=C_ACT, lw=1.0, alpha=0.9, label='Actual')
    ax_xy.plot(x_s[0], x_s[1], 'o', color='lime', ms=10, label='Start')
    ax_xy.plot(x_f[0], x_f[1], '*', color='gold', ms=14, label='Goal')

    for obs in obstacles:
        rect = plt.Rectangle((obs['xl'], obs['yl']),
                              obs['xu'] - obs['xl'], obs['yu'] - obs['yl'],
                              facecolor=C_OBS, alpha=0.5, edgecolor='white', lw=1)
        ax_xy.add_patch(rect)
    for nfz in no_fly_zones:
        rect = plt.Rectangle((nfz['xl'], nfz['yl']),
                              nfz['xu'] - nfz['xl'], nfz['yu'] - nfz['yl'],
                              facecolor=C_NFZ, alpha=0.2, edgecolor='white', lw=1,
                              linestyle='--')
        ax_xy.add_patch(rect)

    ax_xy.set_xlabel('X (m)', color=C_TXT)
    ax_xy.set_ylabel('Y (m)', color=C_TXT)
    ax_xy.set_title('Top-Down View (X-Y)', color=C_TXT, fontsize=13, fontweight='bold')
    ax_xy.legend(fontsize=8)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.15, color='white')
    ax_xy.tick_params(colors=C_TXT)

    # ---- Panel 3:  position tracking ---- #
    ax_pos = fig.add_subplot(2, 2, 3)
    ax_pos.set_facecolor('#1a1a2e')
    labels = ['X', 'Y', 'Z']
    colors_ref = ['#00d4ff', '#00ff88', '#ffaa00']
    colors_act = ['#ff6b6b', '#ff44cc', '#ff4444']
    for dim in range(3):
        ax_pos.plot(ref_t, ref_pos[:, dim], '--', color=colors_ref[dim],
                    lw=1.0, alpha=0.6, label=f'{labels[dim]} ref')
        ax_pos.plot(log_t, log_pos[:, dim], '-', color=colors_act[dim],
                    lw=0.8, alpha=0.9, label=f'{labels[dim]} actual')
    ax_pos.set_xlabel('Time (s)', color=C_TXT)
    ax_pos.set_ylabel('Position (m)', color=C_TXT)
    ax_pos.set_title('Position Tracking', color=C_TXT, fontsize=13, fontweight='bold')
    ax_pos.legend(fontsize=7, ncol=2)
    ax_pos.grid(True, alpha=0.15, color='white')
    ax_pos.tick_params(colors=C_TXT)

    # ---- Panel 4:  tracking error ---- #
    ax_err = fig.add_subplot(2, 2, 4)
    ax_err.set_facecolor('#1a1a2e')

    # Compute tracking error (interpolate ref to sim timeline)
    dt_ref = ref_t[1] - ref_t[0] if len(ref_t) > 1 else 0.01
    err = np.zeros(len(log_t))
    for i, t in enumerate(log_t):
        idx = min(int(t / dt_ref), len(ref_pos) - 1)
        err[i] = np.linalg.norm(log_pos[i] - ref_pos[idx])

    ax_err.fill_between(log_t, 0, err, color='#ff6b6b', alpha=0.3)
    ax_err.plot(log_t, err, '-', color='#ff6b6b', lw=0.8, label='||e_p||')
    ax_err.set_xlabel('Time (s)', color=C_TXT)
    ax_err.set_ylabel('Tracking Error (m)', color=C_TXT)
    ax_err.set_title('Position Tracking Error', color=C_TXT, fontsize=13, fontweight='bold')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.15, color='white')
    ax_err.tick_params(colors=C_TXT)

    # stats annotation
    rms = np.sqrt(np.mean(err**2))
    peak = np.max(err)
    ax_err.text(0.98, 0.95, f'RMS = {rms:.2f} m\nPeak = {peak:.2f} m',
                transform=ax_err.transAxes, fontsize=9, color=C_TXT,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='#2a2a4a', alpha=0.8))

    plt.suptitle('UAV B-Spline Trajectory Planning with Geometric Tracking Control',
                 color='white', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ====================================================================== #
#  MAIN                                                                    #
# ====================================================================== #

def main():
    print("=" * 65)
    print("  UAV B-Spline Trajectory Planner + Dynamic Simulation")
    print("=" * 65)

    # ---- Scenario (Paper Scenario 2) ----
    x_s = np.array([3000.0, -2000.0, 10.0])
    v_s = np.array([-30.0,   30.0,    0.0])
    a_s = np.array([0.0,     0.0,     0.0])
    x_f = np.array([-2000.0, 3000.0,  1.0])

    planner = UAVPlannerMILP(h=10, ts=1.0)
    planner.add_cuboid_obstacle(1600, 2400, -1400, -600, 0, 50, eta=5.0)
    planner.add_no_fly_zone(-1500, 2400, -600, 2000, gamma=20.0)

    # ---- Phase 1: Pre-compute RHC trajectory ----
    print("\n[Phase 1] Pre-computing Receding-Horizon trajectory ...")
    t0 = time.time()
    ref_t, ref_pos, ref_vel, ref_acc, segments = precompute_rhc_trajectory(
        planner, x_s, v_s, a_s, x_f,
        max_iter=80, execute_time=2.0, dt_ref=0.01
    )
    plan_time = time.time() - t0
    print(f"  Planning completed in {plan_time:.1f}s  "
          f"({len(segments)} successful horizons, "
          f"{ref_t[-1]:.1f}s flight time)")

    # ---- Phase 2: Simulate quadrotor dynamics ----
    print("\n[Phase 2] Simulating 6-DOF quadrotor dynamics ...")
    t0 = time.time()
    log_t, log_pos, log_vel, log_eul, log_u = simulate_dynamics(
        ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002
    )
    sim_time = time.time() - t0
    print(f"  Simulation completed in {sim_time:.1f}s  "
          f"({len(log_t)} steps at 500 Hz)")

    # ---- Phase 3: Visualisation ----
    print("\n[Phase 3] Generating publication figure ...")
    fig = create_figure(
        ref_t, ref_pos, ref_vel, ref_acc,
        log_t, log_pos, log_vel, log_eul, log_u,
        planner.obstacles, planner.no_fly_zones,
        x_s, x_f
    )
    fig.savefig('simulation_results.png', dpi=150, facecolor=fig.get_facecolor())
    print("  Saved -> simulation_results.png")
    plt.close(fig)
    print("\nDone! Open simulation_results.png to view the results.")


if __name__ == "__main__":
    main()
