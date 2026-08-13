"""
UAV B-Spline Trajectory Planning & Dynamic Simulation
======================================================
Based on: "Optimal Trajectory-Planning of UAVs via B-Splines
           and Disjunctive Programming" (Babaei & Karimi, 2018)

Standalone script -- saves simulation_results.png.
Run:  python simulation.py
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless: save PNG only, no window needed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

from planner_obstacles import UAVPlannerMILP
from scenario import build_scenario
from core import precompute_rhc_trajectory, simulate_dynamics


# ======================================================================= #
#  DRAWING HELPERS                                                          #
# ======================================================================= #

def draw_cuboid(ax, xl, xu, yl, yu, zl, zu, color='royalblue', alpha=0.35):
    verts = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]],
    ]
    pc = Poly3DCollection(verts, facecolors=color, linewidths=0.5,
                          edgecolors='k', alpha=alpha)
    ax.add_collection3d(pc)


# ======================================================================= #
#  PUBLICATION FIGURE                                                       #
# ======================================================================= #

def create_figure(ref_t, ref_pos, ref_vel, ref_acc,
                  log_t, log_pos, log_vel, log_eul, log_u,
                  obstacles, no_fly_zones, x_s, x_f):

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor('#0e1117')

    C_REF = '#00d4ff'
    C_ACT = '#ff6b6b'
    C_OBS = '#4169E1'
    C_NFZ = '#ff4444'
    C_TXT = '#e0e0e0'

    # ---- Panel 1: 3-D trajectory ----
    ax3d = fig.add_subplot(2, 2, 1, projection='3d')
    ax3d.set_facecolor('#1a1a2e')
    ax3d.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
              '-', color=C_REF, lw=1.5, alpha=0.6, label='Planned (B-spline)')
    ax3d.plot(log_pos[:,0], log_pos[:,1], log_pos[:,2],
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
    ax3d.tick_params(colors=C_TXT)

    # ---- Panel 2: X-Y top-down ----
    ax_xy = fig.add_subplot(2, 2, 2)
    ax_xy.set_facecolor('#1a1a2e')
    ax_xy.plot(ref_pos[:,0], ref_pos[:,1], '-', color=C_REF, lw=1.5, alpha=0.6, label='Planned')
    ax_xy.plot(log_pos[:,0], log_pos[:,1], '-', color=C_ACT, lw=1.0, alpha=0.9, label='Actual')
    ax_xy.plot(x_s[0], x_s[1], 'o', color='lime', ms=10, label='Start')
    ax_xy.plot(x_f[0], x_f[1], '*', color='gold', ms=14, label='Goal')
    for obs in obstacles:
        rect = plt.Rectangle((obs['xl'], obs['yl']),
                              obs['xu']-obs['xl'], obs['yu']-obs['yl'],
                              facecolor=C_OBS, alpha=0.5, edgecolor='white', lw=1)
        ax_xy.add_patch(rect)
    for nfz in no_fly_zones:
        rect = plt.Rectangle((nfz['xl'], nfz['yl']),
                              nfz['xu']-nfz['xl'], nfz['yu']-nfz['yl'],
                              facecolor=C_NFZ, alpha=0.2, edgecolor='white', lw=1, linestyle='--')
        ax_xy.add_patch(rect)
    ax_xy.set_xlabel('X (m)', color=C_TXT)
    ax_xy.set_ylabel('Y (m)', color=C_TXT)
    ax_xy.set_title('Top-Down View (X-Y)', color=C_TXT, fontsize=13, fontweight='bold')
    ax_xy.legend(fontsize=8)
    ax_xy.set_aspect('equal')
    ax_xy.grid(True, alpha=0.15, color='white')
    ax_xy.tick_params(colors=C_TXT)

    # ---- Panel 3: Position tracking ----
    ax_pos = fig.add_subplot(2, 2, 3)
    ax_pos.set_facecolor('#1a1a2e')
    labels     = ['X', 'Y', 'Z']
    c_ref_list = ['#00d4ff', '#00ff88', '#ffaa00']
    c_act_list = ['#ff6b6b', '#ff44cc', '#ff4444']
    for dim in range(3):
        ax_pos.plot(ref_t, ref_pos[:,dim], '--', color=c_ref_list[dim],
                    lw=1.0, alpha=0.6, label=f'{labels[dim]} ref')
        ax_pos.plot(log_t, log_pos[:,dim], '-', color=c_act_list[dim],
                    lw=0.8, alpha=0.9, label=f'{labels[dim]} actual')
    ax_pos.set_xlabel('Time (s)', color=C_TXT)
    ax_pos.set_ylabel('Position (m)', color=C_TXT)
    ax_pos.set_title('Position Tracking', color=C_TXT, fontsize=13, fontweight='bold')
    ax_pos.legend(fontsize=7, ncol=2)
    ax_pos.grid(True, alpha=0.15, color='white')
    ax_pos.tick_params(colors=C_TXT)

    # ---- Panel 4: Tracking error ----
    ax_err = fig.add_subplot(2, 2, 4)
    ax_err.set_facecolor('#1a1a2e')
    dt_ref = ref_t[1] - ref_t[0] if len(ref_t) > 1 else 0.01
    err = np.array([np.linalg.norm(log_pos[i] - ref_pos[min(int(log_t[i]/dt_ref), len(ref_pos)-1)])
                    for i in range(len(log_t))])
    ax_err.fill_between(log_t, 0, err, color='#ff6b6b', alpha=0.3)
    ax_err.plot(log_t, err, '-', color='#ff6b6b', lw=0.8, label='||e_p||')
    rms  = np.sqrt(np.mean(err**2))
    peak = np.max(err)
    ax_err.text(0.98, 0.95, f'RMS = {rms:.2f} m\nPeak = {peak:.2f} m',
                transform=ax_err.transAxes, fontsize=9, color=C_TXT,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='#2a2a4a', alpha=0.8))
    ax_err.set_xlabel('Time (s)', color=C_TXT)
    ax_err.set_ylabel('Tracking Error (m)', color=C_TXT)
    ax_err.set_title('Position Tracking Error', color=C_TXT, fontsize=13, fontweight='bold')
    ax_err.legend(fontsize=9)
    ax_err.grid(True, alpha=0.15, color='white')
    ax_err.tick_params(colors=C_TXT)

    plt.suptitle('UAV B-Spline Trajectory Planning with Geometric Tracking Control\n'
                 'Dense Urban Canyon -- 8 Buildings + 2 No-Fly Zones',
                 color='white', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def main():
    print("=" * 65)
    print("  UAV B-Spline Trajectory Planner + Dynamic Simulation")
    print("  Scenario: Dense Urban Canyon (8 buildings, 2 no-fly zones)")
    print("=" * 65)

    planner = UAVPlannerMILP(h=10, ts=1.0)
    x_s, v_s, a_s, x_f = build_scenario(planner)

    # Phase 1: RHC trajectory
    print("\n[Phase 1] Pre-computing Receding-Horizon trajectory ...")
    t0 = time.time()
    ref_t, ref_pos, ref_vel, ref_acc, segments = precompute_rhc_trajectory(
        planner, x_s, v_s, a_s, x_f, max_iter=100, execute_time=2.0, dt_ref=0.01
    )
    print(f"  Planning completed in {time.time()-t0:.1f}s  "
          f"({len(segments)} horizons, {ref_t[-1]:.1f}s flight)")

    # Phase 2: Dynamics simulation
    print("\n[Phase 2] Simulating 6-DOF quadrotor dynamics ...")
    t0 = time.time()
    log_t, log_pos, log_vel, log_eul, log_u = simulate_dynamics(
        ref_t, ref_pos, ref_vel, ref_acc, dt_sim=0.002
    )
    print(f"  Simulation completed in {time.time()-t0:.1f}s  "
          f"({len(log_t)} steps at 500 Hz)")

    # Phase 3: Publication figure
    print("\n[Phase 3] Generating publication figure ...")
    fig = create_figure(ref_t, ref_pos, ref_vel, ref_acc,
                        log_t, log_pos, log_vel, log_eul, log_u,
                        planner.obstacles, planner.no_fly_zones, x_s, x_f)
    fig.savefig('simulation_results.png', dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print("  Saved -> simulation_results.png")
    print("\nDone!")


if __name__ == "__main__":
    main()
