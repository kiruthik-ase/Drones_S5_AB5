"""
Live GUI Visualizer  --  Clean Edition
=======================================
Features:
  - Clean dark professional theme
  - Drone shown as a solid glowing dot (no arm lines)
  - Short buildings (flyover) in teal, Tall buildings (lateral) in steel blue
  - Live telemetry: speed | altitude | tracking error
  - SPACE to pause and freely rotate/zoom the 3D view
  - Optional GIF/MP4 export

Run:
    python visualizer.py               # interactive window
    python visualizer.py --save        # also save flight.gif/mp4
    python visualizer.py --fresh       # force recompute (ignore cache)
"""

import sys
import os
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time

from planner_obstacles import UAVPlannerMILP
from quadrotor import Quadrotor
from scenario import build_scenario, create_dynamic_obstacle, Z_CEILING
from core import run_closed_loop_rhc


# ======================================================================= #
#  THEME                                                                    #
# ======================================================================= #

C_BG      = '#0d1117'   # main background
C_PANEL   = '#161b22'   # panel background
C_BORDER  = '#30363d'   # panel borders
C_TXT     = '#c9d1d9'   # primary text
C_TXT2    = '#8b949e'   # secondary text

C_REF     = '#58a6ff'   # reference trajectory (blue)
C_ACT     = '#ff7b72'   # actual trajectory (coral)
C_TALL    = '#1f3a5f'   # tall building face (dark steel blue)
C_TALL_E  = '#388bfd'   # tall building edge
C_LOW     = '#0d3d2e'   # short/overfly-able building face (dark teal)
C_LOW_E   = '#2ea875'   # short building edge
C_NFZ     = '#3d1515'   # no-fly zone face (dark red)
C_NFZ_E   = '#da3633'   # no-fly zone edge
C_DRN     = '#ffa657'   # drone body (warm amber)
C_MOTEUR  = '#ffffff'   # motor dots
C_GRID    = '#21262d'   # grid lines
C_ALT     = '#d2a8ff'   # altitude line (purple)
C_SPD     = '#79c0ff'   # speed line
C_ERR     = '#ff7b72'   # error line
C_ROLL    = '#f0c040'   # roll angle (gold)
C_PITCH   = '#56d364'   # pitch angle (lime green)


# ======================================================================= #
#  HELPERS                                                                  #
# ======================================================================= #

def draw_cuboid_3d(ax, xl, xu, yl, yu, zl, zu,
                   face_color, edge_color, face_alpha=0.55, edge_alpha=0.8):
    verts = [
        [[xl,yl,zl],[xu,yl,zl],[xu,yu,zl],[xl,yu,zl]],
        [[xl,yl,zu],[xu,yl,zu],[xu,yu,zu],[xl,yu,zu]],
        [[xl,yl,zl],[xl,yu,zl],[xl,yu,zu],[xl,yl,zu]],
        [[xu,yl,zl],[xu,yu,zl],[xu,yu,zu],[xu,yl,zu]],
        [[xl,yl,zl],[xu,yl,zl],[xu,yl,zu],[xl,yl,zu]],
        [[xl,yu,zl],[xu,yu,zl],[xu,yu,zu],[xl,yu,zu]],
    ]
    pc = Poly3DCollection(verts,
                          facecolors=face_color,
                          linewidths=0.5,
                          edgecolors=edge_color,
                          alpha=face_alpha)
    ax.add_collection3d(pc)


def draw_rect_2d(ax, xl, xu, yl, yu, face_color, edge_color, alpha=0.6, ls='-'):
    ax.add_patch(Rectangle((xl, yl), xu-xl, yu-yl,
                            facecolor=face_color, alpha=alpha,
                            edgecolor=edge_color, lw=0.8, linestyle=ls))


def style_2d_ax(ax):
    ax.set_facecolor(C_PANEL)
    ax.tick_params(colors=C_TXT2, labelsize=7)
    for sp in ax.spines.values():
        sp.set_edgecolor(C_BORDER)
    ax.grid(True, alpha=0.25, color=C_GRID, lw=0.5)


# ======================================================================= #
#  MAIN                                                                     #
# ======================================================================= #

def main():
    save_mp4   = '--save'  in sys.argv
    use_fresh  = '--fresh' in sys.argv
    CACHE_FILE = 'simulation_cache.npz'

    print("=" * 68)
    print("  UAV Live Visualizer  |  Mixed-Height Urban Canyon")
    print("  Tall buildings: lateral nav  |  Short buildings: fly-over")
    print("=" * 68)

    # Always fast -- just registers obstacles/NFZs in the planner lists
    planner = UAVPlannerMILP(h=10, ts=1.0, z_max=Z_CEILING)
    x_s, v_s, a_s, x_f = build_scenario(planner)
    dyn_obs = create_dynamic_obstacle()

    # ------------------------------------------------------------------ #
    #  Load from cache OR run full simulation                              #
    # ------------------------------------------------------------------ #
    if not use_fresh and os.path.exists(CACHE_FILE):
        print(f"\n[Phase 1+2] Loading cached results from {CACHE_FILE} ...")
        t0   = time.time()
        data = np.load(CACHE_FILE)
        ref_t    = data['ref_t'];    ref_pos  = data['ref_pos']
        log_t    = data['log_t'];    log_pos  = data['log_pos']
        log_vel  = data['log_vel'];  log_eul  = data['log_eul']
        log_u    = data['log_u'];    log_wind = data['log_wind']
        print(f"  Loaded {len(log_t):,} steps ({log_t[-1]:.1f}s flight) "
              f"in {time.time()-t0:.2f}s  \u2713")
        print("  Tip: run 'python visualizer.py --fresh' to recompute")
    else:
        print(f"\n[Phase 1+2] Running closed-loop RHC ...")
        t0 = time.time()
        (ref_t, ref_pos,
         log_t, log_pos, log_vel, log_eul, log_u, log_wind,
         plan_log) = run_closed_loop_rhc(
            planner, x_s, v_s, a_s, x_f,
            max_iter=120, execute_time=2.0, dt_sim=0.002,
            dyn_obs=dyn_obs, enable_wind=True
        )
        print(f"  Done in {time.time()-t0:.1f}s  -- {log_t[-1]:.1f}s flight")
        np.savez_compressed(CACHE_FILE,
                            ref_t=ref_t, ref_pos=ref_pos,
                            log_t=log_t, log_pos=log_pos,
                            log_vel=log_vel, log_eul=log_eul,
                            log_u=log_u, log_wind=log_wind)
        print(f"  Cache saved -> {CACHE_FILE}")

    # ------------------------------------------------------------------ #
    #  Downsample for animation                                            #
    # ------------------------------------------------------------------ #
    ANIM_FPS   = 30
    SPEED_MULT = 5.0
    dt_anim    = SPEED_MULT / ANIM_FPS
    dt_sim_val = log_t[1] - log_t[0] if len(log_t) > 1 else 0.002
    stride     = max(1, int(dt_anim / dt_sim_val))

    a_t    = log_t[::stride]
    a_pos  = log_pos[::stride]
    a_vel  = log_vel[::stride]
    a_eul  = log_eul[::stride]
    a_wind = log_wind[::stride]
    N      = len(a_t)
    dt_ref_s = ref_t[1] - ref_t[0] if len(ref_t) > 1 else 0.002

    paused = [False]

    # ------------------------------------------------------------------ #
    #  Build figure                                                        #
    # ------------------------------------------------------------------ #
    plt.rcParams.update({
        'font.family': 'monospace',
        'axes.labelcolor': C_TXT,
        'xtick.color': C_TXT2,
        'ytick.color': C_TXT2,
    })

    fig = plt.figure(figsize=(21, 10), facecolor=C_BG)
    try:
        fig.canvas.manager.set_window_title(
            'UAV Mixed-Height Urban Canyon  |  Closed-Loop RHC')
    except Exception:
        pass

    gs = gridspec.GridSpec(4, 2, width_ratios=[1.65, 1],
                           hspace=0.52, wspace=0.22,
                           left=0.03, right=0.97, top=0.93, bottom=0.06)

    ax3d   = fig.add_subplot(gs[:, 0], projection='3d')
    ax_xy  = fig.add_subplot(gs[0, 1])
    ax_xz  = fig.add_subplot(gs[1, 1])
    ax_tel = fig.add_subplot(gs[2, 1])
    ax_tilt = fig.add_subplot(gs[3, 1])

    style_2d_ax(ax_xy)
    style_2d_ax(ax_xz)
    style_2d_ax(ax_tel)
    style_2d_ax(ax_tilt)
    ax3d.set_facecolor(C_BG)

    # ---------- Compute z_ceil for distinguishing tall vs short ----------
    TALL_THRESH = Z_CEILING  # anything above this cannot be overflown

    # ---------- 3D: Draw buildings (two styles) ----------
    for obs in planner.obstacles:
        is_short = obs['zu'] <= TALL_THRESH
        fc = C_LOW  if is_short else C_TALL
        ec = C_LOW_E if is_short else C_TALL_E
        disp_zu = obs['zu'] if is_short else 35.0
        draw_cuboid_3d(ax3d, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                       obs['zl'], disp_zu, fc, ec, face_alpha=0.65)

    # 3D NFZ volumes (very transparent)
    for nfz in planner.no_fly_zones:
        draw_cuboid_3d(ax3d, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                       0, 28, C_NFZ, C_NFZ_E, face_alpha=0.08)

    # Reference ghost path
    ax3d.plot(ref_pos[:,0], ref_pos[:,1], ref_pos[:,2],
              '-', color=C_REF, lw=0.7, alpha=0.18)

    # Z ceiling plane (subtle)
    xl_env = min(o['xl'] for o in planner.obstacles) - 100
    xu_env = max(o['xu'] for o in planner.obstacles) + 100
    yl_env = min(o['yl'] for o in planner.obstacles) - 100
    yu_env = max(o['yu'] for o in planner.obstacles) + 100
    xx, yy = np.meshgrid([xl_env, xu_env], [yl_env, yu_env])
    zz = np.full_like(xx, Z_CEILING)
    ax3d.plot_surface(xx, yy, zz, alpha=0.04, color='#58a6ff',
                      linewidth=0, antialiased=False)

    ax3d.scatter(*x_s, color='#3fb950', s=90, zorder=9,
                 marker='o', label='Start', depthshade=False)
    ax3d.scatter(*x_f, color='#f0883e', s=130, zorder=9,
                 marker='*', label='Goal', depthshade=False)

    # Axis limits
    all_x = [x_s[0], x_f[0]] + [o['xl'] for o in planner.obstacles] + \
            [o['xu'] for o in planner.obstacles]
    all_y = [x_s[1], x_f[1]] + [o['yl'] for o in planner.obstacles] + \
            [o['yu'] for o in planner.obstacles]
    xlo, xhi = min(all_x) - 200, max(all_x) + 200
    ylo, yhi = min(all_y) - 200, max(all_y) + 200

    ax3d.set_xlim(xlo, xhi)
    ax3d.set_ylim(ylo, yhi)
    ax3d.set_zlim(0, 38)
    ax3d.set_xlabel('X (m)', color=C_TXT2, fontsize=8, labelpad=2)
    ax3d.set_ylabel('Y (m)', color=C_TXT2, fontsize=8, labelpad=2)
    ax3d.set_zlabel('Z (m)', color=C_TXT2, fontsize=8, labelpad=2)
    ax3d.set_title('3-D Urban Canyon  |  \u25a0 Tall (lateral)  \u25a0 Low (fly-over)',
                   color=C_TXT, fontsize=11, fontweight='bold', pad=4)
    ax3d.view_init(elev=28, azim=-50)
    ax3d.tick_params(colors=C_TXT2, labelsize=7)
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor(C_BORDER)
    ax3d.yaxis.pane.set_edgecolor(C_BORDER)
    ax3d.zaxis.pane.set_edgecolor(C_BORDER)
    ax3d.legend(fontsize=8, loc='upper left', facecolor=C_PANEL,
                labelcolor=C_TXT, framealpha=0.9, edgecolor=C_BORDER)

    # ---------- Dynamic 3D elements ----------
    trail_len = int(10.0 / dt_anim)   # 10s of trail

    trail3d,  = ax3d.plot([], [], [], '-', color=C_ACT, lw=2.0, alpha=0.9)

    # Drone: single prominent glowing dot
    drone_body = ax3d.scatter([], [], [], s=220, color=C_DRN,
                              zorder=12, depthshade=False,
                              edgecolors='white', linewidths=1.2,
                              marker='o')

    # HUD text (clean monospace)
    hud_txt = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                           color=C_TXT, fontsize=8.5,
                           fontfamily='monospace', va='top',
                           bbox=dict(boxstyle='round,pad=0.4',
                                     facecolor=C_PANEL,
                                     edgecolor=C_BORDER,
                                     alpha=0.85))

    # ---------- 2D Top-Down Map (X-Y) ----------
    for obs in planner.obstacles:
        is_short = obs['zu'] <= TALL_THRESH
        fc = C_LOW  if is_short else C_TALL
        ec = C_LOW_E if is_short else C_TALL_E
        draw_rect_2d(ax_xy, obs['xl'], obs['xu'], obs['yl'], obs['yu'],
                     fc, ec, alpha=0.75)
    for nfz in planner.no_fly_zones:
        draw_rect_2d(ax_xy, nfz['xl'], nfz['xu'], nfz['yl'], nfz['yu'],
                     C_NFZ, C_NFZ_E, alpha=0.25, ls='--')

    ax_xy.plot(ref_pos[:,0], ref_pos[:,1], '-',
               color=C_REF, lw=0.9, alpha=0.30)
    trail2d, = ax_xy.plot([], [], '-', color=C_ACT, lw=1.8, alpha=0.9)
    dot2d,   = ax_xy.plot([], [], 'o', color=C_DRN, ms=8, zorder=10,
                          markeredgecolor='white', markeredgewidth=0.5)
    ax_xy.plot(x_s[0], x_s[1], 'o', color='#3fb950', ms=8, zorder=9)
    ax_xy.plot(x_f[0], x_f[1], '*', color='#f0883e', ms=12, zorder=9)
    ax_xy.set_xlim(xlo, xhi)
    ax_xy.set_ylim(ylo, yhi)
    ax_xy.set_xlabel('X (m)', fontsize=7.5)
    ax_xy.set_ylabel('Y (m)', fontsize=7.5)
    ax_xy.set_title('Top-Down Map (X-Y)', color=C_TXT, fontsize=9.5, fontweight='bold')

    # ---------- 2D Side Elevation Profile (X-Z) ----------
    for obs in planner.obstacles:
        is_short = obs['zu'] <= TALL_THRESH
        fc = C_LOW  if is_short else C_TALL
        ec = C_LOW_E if is_short else C_TALL_E
        disp_zu = obs['zu'] if is_short else 35.0
        draw_rect_2d(ax_xz, obs['xl'], obs['xu'], obs['zl'], disp_zu,
                     fc, ec, alpha=0.75)

    ax_xz.axhline(Z_CEILING, color=C_REF, lw=0.8, ls=':', alpha=0.6)
    ax_xz.text(xhi - 400, Z_CEILING + 1.0, f'z_max={Z_CEILING:.0f}m',
               color=C_REF, fontsize=6.5, alpha=0.8)

    ax_xz.plot(ref_pos[:,0], ref_pos[:,2], '-',
               color=C_REF, lw=0.9, alpha=0.30)
    trail_xz, = ax_xz.plot([], [], '-', color=C_ACT, lw=1.8, alpha=0.9)
    dot_xz,   = ax_xz.plot([], [], 'o', color=C_DRN, ms=8, zorder=10,
                           markeredgecolor='white', markeredgewidth=0.5)
    ax_xz.plot(x_s[0], x_s[2], 'o', color='#3fb950', ms=8, zorder=9)
    ax_xz.plot(x_f[0], x_f[2], '*', color='#f0883e', ms=12, zorder=9)
    ax_xz.set_xlim(xlo, xhi)
    ax_xz.set_ylim(0, 38)
    ax_xz.set_xlabel('X (m)', fontsize=7.5)
    ax_xz.set_ylabel('Z (m)', fontsize=7.5)
    ax_xz.set_title('Side Elevation (X-Z Profile)', color=C_TXT, fontsize=9.5, fontweight='bold')

    # ---------- Altitude side-view strip ----------
    ax_tel.set_xlabel('Sim time (s)', fontsize=7.5)
    ax_tel.set_title('Live Telemetry', color=C_TXT, fontsize=9.5, fontweight='bold')

    ln_alt,  = ax_tel.plot([], [], '-',  color=C_ALT, lw=1.6, label='Altitude (m)')
    ln_spd,  = ax_tel.plot([], [], '-',  color=C_SPD, lw=1.6, label='Speed (m/s)')
    ln_err,  = ax_tel.plot([], [], '--', color=C_ERR, lw=1.2, label='Track err ×5 (m)')

    # Draw altitude ceiling line
    ax_tel.axhline(Z_CEILING, color=C_REF, lw=0.8, ls=':', alpha=0.6)

    ax_tel.legend(fontsize=6.5, facecolor=C_PANEL, labelcolor=C_TXT,
                  framealpha=0.9, edgecolor=C_BORDER, loc='upper right')

    # ---------- Drone Tilt (Roll / Pitch) ----------
    ax_tilt.set_xlabel('Sim time (s)', fontsize=7.5)
    ax_tilt.set_title('Drone Tilt  (Roll φ  |  Pitch θ)',
                      color=C_TXT, fontsize=9.5, fontweight='bold')
    ax_tilt.axhline(0, color=C_BORDER, lw=0.8, ls='--', alpha=0.7)   # zero reference
    ax_tilt.axhline( 20, color=C_ERR, lw=0.5, ls=':', alpha=0.4)     # +20° soft limit
    ax_tilt.axhline(-20, color=C_ERR, lw=0.5, ls=':', alpha=0.4)     # -20° soft limit

    ln_roll,  = ax_tilt.plot([], [], '-', color=C_ROLL,  lw=1.6, label='Roll φ (°)')
    ln_pitch, = ax_tilt.plot([], [], '-', color=C_PITCH, lw=1.6, label='Pitch θ (°)')
    ax_tilt.set_ylabel('Angle (°)', fontsize=7.5, color=C_TXT2)
    ax_tilt.set_ylim(-35, 35)
    ax_tilt.legend(fontsize=6.5, facecolor=C_PANEL, labelcolor=C_TXT,
                   framealpha=0.9, edgecolor=C_BORDER, loc='upper right')

    buf_t = []; buf_alt = []; buf_spd = []; buf_err = []
    buf_roll = []; buf_pitch = []
    TEL_WIN = 350

    # ---------- Super-title ----------
    plt.suptitle(
        'B-Spline MILP  \u2022  Closed-Loop RHC  \u2022  SE(3) Geometric Control  \u2022  Dryden Wind'
        '      [SPACE \u2192 pause / rotate]',
        color=C_TXT, fontsize=10, fontweight='bold', y=0.99)

    # ------------------------------------------------------------------ #
    #  Animation update                                                    #
    # ------------------------------------------------------------------ #

    def on_key(event):
        if event.key == ' ':
            paused[0] = not paused[0]
            status = 'PAUSED \u2014 rotate/zoom freely' if paused[0] else 'PLAYING'
            print(f'  [{status}]  press SPACE to toggle')

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if paused[0]:
            return ()
        if frame >= N:
            return (trail3d, drone_body, trail2d, dot2d, trail_xz, dot_xz,
                    hud_txt, ln_alt, ln_spd, ln_err, ln_roll, ln_pitch)

        pos = a_pos[frame]
        t_f = a_t[frame]
        spd = np.linalg.norm(a_vel[frame])
        alt = pos[2]
        phi, theta, psi = a_eul[frame]
        w_acc = a_wind[frame]

        ref_idx = min(int(t_f / dt_ref_s), len(ref_pos) - 1)
        err = np.linalg.norm(pos - ref_pos[ref_idx])

        # ---- 3D trail ----
        lo = max(0, frame - trail_len)
        trail3d.set_data(a_pos[lo:frame+1, 0], a_pos[lo:frame+1, 1])
        trail3d.set_3d_properties(a_pos[lo:frame+1, 2])

        # ---- Drone body dot ----
        drone_body._offsets3d = ([pos[0]], [pos[1]], [pos[2]])

        # ---- Slow camera drift ----
        ax3d.view_init(elev=24 + 5 * np.sin(t_f * 0.04),
                       azim=-50 + t_f * 0.15)

        # ---- HUD ----
        mode = '\u2191 FLY-OVER' if alt > 18 else '\u2194 LATERAL'
        hud_txt.set_text(
            f'  t   = {t_f:6.1f} s\n'
            f'  X   = {pos[0]:+7.0f} m\n'
            f'  Y   = {pos[1]:+7.0f} m\n'
            f'  Z   = {pos[2]:6.2f} m\n'
            f'  spd = {spd:6.1f} m/s\n'
            f'  err = {err:6.2f} m\n'
            f'  wind= {np.linalg.norm(w_acc):5.2f} m/s\u00b2\n'
            f'  mode: {mode}'
        )

        # ---- 2D Top-Down trail + dot ----
        trail2d.set_data(a_pos[lo:frame+1, 0], a_pos[lo:frame+1, 1])
        dot2d.set_data([pos[0]], [pos[1]])

        # ---- 2D Side Elevation trail + dot ----
        trail_xz.set_data(a_pos[lo:frame+1, 0], a_pos[lo:frame+1, 2])
        dot_xz.set_data([pos[0]], [pos[2]])

        # ---- Telemetry ----
        buf_t.append(t_f)
        buf_alt.append(alt)
        buf_spd.append(spd)
        buf_err.append(err * 5)
        if len(buf_t) > TEL_WIN:
            buf_t.pop(0); buf_alt.pop(0)
            buf_spd.pop(0); buf_err.pop(0)

        ln_alt.set_data(buf_t, buf_alt)
        ln_spd.set_data(buf_t, buf_spd)
        ln_err.set_data(buf_t, buf_err)
        ax_tel.set_xlim(max(0, t_f - TEL_WIN * dt_anim), t_f + 2)
        ax_tel.set_ylim(0, max(35, spd + 5))

        # ---- Drone Tilt (Roll / Pitch in degrees) ----
        roll_deg  = np.degrees(phi)
        pitch_deg = np.degrees(theta)
        buf_roll.append(roll_deg)
        buf_pitch.append(pitch_deg)
        if len(buf_roll) > TEL_WIN:
            buf_roll.pop(0); buf_pitch.pop(0)

        ln_roll.set_data(buf_t, buf_roll)
        ln_pitch.set_data(buf_t, buf_pitch)
        # Dynamic y-limit: keep ±35° headroom, expand if needed
        max_ang = max(35.0, abs(roll_deg) + 5, abs(pitch_deg) + 5)
        ax_tilt.set_xlim(max(0, t_f - TEL_WIN * dt_anim), t_f + 2)
        ax_tilt.set_ylim(-max_ang, max_ang)

        return (trail3d, drone_body, trail2d, dot2d, trail_xz, dot_xz,
                hud_txt, ln_alt, ln_spd, ln_err, ln_roll, ln_pitch)

    # ------------------------------------------------------------------ #
    #  Launch                                                              #
    # ------------------------------------------------------------------ #
    interval_ms = int(1000 / ANIM_FPS)
    fig._anim = FuncAnimation(fig, update, frames=N,
                               interval=interval_ms, blit=False, repeat=False)

    print(f"\n[Phase 3] Launching GUI ({N} frames @ {ANIM_FPS} fps, "
          f"speed x{SPEED_MULT:.0f}) -- close window to exit.\n")

    if save_mp4:
        saved = False
        print("Saving animation ...")
        try:
            fig._anim.save('flight.mp4', writer='ffmpeg', fps=ANIM_FPS,
                            dpi=100, savefig_kwargs={'facecolor': C_BG})
            print("  Saved -> flight.mp4")
            saved = True
        except Exception as e:
            print(f"  ffmpeg unavailable, saving GIF ...")
        if not saved:
            try:
                fig._anim.save('flight.gif', writer='pillow', fps=ANIM_FPS,
                                dpi=80, savefig_kwargs={'facecolor': C_BG})
                print("  Saved -> flight.gif")
            except Exception as e2:
                print(f"  GIF save failed: {e2}")

    plt.show()
    print("Done.")


if __name__ == "__main__":
    main()
