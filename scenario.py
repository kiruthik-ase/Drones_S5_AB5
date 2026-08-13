"""
Scenario Definition
===================
All environment parameters (start, goal, obstacles, no-fly zones)
are defined here so they are shared between simulation.py and visualizer.py.

Scenario: Dense Urban Canyon
  - A grid of 8 tall buildings the drone must weave through
  - 2 no-fly zones (restricted airspace corridors)
  - Drone must also vary altitude to clear a low bridge obstacle
  - Start: [2500, -1800, 15]   Goal: [-2200, 2500, 10]
"""
import numpy as np


def build_scenario(planner):
    """
    Add all obstacles and no-fly zones to a planner instance.
    Returns (x_s, v_s, a_s, x_f).
    """

    # ---------- Start / Goal ----------
    x_s = np.array([ 2500.0, -1800.0, 15.0])
    v_s = np.array([-25.0,    20.0,    0.0])
    a_s = np.array([  0.0,     0.0,    0.0])
    x_f = np.array([-2200.0,  2500.0, 10.0])

    # ---------- City Block: 8 Tall Buildings ----------
    # Row 1 (south side)
    planner.add_cuboid_obstacle(1800, 2200, -1600, -1000, 0, 80,  eta=5.0)  # B1
    planner.add_cuboid_obstacle( 800, 1200, -1600, -1000, 0, 120, eta=5.0)  # B2 (taller)
    planner.add_cuboid_obstacle(-200,  200, -1600, -1000, 0, 80,  eta=5.0)  # B3

    # Row 2 (middle)
    planner.add_cuboid_obstacle(1300, 1700,  -300,   300, 0, 100, eta=5.0)  # B4
    planner.add_cuboid_obstacle( 200,  600,  -300,   300, 0, 90,  eta=5.0)  # B5
    planner.add_cuboid_obstacle(-900, -500,  -300,   300, 0, 110, eta=5.0)  # B6 (tallest)

    # Row 3 (north side)
    planner.add_cuboid_obstacle( 600, 1000,  1000,  1600, 0, 75,  eta=5.0)  # B7
    planner.add_cuboid_obstacle(-800, -400,  1000,  1600, 0, 95,  eta=5.0)  # B8

    # ---------- No-Fly Zones (restricted corridors) ----------
    # Corridor 1: covers the eastern half broadly
    planner.add_no_fly_zone(1000, 2800, -500, 1200, gamma=20.0)
    # Corridor 2: covers the central-north area
    planner.add_no_fly_zone(-600,  800,  900, 2800, gamma=20.0)

    return x_s, v_s, a_s, x_f
