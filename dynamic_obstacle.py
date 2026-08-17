"""
DynamicObstacle
===============
Models an intruder UAV or moving vehicle that travels at constant velocity.
Used in RHC planning to demonstrate online replanning capability.
"""
import numpy as np


class DynamicObstacle:
    """
    A moving obstacle represented as a constant-velocity body.

    Parameters
    ----------
    pos0          : (3,) initial world-frame position at t=0
    velocity      : (3,) constant velocity (m/s)
    radius        : half-footprint for MILP box constraint (m)
    safety_margin : gamma added around the box (m)
    """
    def __init__(self, pos0, velocity, radius=200.0, safety_margin=60.0):
        self.pos0          = np.asarray(pos0,     dtype=float)
        self.velocity      = np.asarray(velocity, dtype=float)
        self.radius        = float(radius)
        self.safety_margin = float(safety_margin)

    def position(self, t):
        """Return world-frame centre position at simulation time t (s)."""
        return self.pos0 + self.velocity * t

    def as_nfz(self, t):
        """Return an NFZ dict compatible with UAVPlannerMILP at time t."""
        p = self.position(t)
        r = self.radius
        return {
            'xl': p[0] - r,
            'xu': p[0] + r,
            'yl': p[1] - r,
            'yu': p[1] + r,
            'gamma': self.safety_margin,
        }
