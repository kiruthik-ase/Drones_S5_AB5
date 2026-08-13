"""
Geometric Tracking Controller for a Quadrotor
Based on: T. Lee, M. Leok, N.H. McClamroch,
"Geometric Tracking Control of a Quadrotor UAV on SE(3)", CDC 2010.

Takes the full 12-state vector from quadrotor.py and reference
(position, velocity, acceleration) from the B-spline planner.
Returns [thrust, tau_x, tau_y, tau_z].
"""
import numpy as np
from quadrotor import Quadrotor


def _vee(S):
    """Extract vector from skew-symmetric matrix."""
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class GeometricController:
    def __init__(self, mass=1.0, inertia=None):
        self.mass = mass
        self.g = 9.81

        if inertia is None:
            self.I = np.diag([0.0196, 0.0196, 0.0264])
        else:
            self.I = np.asarray(inertia)

        # ---------- position loop ----------
        # Critically-damped  ω_n ≈ 4 rad/s  ->  Kp = ω² = 16,  Kd = 2ζω = 8
        self.Kp = np.diag([16.0, 16.0, 20.0])
        self.Kd = np.diag([8.0, 8.0, 9.0])

        # ---------- attitude loop ----------
        # Much faster than position loop: ω_n ≈ 30 rad/s
        self.Kr = np.diag([120.0, 120.0, 60.0])
        self.Kw = np.diag([16.0, 16.0, 8.0])

    # ------------------------------------------------------------------ #
    def compute(self, state, p_d, v_d, a_d, yaw_d=0.0):
        """
        Parameters
        ----------
        state : ndarray (12,)  [pos, vel, euler, omega]
        p_d, v_d, a_d : ndarray (3,)  desired position / velocity / accel
        yaw_d : float  desired yaw angle (rad)

        Returns
        -------
        u : ndarray (4,) [thrust, tau_x, tau_y, tau_z]
        """
        pos   = state[0:3]
        vel   = state[3:6]
        phi, theta, psi = state[6], state[7], state[8]
        omega = state[9:12]

        R = Quadrotor.rotation_matrix_euler(phi, theta, psi)

        # ---- position error ----
        e_p = pos - p_d
        e_v = vel - v_d

        # ---- desired force in world frame ----
        e3   = np.array([0.0, 0.0, 1.0])
        F_des = (-self.Kp @ e_p
                 - self.Kd @ e_v
                 + self.mass * self.g * e3
                 + self.mass * a_d)

        # ---- total thrust (projection onto current body-z) ----
        b3 = R[:, 2]
        thrust = float(np.dot(F_des, b3))
        thrust = max(0.1 * self.mass * self.g, thrust)   # never cut thrust fully

        # ---- desired rotation matrix ----
        F_norm = np.linalg.norm(F_des)
        if F_norm < 1e-6:
            F_norm = 1e-6
        b3_des = F_des / F_norm

        # Desired heading
        b1_c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        b2_des = np.cross(b3_des, b1_c)
        b2_norm = np.linalg.norm(b2_des)
        if b2_norm < 1e-6:
            b2_des = np.array([0.0, 1.0, 0.0])
        else:
            b2_des /= b2_norm
        b1_des = np.cross(b2_des, b3_des)
        R_des = np.column_stack([b1_des, b2_des, b3_des])

        # ---- attitude error ----
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = _vee(e_R_mat)
        e_w = omega   # desired ω ≈ 0

        # ---- torques ----
        torques = -self.Kr @ e_R - self.Kw @ e_w + np.cross(omega, self.I @ omega)

        return np.array([thrust, torques[0], torques[1], torques[2]])
