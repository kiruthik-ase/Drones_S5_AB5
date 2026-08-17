"""
6-DOF Quadrotor Dynamics Model
================================
Uses ZYX Euler angles and RK4 integration.
State vector:  [x, y, z, vx, vy, vz, phi, theta, psi, p, q, r]
Control input: [total_thrust, tau_x, tau_y, tau_z]

Also contains WindModel -- a Dryden-inspired first-order Markov
turbulence model that generates persistent gust disturbances.
"""
import numpy as np


# ======================================================================= #
#  Wind turbulence model                                                    #
# ======================================================================= #

class WindModel:
    """
    Dryden-inspired wind turbulence model.

    Models wind as a first-order Markov (auto-regressive) process:
        w[k] = rho * w[k-1] + sigma * sqrt(1 - rho^2) * N(0,1)

    Parameters
    ----------
    rho   : autocorrelation coefficient (0.995 = slow, persistent gusts)
    sigma : per-axis wind acceleration std-dev (m/s^2)
              [horizontal_x, horizontal_y, vertical_z]
    seed  : random seed for reproducibility
    """
    def __init__(self, rho=0.995,
                 sigma=None,
                 seed=42):
        self.rho  = float(rho)
        self.sigma = np.array([1.8, 1.8, 0.4]) if sigma is None else np.asarray(sigma, float)
        self._rng  = np.random.RandomState(seed)
        self.wind  = np.zeros(3)          # current wind acceleration (m/s^2)

    def step(self):
        """Advance Markov chain and return wind acceleration vector (m/s^2)."""
        noise     = self._rng.randn(3)
        self.wind = (self.rho * self.wind
                     + self.sigma * np.sqrt(1.0 - self.rho**2) * noise)
        return self.wind.copy()

    def reset(self):
        self.wind = np.zeros(3)


# ======================================================================= #
#  Quadrotor rigid-body dynamics                                            #
# ======================================================================= #

class Quadrotor:
    def __init__(self, mass=1.0):
        self.mass = mass
        self.g    = 9.81

        # Inertia tensor (typical small quadrotor)
        self.Ixx = 0.0196
        self.Iyy = 0.0196
        self.Izz = 0.0264
        self.I     = np.diag([self.Ixx, self.Iyy, self.Izz])
        self.I_inv = np.linalg.inv(self.I)

        # 12-element state: pos(3), vel(3), euler(3), body_rates(3)
        self.state = np.zeros(12)

    # ------------------------------------------------------------------ #
    #  State helpers                                                       #
    # ------------------------------------------------------------------ #
    def set_state(self, pos, vel=None, euler=None, omega=None):
        self.state[0:3] = np.asarray(pos, dtype=float)
        if vel   is not None: self.state[3:6]  = np.asarray(vel,   dtype=float)
        if euler is not None: self.state[6:9]  = np.asarray(euler, dtype=float)
        if omega is not None: self.state[9:12] = np.asarray(omega, dtype=float)

    @property
    def pos(self):   return self.state[0:3].copy()

    @property
    def vel(self):   return self.state[3:6].copy()

    @property
    def euler(self): return self.state[6:9].copy()

    @property
    def omega(self): return self.state[9:12].copy()

    # ------------------------------------------------------------------ #
    #  Rotation matrix  (ZYX convention)                                   #
    # ------------------------------------------------------------------ #
    @staticmethod
    def rotation_matrix_euler(phi, theta, psi):
        cp, sp = np.cos(phi),   np.sin(phi)
        ct, st = np.cos(theta), np.sin(theta)
        cy, sy = np.cos(psi),   np.sin(psi)
        return np.array([
            [cy*ct,  cy*st*sp - sy*cp,  cy*st*cp + sy*sp],
            [sy*ct,  sy*st*sp + cy*cp,  sy*st*cp - cy*sp],
            [-st,    ct*sp,             ct*cp            ],
        ])

    def R(self):
        return self.rotation_matrix_euler(*self.state[6:9])

    # ------------------------------------------------------------------ #
    #  Equations of motion                                                 #
    # ------------------------------------------------------------------ #
    def _derivatives(self, s, u, wind_acc=None):
        """
        s        : state (12,)
        u        : [thrust, tau_x, tau_y, tau_z]
        wind_acc : (3,) wind acceleration disturbance in world frame (m/s^2)
        """
        phi, theta, psi = s[6], s[7], s[8]
        p, q, r = s[9], s[10], s[11]
        T   = u[0]
        tau = np.array([u[1], u[2], u[3]])

        Rot = self.rotation_matrix_euler(phi, theta, psi)

        # --- translational dynamics ---
        thrust_world = Rot @ np.array([0.0, 0.0, T / self.mass])
        gravity      = np.array([0.0, 0.0, -self.g])
        acc = thrust_world + gravity
        if wind_acc is not None:
            acc = acc + wind_acc          # wind disturbance

        # --- Euler-rate kinematic relation (ZYX) ---
        sp = np.sin(phi);  cp = np.cos(phi)
        ct = np.cos(theta)
        if abs(ct) < 1e-8:
            ct = np.sign(ct) * 1e-8      # avoid gimbal-lock singularity
        tt = np.tan(theta)

        euler_dot = np.array([
            p + sp*tt*q + cp*tt*r,
            cp*q - sp*r,
            (sp*q + cp*r) / ct,
        ])

        # --- rotational dynamics ---
        omega_vec = np.array([p, q, r])
        omega_dot = self.I_inv @ (tau - np.cross(omega_vec, self.I @ omega_vec))

        return np.concatenate([s[3:6], acc, euler_dot, omega_dot])

    # ------------------------------------------------------------------ #
    #  RK4 integration step                                                #
    # ------------------------------------------------------------------ #
    def step(self, u, dt, wind_acc=None):
        """
        Advance state by dt seconds.

        wind_acc : optional (3,) wind acceleration (m/s^2) in world frame.
        """
        s  = self.state
        k1 = self._derivatives(s,               u, wind_acc)
        k2 = self._derivatives(s + 0.5*dt*k1,   u, wind_acc)
        k3 = self._derivatives(s + 0.5*dt*k2,   u, wind_acc)
        k4 = self._derivatives(s + dt*k3,        u, wind_acc)
        self.state = s + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        # wrap Euler angles to [-pi, pi]
        self.state[6:9] = (self.state[6:9] + np.pi) % (2*np.pi) - np.pi

        return self.state.copy()
