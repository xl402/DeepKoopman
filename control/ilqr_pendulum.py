import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

from ilqr import iLQR
from ilqr.dynamics import AutoDiffDynamics
from ilqr.cost import QRCost


def ilqr_pend(dt=0.1):
    x = T.dscalar("x")  # Position.
    x_dot = T.dscalar("x_dot")  # Velocity.
    u = T.dscalar("u")  # Force.

    #x_dot_dot = l*(x_dot - x**2) -u
    x_dot_dot = -np.sin(x) - u

    f = T.stack([
        x + (x_dot) * dt,
        x_dot + x_dot_dot * dt,
    ])

    x_inputs = [x, x_dot]  # State vector.
    u_inputs = [u]  # Control vector.

    dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)
    return dynamics

def pend_cost(x_goal = [0.0, 0.0], Q=None, R=None):
    Q = 1 * np.eye(2)
    R = 1 * np.eye(1)
    x_goal = np.array(x_goal)
    cost = QRCost(Q, R, x_goal=x_goal)
    return cost
