"""Methods that generate/save data."""
import numpy as np
import numpy.linalg as la
import torch

from swift_control.eval import eval_cs, simulate_sys


def build_ccf_data(system, system_est , xs, us, ts):
    """Estimate error in the derivate of the CCF function.

    uses forward differencing
    """
    av_x = (xs[:-1] + xs[1:]) / 2
    zs = [
        (
            system.lyap.eval(xs[i + 1], ts[i + 1])
            - system.lyap.eval(xs[i], ts[i])
        )
        / (ts[i + 1] - ts[i])
        - system_est.lyap.eval_dot(av_x[i], us[i], ts[i])
        for i in range(len(us))
    ]
    ys = np.concatenate((np.ones((len(us), 1)), us), axis=1)
    return av_x, ys, zs


def xdot_training_data_gen(plant, controller, x_0=None, T=None, num_steps=None):
    """Generate training data given a controller."""
    xs, us, ts = simulate_sys(plant.system, controller, x_0, T, num_steps)
    ys = np.concatenate((np.ones((len(us), 1)), us), axis=1)
    x_dots = np.array([plant.system.eval_dot(x, u, t) for x, u, t in zip(xs, us, ts)])
    return xs[:-1], ys, x_dots

def training_data_gen(plant, controller, x_0=None, T=None, num_steps=None):
    """Generate training data given a controller."""
    xs, us, ts = simulate_sys(plant.system, controller, x_0, T, num_steps)
    xs, ys, zs = build_ccf_data(plant.system, plant.system_est, xs, us, ts)  # ts=ts[1:-1]
    return xs, ys, zs


def create_grid_data(plant, T=None, num_steps=None, data_gen=training_data_gen):
    """Initialize grided training data.

    run a grid of initial points with nominal controller for several steps
    """
    initial_x0s = (
        np.mgrid[0.1 : np.pi : 1, -1:1.1:0.4, 0 : np.pi : 1, -1:1.1:0.4]
        .reshape(4, -1)
        .T
    )
    for i, x_0 in enumerate(initial_x0s):
        if i == 0:
            xs, ys, zs = data_gen(
                plant, plant.qp_controller, torch.from_numpy(x_0), T, num_steps
            )
        else:
            x, y, z = data_gen(
                plant, plant.qp_controller, torch.from_numpy(x_0), T, num_steps
            )
            xs = np.concatenate((xs, x))
            ys = np.concatenate((ys, y))
            zs = np.concatenate((zs, z))

    x, y, z = data_gen(
        plant, plant.qp_controller, torch.FloatTensor([0.1, 0, 0, 0]), T, num_steps
    )
    xs = np.concatenate((xs, x))
    ys = np.concatenate((ys, y))
    zs = np.concatenate((zs, z))
    return xs, ys, zs


def diff_data_gen(plant, controller, x_0=None, T=None, num_steps=None):
    """Return norm of (true c_dot for gp/qp controller - oracle controller)"""

    gp_cs, ts = eval_cs(plant.system, controller, x_0, T, num_steps)

    qp_cs, _ = eval_cs(plant, plant.qp_controller, x_0, T, num_steps)
    model_cs, _ = eval_cs(plant, plant.oracle_controller, x_0, T, num_steps)

    return la.norm(gp_cs - model_cs, axis=1), la.norm(qp_cs - model_cs, axis=1), ts
