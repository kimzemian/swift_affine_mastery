"""Trains data driven gps and controllers."""
import numpy as np
from tqdm import tqdm

from .data import training_data_gen
from gp_factory.init_gp import init_trained_gp
from swift_control.init_gpc import init_gpcontroller


def train_episodic(plant, model_path, x_0):
    """Episodically train data driven controllers."""
    xs, ys, zs = training_data_gen(
        plant, plant.qp_controller, x_0, plant.episodic_T, plant.episodic_num_steps
    )
    data = xs, ys, zs
    gp = init_trained_gp(model_path, data)
    print(gp.name)
    gp_controller = init_gpcontroller(plant, gp)

    for epoch in tqdm(range(plant.epochs)):
        x, y, z = training_data_gen(
            plant, gp_controller, x_0, plant.episodic_T, plant.episodic_num_steps
        )
        xs = np.concatenate((xs, x))
        ys = np.concatenate((ys, y))
        zs = np.concatenate((zs, z))
        data = xs, ys, zs

        gp = init_trained_gp(model_path, data)
        gp_controller = init_gpcontroller(plant, gp)
        print(f"iteration {epoch} done")
    return gp_controller, gp


def train_grid(plant, grid_data, model_path):
    """Train data driven controllers on a grid."""
    gp = init_trained_gp(model_path, grid_data)
    gp_controller = init_gpcontroller(plant, gp)
    return gp_controller, gp
