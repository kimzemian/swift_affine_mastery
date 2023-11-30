from swift_control.data import (
    create_grid_data,
    training_data_gen,
    xdot_training_data_gen,
)
from plant_factory import ControllerFactory
import numpy as np
import mosek

swift_path = "/share/dean/fast_control/models/swift_grid/"
plant_conf = swift_path + "base_config.toml"


plant = ControllerFactory(plant_conf)
xs, ys, zs = create_grid_data(
    plant, T=plant.grid_T, num_steps=plant.grid_num_steps, data_gen=training_data_gen
)
np.savez(swift_path + "grid_data.npz", xs=xs, ys=ys, zs=zs)
xdot_xs, xdot_ys, xdot_zs = create_grid_data(
    plant,
    T=plant.grid_T,
    num_steps=plant.grid_num_steps,
    data_gen=xdot_training_data_gen,
)
np.savez(swift_path + "x_dot_grid_data.npz", xs=xdot_xs, ys=xdot_ys, zs=xdot_zs)
