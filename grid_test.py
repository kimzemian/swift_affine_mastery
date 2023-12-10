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
name = "grid_225_100_steps.npz"

plant = ControllerFactory(plant_conf)
xs, ys, zs = create_grid_data(
    plant, T=plant.grid_T, num_steps=plant.grid_num_steps, data_gen=training_data_gen
)
np.savez(swift_path + name, xs=xs, ys=ys, zs=zs)
xdot_xs, xdot_ys, xdot_zs = create_grid_data(
    plant,
    T=plant.grid_T,
    num_steps=plant.grid_num_steps,
    data_gen=xdot_training_data_gen,
)
np.savez(swift_path + "xdot" + name, xs=xdot_xs, ys=xdot_ys, zs=xdot_zs)
