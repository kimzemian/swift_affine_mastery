import numpy as np
from .gp_controller import GPController

def init_gpcontroller(plant, gp):
    """Initialize controlller for a given gp."""
    gp_controller = GPController(plant.system_est, gp)
    if plant.m == 2:
        gp_controller.add_regularizer(plant.system_est.fb_lin, 25)
        gp_controller.add_static_cost(np.identity(2))
    elif plant.m == 1:
        gp_controller.add_static_cost(np.identity(1))

    gp_controller.add_stability_constraint()
    return gp_controller