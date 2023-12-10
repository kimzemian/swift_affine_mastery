import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch

from swift_control.data import xdot_training_data_gen
from swift_control.init_gpc import init_gpcontroller
from gp_factory.init_gp import init_trained_gp
from plant_factory import ControllerFactory
import numpy as np
import mosek
import time

def plot(preds, labels, ts, gps_names, x_test):
    plot_path ="/share/dean/fast_control/models/swift_grid/x_dot_preds/"
    for i in range(4):
        plt.figure()
        for  gp_pred, name in zip(preds, gps_names):
            sns.lineplot(x=ts,y=gp_pred[:,i],label=name+" $\dot{x}$ prediction")
        sns.lineplot(x=ts,y=labels[:,i],color='black', linestyle='--',label="true $\dot{x}$")
        plt.text(0, 0, x_test)
        plt.xlabel("time")
        plt.ylabel(f"{i}-th coordinate of x_dot")
        plt.tight_layout()
        plt.legend()
        plt.savefig(f"{plot_path}x_dot_error_{i}_int{time.time()}.png", dpi=300)
        plt.show()
        plt.close()


if __name__ == "__main__":

    swift_path = "/share/dean/fast_control/models/swift_grid/"
    plant_conf = swift_path + "base_config.toml"

    x_test = np.array([2.1,0,0,0])

    plant = ControllerFactory(plant_conf)
    args = x_test, plant.episodic_T, plant.episodic_num_steps
    ts = np.linspace(0, plant.episodic_T, plant.episodic_num_steps)
    x_dot_grid = np.load(swift_path + "xdotgrid_225_100_steps.npz").values()

    qp_xs, qp_ys, qp_zs = xdot_training_data_gen(plant, plant.qp_controller, *args)
    or_xs, or_ys, or_zs = xdot_training_data_gen(plant, plant.oracle_controller, *args)

    confs = []
    gp_controller_pair = []
    gp_qp_preds = []
    gp_or_preds = []
    names = []
    for i in range(1,6):
        # model_conf = swift_path + f"m{i}_config.toml"
        model_conf = '/home/kk983/swift_affine_mastery/'+ f"m{i}_config.toml"
        gp = init_trained_gp(model_conf, x_dot_grid)
        gp_controller = init_gpcontroller(plant, gp)
        # gp_xs, gp_ys, gp_zs = xdot_training_data_gen(plant, gp_controller, *args)
        gp_qp_preds.append(gp.test(qp_xs, qp_ys))
        gp_or_preds.append(gp.test(or_xs, or_ys))
        confs.append(model_conf)
        gp_controller_pair.append((gp_controller, gp))
        names.append(gp_controller.name+f'{i}')


    np.save(swift_path + "x_dot_preds/gp_qp_preds.npy", gp_qp_preds)
    np.save(swift_path + "x_dot_preds/gp_zs.npy", gp_or_preds)

    plot(gp_or_preds, or_zs, ts[:-1], names, x_test)


