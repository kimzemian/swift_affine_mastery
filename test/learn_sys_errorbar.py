import sys
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from swift_control.data import xdot_training_data_gen
from swift_control.init_gpc import init_gpcontroller
from gp_factory.init_gp import init_trained_gp
from plant_factory import ControllerFactory
import numpy as np
import mosek
import time

def load_and_split_data(path):
    grid_data = np.load(path + "xdotgrid_225_100_steps.npz")
    xs, ys, zs = grid_data["xs"], grid_data["ys"], grid_data["zs"]
    # smaller dataset
    xs, ys, zs =xs[::2],ys[::2],zs[::2]
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        np.asarray(xs), np.asarray(ys), np.asarray(zs), test_size=0.10, shuffle=True
    )
    print("training size: ", len(z_train), "test size: ", len(z_test))
    return x_train, x_test, y_train, y_test, z_train, z_test

def plot(preds, labels, ts, gps_names):
    plot_path ="/share/dean/fast_control/models/swift_grid/x_dot_preds/"
    plot_xs = np.tile(ts, labels.shape[0])
    for i in range(4):
        plt.figure()
        for  name_idx, name in enumerate(gps_names):
            sns.lineplot(
                x=plot_xs,
                y=preds[name_idx,:,i,:].flatten(),
                estimator=np.median,
                errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
                label=name+" $\dot{x}$ prediction",
            )       
        sns.lineplot(x=plot_xs,y=labels[:,i,:].flatten("F"),estimator=np.median,
            errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
            color='black', linestyle='--',label="true $\dot{x}$")
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

    plant = ControllerFactory(plant_conf)
    x_train, x_test, y_train, y_test, z_train, z_test = load_and_split_data(swift_path)
    train_data = x_train, y_train, z_train
    test_data = x_test, y_test, z_test
    args = plant.episodic_T, plant.episodic_num_steps
    ts = np.linspace(0, plant.episodic_T, plant.episodic_num_steps)

    confs = []
    names = []
    gp_controller_pair = []
    oracle_zs = np.empty((len(x_test), len(ts) - 1, x_test.shape[1]))
    qpcon_zs = np.empty((len(x_test), len(ts) - 1, x_test.shape[1]))
    gp_or_preds = np.empty((4, len(x_test), len(ts) - 1, x_test.shape[1]))
    gp_qp_preds = np.empty((4, len(x_test), len(ts) - 1, x_test.shape[1]))

    for i in range(1,2):
        # model_conf = swift_path + f"m{i}_config.toml"
        model_conf = '/home/kk983/swift_affine_mastery/'+ f"m{i}_config.toml"
        gp = init_trained_gp(model_conf, train_data)
        gp_controller = init_gpcontroller(plant, gp)
        confs.append(model_conf)
        gp_controller_pair.append((gp_controller, gp))
        names.append(gp_controller.name+f'{i}')


    for idx,x_0 in enumerate(x_test):
        or_xs, or_ys, or_zs = xdot_training_data_gen(plant, plant.oracle_controller, x_0, *args)
        qp_xs, qp_ys, qp_zs = xdot_training_data_gen(plant, plant.qp_controller, x_0, *args)
        oracle_zs[idx, :,:] = or_zs
        qpcon_zs[idx, :,:] = qp_zs
        for i in range(1,2):
            gpc, gp = gp_controller_pair[i-1]
            gp_or_preds[i-1, idx, :, :] = gp.test(or_xs, or_ys)
            gp_qp_preds[i-1, idx, :, :] = gp.test(qp_xs, qp_ys)
        

    np.save(swift_path + "x_dot_preds/oracle_zs.npy", oracle_zs)
    np.save(swift_path + "x_dot_preds/qp_zs.npy", qpcon_zs)
    np.save(swift_path + "x_dot_preds/gp_qp_preds.npy", gp_qp_preds)
    np.save(swift_path + "x_dot_preds/gp_or_preds.npy", gp_or_preds)
    plot(gp_or_preds, oracle_zs, ts[:-1], names)


