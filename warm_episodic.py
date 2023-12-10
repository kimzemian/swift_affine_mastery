from swift_control.data import (
    create_grid_data,
    training_data_gen,
    xdot_training_data_gen,
)
from swift_control.eval import eval_cs
from swift_control.train import train_episodic
from plant_factory import ControllerFactory
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import mosek

def plotit(ts, qp_cs, oracle_cs, gp_cs, names, c_cdot, plot_path, x_test):
    sns.lineplot(
        x=ts,
        y=qp_cs[c_cdot],
        linestyle="dotted",
        color="black",
        label="qp_controller",
    )
    sns.lineplot(
        x=ts,
        y=oracle_cs[c_cdot],
        linestyle="dashdot",
        color="black",
        label="oracle_controller",
    )
    for model_cs, name in zip(gp_cs, names):
        sns.lineplot(
            x=ts,
            y=model_cs[c_cdot],
            label=name,
            alpha=0.5,
        )
    plt.xlabel("time")
    plt.text(0, 0, x_test)
    # plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"{int(time.time())}", dpi=300)
    plt.show()
    plt.close()    

swift_path = "/share/dean/fast_control/models/swift_grid/"
plant_conf = swift_path + "base_config.toml"
grid_path = swift_path + "grid_225_100_steps.npz"
plot_path = swift_path + "warm_episodic/"
x_0 = np.array([2.0, 0.0, 0.0, 0.0])

plant = ControllerFactory(plant_conf)
kwargs = x_0, plant.episodic_T, plant.episodic_num_steps

oracle_cs, ts = eval_cs(plant.system, plant.oracle_controller, *kwargs)
qp_cs, _ = eval_cs(plant.system, plant.qp_controller, *kwargs)


confs = []
gp_controller_pair = []
gp_cs = []
names = []
train_time = []

for i in range(1,5):
    model_conf = swift_path + f"m{i}_config.toml"
    xs, ys, zs = np.load(grid_path).values()
    data = xs[::5],ys[::5],zs[::5]
    gp_controller, gp = train_episodic(plant, model_conf, x_0, warm_start=True, data=data)
    model_cs, _ = eval_cs(plant.system, gp_controller, *kwargs)

    confs.append(model_conf)
    gp_controller_pair.append((gp_controller, gp))
    gp_cs.append(model_cs)
    names.append(gp_controller.name)

np.save(plot_path+'qp_cs.npy', qp_cs)
np.save(plot_path+'oracle_cs.npy', oracle_cs)
np.save(plot_path+'gp_cs.npy', gp_cs)
plotit(ts, qp_cs, oracle_cs, gp_cs, names, 0, plot_path, x_0)
plotit(ts, qp_cs, oracle_cs, gp_cs, names, 1, plot_path, x_0)

