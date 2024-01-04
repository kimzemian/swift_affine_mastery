from swift_control.data import (
    create_grid_data,
    training_data_gen,
    xdot_training_data_gen,
)
from swift_control.eval import eval_cs
from swift_control.train import train_episodic
from plant_factory import ControllerFactory
import numpy as np
import mosek


swift_path = "/share/dean/fast_control/models/swift_grid/"
plant_conf = swift_path + "base_config.toml"

x_0 = np.array([2.0, 0.0, 0.0, 0.0])

plant = ControllerFactory(plant_conf)
kwargs = x_0, plant.episodic_T, plant.episodic_num_steps

oracle_cs, ts = eval_cs(plant.system, plant.oracle_controller, *kwargs)
qp_cs, _ = eval_cs(plant.system, plant.qp_controller, *kwargs)


c_cdot = 0


model_conf = swift_path + f"m3_config.toml"
gp_controller, gp = train_episodic(plant, model_conf, x_0)
model_cs, _ = eval_cs(plant.system, gp_controller, *kwargs)

import seaborn as sns

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
sns.lineplot(
    x=ts,
    y=model_cs[c_cdot],
    label=gp.name,
    alpha=0.5,
)
