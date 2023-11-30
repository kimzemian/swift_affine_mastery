import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gp_factory.init_gp import init_trained_gp


def load_and_split_data(data_path):
    grid_data = np.load(data_path + "grid_data.npz")
    xs, ys, zs = grid_data["xs"], grid_data["ys"], grid_data["zs"]
    # test/train split
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        np.asarray(xs), np.asarray(ys), np.asarray(zs), test_size=0.20, shuffle=True
    )
    print("training size: ", len(z_train), "test size: ", len(z_test))
    return x_train, x_test, y_train, y_test, z_train, z_test


def plot(data, name, gp_names):
    plot_path = "/home/kk983/fast_control/plots/"
    steps = data.shape[1]
    xs = np.arange(100, steps * 101, 100)
    xs = np.tile(xs, 10)
    plt.figure()
    sns.lineplot(
        x=xs,
        y=data[0, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label=gp_names[0],
    )
    sns.lineplot(
        x=xs,
        y=data[2, :, :].flatten("F"),
        estimator=np.median,
        errorbar=lambda x: (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        label=gp_names[2],
    )
    sns.lineplot(x=xs, y=data[1, :, :].flatten("F"), label=gp_names[1])
    sns.lineplot(x=xs, y=data[3, :, :].flatten("F"), label=gp_names[3])
    plt.xlabel("number of random features")
    plt.ylabel(f"{name}")
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"static_tests/{name}_grid.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    swift_path = "/share/dean/fast_control/models/swift_grid/"
    x_train, x_test, y_train, y_test, z_train, z_test = load_and_split_data(swift_path)
    train_data = x_train, y_train, z_train
    test_data = x_test, y_test, z_test

    seeds = [1, 26, 42, 67, 101, 218, 314, 415, 512, 666]
    num_steps = len(z_train) // 100
    rmse = np.empty([4, num_steps, 10])
    mae = np.empty([4, num_steps, 10])
    names = np.empty(4)

    paths = [swift_path + f"m{i}_config.toml" for i in range(1, 5)]

    for num in tqdm(range(num_steps)):
        for seed in seeds:
            for i in {0, 2}:
                gp = init_trained_gp(paths[i], train_data, seed, 100 * (num + 1))
                names[i] = gp.name
                z_pred = gp.test(x_test, y_test)
                rmse[i, num, seed] = mean_squared_error(z_test, z_pred, squared=False)
                mae[i, num, seed] = mean_absolute_error(z_test, z_pred)

    for i in {1, 3}:
        gp = init_trained_gp(paths[i], train_data)
        names[i] = gp.name
        z_pred = gp.test(x_test, y_test)
        rmse[i, :, :] = mean_squared_error(z_test, z_pred, squared=False)
        mae[i, :, :] = mean_absolute_error(z_test, z_pred)

    np.save(swift_path + "static_tests_rmse_grid.npy", rmse)
    np.save(swift_path + "static_tests_mae_grid.npy", mae)
    plot(rmse, "rmse", names)
    plot(mae, "mae", names)
