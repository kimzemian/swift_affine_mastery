import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from gp_factory.init_gp import init_trained_gp


def load_and_split_data(data_path):
    grid_data = np.load(data_path + "grid_225_100_steps.npz")
    xs, ys, zs = grid_data["xs"], grid_data["ys"], grid_data["zs"]
    # smaller dataset
    # xs, ys, zs =xs[::2],ys[::2],zs[::2]
    x_train, x_test, y_train, y_test, z_train, z_test = train_test_split(
        np.asarray(xs), np.asarray(ys), np.asarray(zs), test_size=0.20, shuffle=True
    )
    print("training size: ", len(z_train), "test size: ", len(z_test))
    return x_train, x_test, y_train, y_test, z_train, z_test

if __name__ == "__main__":
    swift_path = "/share/dean/fast_control/models/swift_grid/"
    static_path = "/share/dean/fast_control/models/swift_grid/static_tests/"
    x_train, x_test, y_train, y_test, z_train, z_test = load_and_split_data(swift_path)
    train_data = x_train, y_train, z_train
    test_data = x_test, y_test, z_test

    num_steps = len(z_train) // 100 - 1
    rmse = np.empty([1, num_steps, 10])
    mae = np.empty([1, num_steps, 10])
    names = np.empty(1,dtype=str)
    train_time = np.empty((1, num_steps, 10))
    test_time = np.empty((1, num_steps, 10))

    i=0
    gp = init_trained_gp(swift_path + f"m{5}_config.toml", train_data)
    train_time[i,:,:] = gp.training_time
    names[i] = gp.name
    z_pred = gp.test(x_test, y_test)
    test_time[i,:,:] = gp.test_time
    rmse[i, :, :] = mean_squared_error(z_test, z_pred, squared=False)
    # mae[i, :, :] = mean_absolute_error(z_test, z_pred)

    train_time = train_time.mean(axis=2)
    test_time = test_time.mean(axis=2)
    np.save(static_path + "small/vanillarmse_grid.npy", rmse)

    np.save(static_path + "small/vanillatrain_time_grid.npy", train_time)
    np.save(static_path + "small/vanillatest_time_grid.npy", test_time)
    plot(rmse, "rmse", train_time, test_time, names)
    # plot(mae, "mae", train_time, test_time, names)
