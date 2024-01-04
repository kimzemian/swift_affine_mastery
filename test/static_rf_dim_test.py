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


def plot(data, name, train_time, test_time, gp_names):
    plot_path = "/share/dean/fast_control/models/swift_grid/static_tests/"
    steps = data.shape[1]
    pre_xs = np.arange(100, steps * 101, 100)
    print(pre_xs)
    xs = np.tile(pre_xs, 10)
    print(xs)
    print(data.shape)
    plt.figure()
    ax = sns.lineplot(
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

    sns.lineplot(x=xs, y=data[1, :, :].flatten("F"), label=gp_names[1], alpha=0.5)
    sns.lineplot(x=xs, y=data[3, :, :].flatten("F"), label=gp_names[3], alpha=0.5)

    ax2 = plt.twinx()
    for i in range(4):
        sns.lineplot(x=pre_xs, y=train_time[i,:], ax=ax2)
    # sns.lineplot(x=xs, y=test_time, ax=ax2, label="test_time")
    
    ax2.set_ylabel("training time")
    plt.xlabel("number of random features")
    ax.set_ylabel(f"{name}")
    
    plt.tight_layout()
    plt.legend()
    plt.savefig(plot_path + f"{name}_grid.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    swift_path = "/share/dean/fast_control/models/swift_grid/"
    static_path = "/share/dean/fast_control/models/swift_grid/static_tests/"
    x_train, x_test, y_train, y_test, z_train, z_test = load_and_split_data(swift_path)
    train_data = x_train, y_train, z_train
    test_data = x_test, y_test, z_test

    seeds = [1, 26, 42, 67, 101, 218, 314, 415, 512, 666]
    num_steps = len(z_train) // 100 - 1
    rmse = np.empty([4, num_steps, 10])
    mae = np.empty([4, num_steps, 10])
    names = np.empty(4,dtype=str)
    train_time = np.empty((4, num_steps, 10))
    test_time = np.empty((4, num_steps, 10))

    paths = [swift_path + f"m{i}_config.toml" for i in range(1, 5)]

    for i in {1, 3}:
        gp = init_trained_gp(paths[i], train_data)
        train_time[i,:,:] = gp.training_time
        names[i] = gp.name
        z_pred = gp.test(x_test, y_test)
        test_time[i,:,:] = gp.test_time
        rmse[i, :, :] = mean_squared_error(z_test, z_pred, squared=False)
        # mae[i, :, :] = mean_absolute_error(z_test, z_pred)

    for num in tqdm(range(num_steps)):
        for idx, seed in enumerate(seeds):
            for i in {0, 2}:
                gp = init_trained_gp(paths[i], train_data, seed, 100 * (num + 2))
                train_time[i, num, idx] = gp.training_time
                names[i] = gp.name
                z_pred = gp.test(x_test, y_test)
                test_time[i, num, idx] = gp.test_time
                rmse[i, num, idx] = mean_squared_error(z_test, z_pred, squared=False)
                # mae[i, num, idx] = mean_absolute_error(z_test, z_pred)
        np.save(static_path + "forget/static_tests_rmse_grid.npy", rmse)
        np.save(static_path + "forget/static_tests_train_time_grid.npy", train_time)
        np.save(static_path + "forget/static_tests_test_time_grid.npy", test_time)



    train_time = train_time.mean(axis=2)
    test_time = test_time.mean(axis=2)
    np.save(static_path + "forget/static_tests_rmse_grid.npy", rmse)
    np.save(static_path + "forget/static_tests_mae_grid.npy", mae)
    np.save(static_path + "forget/static_tests_train_time_grid.npy", train_time)
    np.save(static_path + "forget/static_tests_test_time_grid.npy", test_time)
    plot(rmse, "rmse", train_time, test_time, names)
    # plot(mae, "mae", train_time, test_time, names)
