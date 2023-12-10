import sys

sys.path.append("/home/kk983/fast_control")
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from fast_control.gp_factory import (
    ADPKernel,
    ADPRandomFeatures,
    ADRandomFeatures,
    ADKernel,
)
from fast_control.gp_factory import GPFactory, init_gp_dict, train_gps
from fast_control.util import update_config
import pandas as pd


def tune_params(xs, ys, zs):
    reg_params = [0.1, 1.0, 10.0]
    sigma_values = [0.1, 1.0, 10.0]

    gp_factory = GPFactory()
    gp_dict = gp_factory.gp_param_dict
    num_gps = len(gp_dict.keys())
    # Create a MultiIndex from the combinations of reg_params and sigma_values
    index = pd.MultiIndex.from_product(
        [reg_params, sigma_values], names=["reg_param", "sigma"]
    )
    df = pd.DataFrame(index=index, columns=gp_dict.keys())
    n_splits = 1
    k_fold_rmse_data = np.empty((num_gps, n_splits))
    best_params = np.empty((num_gps, 2))
    best_rmse = np.empty((num_gps, 1))
    for reg_param in reg_params:
        for sigma in sigma_values:
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            for fold, (train_idx, val_idx) in enumerate(kf.split(xs)):
                x_train, y_train, z_train = xs[train_idx], ys[train_idx], zs[train_idx]
                x_val, y_val, z_val = xs[val_idx], ys[val_idx], zs[val_idx]

                # Train gp with the current parameters
                update_param_dict(gp_dict, sigma, reg_param)
                data = init_gp_dict(gp_factory, x_train, y_train, z_train)
                gps = train_gps(gp_factory, data, rf_d=None)

                for gp_index, gp in enumerate(gps):
                    z_pred = gp.test(x_val, y_val)
                    mse = mean_squared_error(z_val, z_pred)
                    k_fold_rmse_data[gp_index, fold] = mse
                    # Update the best lambda if the current one is better

            rmse_data = np.mean(k_fold_rmse_data, axis=1)
            for gp_index, gp in enumerate(gps):
                if rmse_data[gp_index] < best_rmse[gp_index]:
                    best_rmse[gp_index] = rmse_data[gp_index]
                    best_params[gp_index] = [reg_param, sigma]
                df.loc[(reg_param, sigma), gp.name] = rmse_data[gp_index]

    # update_config() #FIXME
    # gps = train_gps(gp_factory, data, rf_d=None)
    # return gps
    return df


def update_param_dict(gp_dict, sgm, reg_param):
    for gp_name in gp_dict:
        gp_dict[gp_name]["sgm"] = sgm
        gp_dict[gp_name]["reg_param"] = reg_param


if __name__ == "__main__":
    data_path = "/home/kk983/fast_control/data/grid/"
    data = np.load(data_path + "init_grid_medium.npz")
    xs, ys, zs = data["xs"], data["ys"], data["zs"]
    print(xs.shape, ys.shape, zs.shape)

    df = tune_params(xs, ys, zs)
    # write rmse data to a pd.DataFrame
    # rmse_data = np.mean(k_fold_rmse_data, axis=3)

    df.to_csv(data_path + "med_grid.csv", index=True, header=True)
