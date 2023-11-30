"""Initialize controllers."""
import sys
from .gp import GaussianProcess
from swift_control.util import load_config

sys.path.append("/home/kk983/core")
from gp_factory import (
    ADKernel,
    ADRandomFeatures,
    ADPKernel,
    ADPRandomFeatures,
)


def init_trained_gp(path, data, seed=42, given_rf_d=None):
    """Initialize and train specified kernel."""

    # Define a dictionary to map gp_name to corresponding class
    gp_classes = {
        ADRandomFeatures.name: ADRandomFeatures,
        ADPRandomFeatures.name: ADPRandomFeatures,
        ADKernel.name: ADKernel,
        ADPKernel.name: ADPKernel,
    }
    # get config data
    conf = load_config(path)["gp"]
    name, conf_rf_d = conf["name"], conf["rf_d"]
    sgm, reg_param = conf["sgm"], conf["reg_param"]

    # Check if gp_name is in the dictionary
    if name not in gp_classes:
        raise ValueError(f"Unsupported gp_name: {name}")

    if given_rf_d is not None:
        rf_d = given_rf_d
    # check if rf_d is provided
    elif conf_rf_d == 0:
        _, _, zs = data
        num = len(zs) // 9
        rf_d = num + 1 if num % 2 else num  # FIXME:rf_d
        rf_d = 2 * rf_d
        print(f"data size:{len(zs)}, calculated rf_d is: {rf_d}")
    else:
        rf_d = conf_rf_d

    # create gp instance and train
    set_gp = gp_classes[name](
        data, sgm=sgm, reg_param=reg_param, rf_d=rf_d, seed=seed, path=path
    )
    set_gp.train()
    print(f"gp_name: {name}, training time: {set_gp.training_time}")
    return set_gp
