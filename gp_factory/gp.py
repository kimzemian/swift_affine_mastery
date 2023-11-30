import numpy as np
from swift_control.util import load_config


class GaussianProcess():
    """
    Gaussian Process kernel.

    methods to overwrite:
    train, test, sigma, mean_var, sigma_var
    """

    def __init__(self, x_train, y_train, z_train, path=None):

        self._set_config(path)

        self.x_train = x_train
        self.y_train = y_train
        self.z_train = z_train
        self.n = len(x_train)  # number of samples
        self.d = len(x_train[0])  # dimension of the data
        self.m = len(y_train[0]) - 1  # dimension of the control input
        self.mu = 0
        self.rf_mu = np.zeros(self.d)
        self.rf_cov = np.identity(self.d)  # gives fourier transform of the RBF kernel


    def _set_config(self, path):
        """
        Set the configuration of the GP and GP controller.
        """
        model_conf = load_config(path)
        gp_conf = model_conf['gp']
        self.name = gp_conf["name"]
        self.sgm = gp_conf["sgm"]
        self.reg_param = gp_conf["reg_param"]
        self.rf_d = gp_conf['rf_d']

        gpc_conf = model_conf['gp_controller']
        self.slack = gpc_conf['slack']
        self.beta = gpc_conf['beta']
        self.coef = gpc_conf['coef']



    def init_trained_gp(self, data):
        pass