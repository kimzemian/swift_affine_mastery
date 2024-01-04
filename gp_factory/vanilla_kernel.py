import timeit
import sys
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess


class VanillaKernel(GaussianProcess):
    """Affine dot product kernel GP."""

    name = "vanilla_kernel"
    
    def __init__(self, data, sgm=1, reg_param=1, rf_d=None, seed=None, path=None):
        super().__init__(*data, path=path)
        self.sgm = sgm
        self.reg_param = reg_param
        self.u_train = self.y_train[:, :-1]
        self.xu_train = np.concatenate((self.x_train, self.u_train), axis=1)
        print(self.y_train.shape,self.u_train.shape, self.xu_train.shape)

    def _compute_kernel(self, xu_test):
        x_dif = self.xu_train.reshape((self.n, 1, -1)) - xu_test
        return np.exp(
            -np.sum(np.square(x_dif), axis=2) / (2 * self.sgm**2)
        )  # (n,n) or (n,n_t)
    
    def train(self):
        tic = timeit.default_timer()
        kernel = self._compute_kernel(self.xu_train)
        self.inv_kernel = la.inv(
            kernel + self.reg_param * np.identity(self.n)
        )
        toc = timeit.default_timer()
        self.training_time = toc - tic

    def test(self, x_test=None, y_test=None):
        tic = timeit.default_timer()
        u_test = y_test[:, :-1]
        xu_test = np.concatenate((x_test, u_test), axis=1)
        k_vec = self._compute_kernel(xu_test)
        pred = self.z_train.T @ self.inv_kernel @ k_vec
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred