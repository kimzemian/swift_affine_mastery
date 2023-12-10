import math
import timeit

import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess


class ADPRandomFeatures(GaussianProcess):
    """Affine dot product random features GP."""

    name = "adp_rf"

    def __init__(self, data, sgm=1, reg_param=1, rf_d=None, seed=None, path=None):
        super().__init__(*data, path=path)
        if seed is not None:
            np.random.seed(seed)
        self.rf_d = rf_d
        self.sgm = sgm
        self.reg_param = reg_param
        self.rf_cov = (self.sgm**2) * np.identity(self.d)
        self.s = (self.m + 1) * self.rf_d
        self.samples = np.random.multivariate_normal(
            self.rf_mu, self.rf_cov, size=((self.m + 1) * self.rf_d // 2)
        )  # (s/2,d)

    def _compute_phi(self, x):  # (n,s)  first var: n or n_t
        phi = np.empty((len(x), self.s))
        dot_product = x @ self.samples.T  # (n,s/2)
        phi[:, 0::2] = np.sin(dot_product)
        phi[:, 1::2] = np.cos(dot_product)
        phi = math.sqrt(2 / self.rf_d) * phi
        return phi

    def _compute_cphi(self, phi, y):  # (n,s) first,third var: n or n_t
        pre_cphi = y[:, :, np.newaxis] * phi.reshape((len(y), self.m + 1, -1))  # (n,s)
        return pre_cphi.reshape((len(y), -1))

    def train(self):
        tic = timeit.default_timer()
        phi = self._compute_phi(self.x_train)  # (n,s)
        self.cphi = self._compute_cphi(phi, self.y_train)  # (n,s)
        self.inv_cphi = la.inv(
            self.cphi.T @ self.cphi
            + self.reg_param * np.identity((self.m + 1) * self.rf_d)
        )  # (s,s)
        toc = timeit.default_timer()
        self.training_time = toc - tic

    def test(self, x_test=None, y_test=None):
        tic = timeit.default_timer()
        phi_test = self._compute_phi(x_test)  # (n_t,rf_d)
        self.cphi_test = self._compute_cphi(phi_test, y_test)  # (n_t,s)
        pred = self.cphi_test @ self.inv_cphi @ self.cphi.T @ self.z_train  # n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def sigma(self, x_test=None, y_test=None):
        return self.reg_param * np.einsum(
            "ij,jk,ik->i", self.cphi_test, self.inv_cphi, self.cphi_test
        )

    def estimate_ckernel(self):
        return self.cphi @ self.cphi.T

    def mean_var(self, x_test):  # n_t=1
        tic = timeit.default_timer()
        self.phi_test = self._compute_phi(x_test)  # (n_t,s)
        rest = np.reshape(
            self.inv_cphi @ self.cphi.T @ self.z_train, (self.rf_d, -1), order="F"
        )  # (rf_d,m+1)
        meanvar = np.einsum(
            "ij,ji->i", self.phi_test.reshape((self.m + 1, -1)), rest
        )  # (m+1)
        toc = timeit.default_timer()
        self.meanvar_time = toc - tic
        # y @  meanvar
        return meanvar

    def sigma_var(self):  # n_t=1
        test = self.phi_test.reshape((self.m + 1, -1))  # (m+1,rf_d)
        inv = self.inv_cphi.reshape(
            (-1, self.rf_d, self.m + 1, self.rf_d)
        )  # (m+1,rf_d,m+1,rf_d)
        sigmavar = np.sqrt(self.reg_param) * sqrtm(
            abs(np.einsum("ij,ijkl,kl->ik", test, inv, test))
        )  # (m+1,m+1)
        # norm(y @ sigmavar.T)
        return sigmavar.T
