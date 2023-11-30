import math
import timeit

import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess


class ADRandomFeatures(GaussianProcess):
    """Affine dense random features GP."""

    name = "ad_rf"

    def __init__(self, data, sgm=1, reg_param=1, rf_d=None, seed=None, path=None):
        super().__init__(*data, path=path)
        if seed is not None:
            np.random.seed(seed)
        self.rf_d = rf_d
        self.sgm = sgm
        self.reg_param = reg_param
        self.rf_cov = (self.sgm**2) * np.identity(self.d)
        self.samples = np.random.multivariate_normal(
            self.rf_mu, self.rf_cov, size=((self.m + 1) * self.rf_d // 2)
        )  # (s/2,d)

    def _prephi(self, x):  # (n,rf_d,m+1)  first var: n or n_t
        phi = np.empty((len(x), self.rf_d, self.m + 1))
        dot_product = np.reshape(
            x @ self.samples.T, (len(x), self.rf_d // 2, -1)
        )  # (n,rf_d//2,m+1)
        phi[:, 0::2, :] = np.sin(dot_product)
        phi[:, 1::2, :] = np.cos(dot_product)
        phi = math.sqrt(2 / self.rf_d) * phi
        return phi

    def _compute_cphi(self, phi, y):  # (n,rf_d)  first var: n or n_t
        return np.squeeze(phi @ y[:, :, np.newaxis])

    def train(self):
        tic = timeit.default_timer()
        prephi = self._prephi(self.x_train)  # (n,rf_d,m+1)
        self.phi = self._compute_cphi(prephi, self.y_train)  # (n,rf_d)
        self.inv_phi = la.inv(
            self.phi.T @ self.phi + self.reg_param * np.identity(self.rf_d)
        )  # (rf_d,rf_d)
        toc = timeit.default_timer()
        self.training_time = toc - tic

    def test(self, x_test, y_test):
        # pred = phi(x)(Z'Z+simga^2I)^{-1}Z'y
        tic = timeit.default_timer()
        self.prephi_test = self._prephi(x_test)  # (n_t,rf_d,m+1)
        self.phi_test = self._compute_cphi(self.prephi_test, y_test)  # (n_t,rf_d)
        pred = self.phi_test @ self.inv_phi @ self.phi.T @ self.z_train  # n_t
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred

    def sigma(self, x_test=None, y_test=None):
        # random features variance computation
        return self.reg_param * np.einsum(
            "ij,jk,ik->i", self.phi_test, self.inv_phi, self.phi_test
        )

    def estimate_kernel(self):
        return self.phi @ self.phi.T

    def mean_var(self, x_test):  # n_t=1
        self.prephi_test = self._prephi(x_test).squeeze(axis=0)  # (rf_d,m+1)
        meanvar = (
            self.prephi_test.T @ self.inv_phi @ self.phi.T @ self.z_train
        )  # (n_t,1)
        # y @ meanvar
        return meanvar

    def sigma_var(self):  # n_t=1
        sigmavar = np.sqrt(self.reg_param) * sqrtm(
            abs(self.prephi_test.T @ self.inv_phi @ self.prephi_test)
        )  # (m+1,m+1)
        # norm(y @ sigmavar.T)
        return sigmavar.T
