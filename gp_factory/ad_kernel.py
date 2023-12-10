import timeit
import sys
import numpy as np
from numpy import linalg as la
from scipy.linalg import sqrtm
from .gp import GaussianProcess


class ADKernel(GaussianProcess):
    """Affine dense kernel GP."""

    name = "ad_kernel"

    def __init__(self, data, sgm=1, reg_param=1, rf_d=None, seed=None, path=None):
        super().__init__(*data, path=path)
        self.sgm = sgm
        self.reg_param = reg_param
        self.kernel = []
        self.inv_kernel = []
        self.k_vec = []

    def rbf(self, x, axis=1, keepdims=False):
        return np.exp(
            -np.sum(np.square(x), axis=axis, keepdims=keepdims) / (2 * self.sgm**2)
        )

    def _compute_kernel_helper(self, x_test=None):
        train_k = self.rbf(self.x_train, keepdims=True)  # (n,1)
        if x_test is not None:
            test_k = self.rbf(x_test, keepdims=True)  # (n_t,1)
        else:
            test_k = train_k
            x_test = self.x_train
        k_mat = train_k @ test_k.T  # (n,n_t)
        x_diff = self.x_train.reshape((self.n, 1, self.d)) - x_test  # (n,n_t,d)
        diag_k = self.rbf(x_diff, axis=2)  # (n,n_t)
        return k_mat, diag_k
        # return k_mat

    def _compute_kernel(self, x_test=None, y_test=None):
        k_mat, diag_k = self._compute_kernel_helper(x_test)
        # k_mat = self._compute_kernel_helper(x_test)
        ytr_sum = np.sum(self.y_train, axis=1, keepdims=True)  # (n,1)
        if y_test is not None:
            ys = ytr_sum @ np.sum(y_test, axis=1, keepdims=True).T  # (n,n_t)
        else:
            ys = ytr_sum @ ytr_sum.T
            y_test = self.y_train
        diag_diff = (self.y_train @ y_test.T) * (diag_k - k_mat)  # (n,n_t)
        return k_mat * ys + diag_diff  # (n,n_t)
        # return k_mat * ys #(n,n_t)

    def _compute_entries(self, x_test, y_test):
        ys = np.square(np.sum(y_test, 1))  # (n_t)
        test_k = self.rbf(x_test)  # (n_t)
        k_diag = np.square(test_k)  # (n_t)
        diff = np.einsum("ij,ji->i", y_test, y_test.T) * (1 - k_diag)  # (n_t)
        return k_diag * ys + diff  # (n_t)
        # return k_diag * ys

    def _compute_k_h(self, x_test):  # n_t=1
        k_mat, diag_k = self._compute_kernel_helper(x_test)  # (n,n_t)
        diag_diff = (diag_k - k_mat) * self.y_train  # (n,m+1)
        self.k_h = k_mat * np.sum(self.y_train, 1, keepdims=True) + diag_diff  # (n,m+1)

    def train(self):
        # kernel training
        tic = timeit.default_timer()
        self.kernel = self._compute_kernel()
        self.inv_kernel = la.inv(
            self.kernel + self.reg_param * np.identity(self.n)
        )  # (n,n)
        toc = timeit.default_timer()
        self.training_time = toc - tic

    def test(self, x_test, y_test):
        # c = (K+lambdaI)^{-1}K(x)
        tic = timeit.default_timer()
        self.n_t = len(x_test)
        self.k_vec = self._compute_kernel(x_test, y_test)  # (n,n_t)
        pred = self.z_train.T @ self.inv_kernel @ self.k_vec  # (dim,n_t)
        toc = timeit.default_timer()
        self.test_time = toc - tic
        return pred.T

    def sigma(self, x_test, y_test):  # n_t=1
        return self._compute_entries(x_test, y_test) - np.einsum(
            "ij,jk,ki->i", self.k_vec.T, self.inv_kernel, self.k_vec
        )

    def mean_var(self, x_test):  # n_t=1
        tic = timeit.default_timer()
        self._compute_k_h(x_test)
        meanvar = self.z_train @ self.inv_kernel @ self.k_h  # m+1
        toc = timeit.default_timer()
        self.meanvar_time = toc - tic
        return meanvar.T  # y @ meanvar

    def sigma_var(self):  # n_t=1
        sigmavar = sqrtm(
            abs(np.ones(self.m + 1) - self.k_h.T @ self.inv_kernel @ self.k_h)
        )  # (m+1,m+1)
        # norm(y @ sigmavar)
        return sigmavar
