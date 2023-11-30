"""Initialize GP controller."""
import os
import sys

import cvxpy as cp
import numpy as np

module_path = os.path.abspath(os.path.join("."))
os.environ["MOSEKLM_LICENSE_FILE"] = module_path
import mosek  # noqa: E402

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path + "/core")

from core.controllers import QPController


class GPController(QPController):
    """Controller to ensure safety and/or stability using gp methods."""

    def __init__(self, system_est, gp):
        """Initialize controller."""
        super().__init__(system_est, system_est.m)
        self.lyap = system_est.lyap
        self.name = f"{gp.name}controller"
        self.gp = gp
        self.comp = system_est.alpha
        self.slack = gp.slack
        self.beta = gp.beta
        self.coef = gp.coef

    def add_stability_constraint(self):
        """Add stability constraint to controller.

        adds slack
        wrapper for build cons
        """
        if self.slack == "none":

            def constraint(x, t):
                return self._build_cons(x, t)

        else:
            delta = cp.Variable()
            self.variables.append(delta)
            if self.slack == "constant":
                self.static_costs.append(self.coef * cp.square(delta))
            elif self.slack == "linear":
                self.static_costs_lambda.append(
                    lambda t: (t + 1) * self.coef * cp.square(delta)
                )
            elif self.slack == "quadratic":
                self.static_costs_lambda.append(
                    lambda t: (cp.square(t) + 1) * self.coef * cp.square(delta)
                )

            def constraint(x, t):
                return self._build_cons(x, t, delta)

        self.constraints.append(constraint)

    def _build_cons(self, x, t, delta=0):
        """Build constraints for controller given x,t."""
        mv, sv = (
            self.gp.mean_var(x[np.newaxis, :]),
            self.gp.sigma_var(),
        )  # m+1, (m+1,m+1)
        input_dep = -(self.lyap.act(x, t) + mv[1:])  # m
        input_indep = delta - (
            self.lyap.drift(x, t) + mv[0] + self.comp * self.lyap.eval(x, t)
        )  # ()
        # print(input_dep, input_indep, "input_dep @ self.u.T + input_indep")
        # print('act and drift respectively at',x,t)
        # print(system_est.lyap.act(x,t))
        # print(system_est.lyap.drift(x,t))
        return cp.SOC(
            input_dep @ self.u.T + input_indep,
            self.beta * sv[1:].T @ self.u + self.beta * sv[0].T,
        )

    def eval(self, x, t):
        """Evaluate controller at x,t."""
        static_cost = cp.sum(
            [s_cost(t) for s_cost in self.static_costs_lambda]
        ) + cp.sum(self.static_costs)
        dynamic_cost = cp.sum([cost(x, t) for cost in self.dynamic_costs])
        obj = cp.Minimize(static_cost + dynamic_cost)
        cons = [constraint(x, t) for constraint in self.constraints]
        prob = cp.Problem(obj, cons)
        prob.solve(solver="MOSEK", warm_start=True)
        # print(self.gp.sigma_var())
        return self.u.value, [variable.value for variable in self.variables]

    def process(self, u):
        """Extend process from core modules."""
        u, _ = u
        if u is None:
            u = np.zeros(self.m)
        return u
