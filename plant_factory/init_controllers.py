"""Initialize controllers."""
import sys

sys.path.append("/home/kk983/core")
import numpy as np
import numpy.linalg as la
from core.controllers import FBLinController, LQRController, QPController
from core.dynamics import AffineQuadCLF


def init_oracle_controller(plant):
    """Initialize oracle controller."""
    if plant.m == 1:
        plant.system.lyap = AffineQuadCLF.build_care(
            plant.system, Q=np.identity(2), R=np.identity(1)
        )
        plant.system.alpha = 1 / max(la.eigvals(plant.system_est.lyap.P))
        plant.oracle_controller = QPController(plant.system, plant.system.m)
        plant.oracle_controller.add_static_cost(np.identity(1))
        plant.oracle_controller.add_stability_constraint(
            plant.system.lyap,
            comp=lambda r: plant.system.alpha * r,
            slacked=True,
            coeff=1e3,
        )
    elif plant.m == 2:
        Q, R = 10 * np.identity(4), np.identity(2)
        plant.system.lyap = AffineQuadCLF.build_care(plant.system, Q, R)
        plant.system.alpha = min(la.eigvalsh(Q)) / max(la.eigvalsh(plant.system.lyap.P))
        lqr = LQRController.build(plant.system, Q, R)
        plant.system.fb_lin = FBLinController(plant.system, lqr)
        plant.oracle_controller = QPController.build_care(plant.system, Q, R)
        plant.oracle_controller.add_regularizer(plant.system.fb_lin, 25)
        plant.oracle_controller.add_static_cost(np.identity(2))
        plant.oracle_controller.add_stability_constraint(
            plant.system.lyap,
            comp=lambda r: plant.system.alpha * r,
            slacked=True,
            coeff=1e6,
        )
    plant.oracle_controller.name = "oracle_controller"


def init_qp_controller(plant):
    """Initialize QP controller."""
    if plant.m == 1:
        plant.system_est.lyap = AffineQuadCLF.build_care(

            plant.system_est, Q=np.identity(2), R=np.identity(1)
        )
        plant.system_est.alpha = 1 / max(la.eigvals(plant.system_est.lyap.P))
        plant.qp_controller = QPController(plant.system_est, plant.system.m)
        plant.qp_controller.add_static_cost(np.identity(1))
        plant.qp_controller.add_stability_constraint(
            plant.system_est.lyap,
            comp=lambda r: plant.system_est.alpha * r,
            slacked=True,
            coeff=1e3,
        )

    elif plant.m == 2:
        Q, R = 10 * np.identity(4), np.identity(2)
        plant.system_est.lyap = AffineQuadCLF.build_care(plant.system_est, Q, R)
        plant.system_est.alpha = min(la.eigvalsh(Q)) / max(
            la.eigvalsh(plant.system_est.lyap.P)
        )
        model_lqr = LQRController.build(plant.system_est, Q, R)
        plant.system_est.fb_lin = FBLinController(plant.system_est, model_lqr)
        plant.qp_controller = QPController.build_care(plant.system_est, Q, R)
        plant.qp_controller.add_regularizer(
            plant.system_est.fb_lin, plant.nominal_regularizer
        )
        plant.qp_controller.add_static_cost(
            plant.nominal_static_cost * np.identity(2)
        )
        plant.qp_controller.add_stability_constraint(
            plant.system_est.lyap,
            comp=lambda r: plant.system_est.alpha * r,
            slacked=True,
            coeff=plant.nominal_coef,
        )
    plant.qp_controller.name = "qp_controller"