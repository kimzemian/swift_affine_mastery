"""Class factory for controlling an unknown system."""
from swift_control.eval import simulate_sys, eval_cs
from swift_control.util import load_config
from plant_factory.init_controllers import init_qp_controller, init_oracle_controller
from core.systems import DoubleInvertedPendulum


class ControllerFactory:
    """Control an unknown system using data driven methods."""

    def __init__(self, path):
        """Initialize an instance of the class."""
        self.config_path = path
        sys_params, sys_est_params = self._set_config(path)
        self.system = DoubleInvertedPendulum(*sys_params)
        self.system_est = DoubleInvertedPendulum(*sys_est_params)
        init_oracle_controller(self)
        init_qp_controller(self)


    def _set_config(self, path):
        """Load configuration from the TOML file."""
        factory_conf = load_config(path)

        sys_conf = factory_conf["system"]
        self.m = sys_conf["m"]
        sys_params = sys_conf["sys_params"]
        sys_est_params = sys_conf["sys_est_params"]
        
        episodic_conf = factory_conf["episodic"]
        self.epochs = episodic_conf["epochs"]
        self.episodic_T = episodic_conf["T"]
        self.episodic_num_steps = episodic_conf["num_steps"]
        self.x_0 = episodic_conf["x_0"]

        grid_conf = factory_conf["grid"]
        self.grid_T = grid_conf["T"]
        self.grid_num_steps = grid_conf["num_steps"]

        nominal_conf = factory_conf["nominal_controller"]
        self.nominal_static_cost = nominal_conf["static_cost"]
        self.nominal_regularizer = nominal_conf["regularizer"]
        self.nominal_coef = nominal_conf["coef"]

        return sys_params, sys_est_params
    

    # def init_oracle_controller(self):
    #     pass

    # def init_qp_controller(self):
    #     pass
