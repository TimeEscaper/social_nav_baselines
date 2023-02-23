from ._controller import AbstractController
from core.predictors import AbstractPredictor, NeuralPredictor, ConstantVelocityPredictor
import numpy as np
import do_mpc
import casadi
from typing import List


class DoMPCController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float,
                 model_type: str,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 W: float,
                 lb: List[float],
                 ub: List[float],
                 r_rob: float,
                 r_ped: float,
                 min_safe_dist: float,
                 predictor: AbstractPredictor,
                 is_store_robot_predicted_trajectory: bool,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float],
                 solver: str) -> None:
        """Initiazition of the class instance

        Args:
            init_state (np.ndarray): Initial state of the system
            goal (np.ndarray): Goal of the system
            horizon (int): Prediction horizon of the controller, [steps]
            dt (float): Time delta, [s]
            model_type (str): Type of the model, ["unicycle", "unicycle_double_integrator"]
            total_peds (int): Amount of pedestrians in the system, >= 0
            Q (np.ndarray): State weighted matrix, [state_dim * state_dim]
            R (np.ndarray): Control weighted matrix, [control_dim * control_dim]
            lb (List[float]): Lower boundaries for system states or/and controls
            ub (List[float]): Upper boundaries for system states or/and controls
            r_rob (float): Robot radius, [m]
            r_ped (float): Pedestrian radius, [m]
            min_safe_dist (float): Minimal safe distance between robot and pedestrian, [m]
            predictor (AbstractPredictor): Predictor, [Constant Velocity Predictor, Neural Predictor]
            # TODO

        Returns:
            _type_: None
        """
        super().__init__(init_state,
                         self.get_ref_direction(init_state,
                                                goal),
                         horizon,
                         dt,
                         state_dummy_ped,
                         max_ghost_tracking_time)
        self._predictor = predictor
        self._is_store_robot_predicted_trajectory = is_store_robot_predicted_trajectory
        self._ghost_tracking_times : List[int]= [0] * total_peds

        # Architecture requires at least one dummy pedestrian in the system
        if total_peds == 0:
            total_peds = 1
        self._total_peds = total_peds

        # System model
        self._model = do_mpc.model.Model("discrete")
        # state variables
        x = self._model.set_variable("_x", "x")
        y = self._model.set_variable("_x", "y")
        phi = self._model.set_variable("_x", "phi")
        if model_type == "unicycle":
            # control input variables
            v = self._model.set_variable("_u", "v")
            w = self._model.set_variable("_u", "w")
        elif model_type == "unicycle_double_integrator":
            # state variables
            v = self._model.set_variable("_x", "v")
            w = self._model.set_variable("_x", "w")
            # control input variables
            u_a = self._model.set_variable("_u", "u_a")
            u_alpha = self._model.set_variable("_u", "u_alpha")
            # discrete equations of motion
            self._model.set_rhs("v", v + u_a * dt)
            self._model.set_rhs("w", w + u_alpha * dt)
        # discrete equations of motion
        self._model.set_rhs("x", x + v * casadi.cos(phi) * dt)
        self._model.set_rhs("y", y + v * casadi.sin(phi) * dt)
        self._model.set_rhs("phi", phi + w * dt)
        # pedestrians
        self._state_peds = self._model.set_variable(
            "_tvp", "p_peds", shape=(4, total_peds))
        # covariances
        self._covariances_peds = self._model.set_variable(
            "_tvp", "cov_peds", shape=(4, total_peds)) # Each covariance matrix is represented a row [a1, a2, b1, b2]
        # goal
        current_goal = self._model.set_variable(
            "_tvp", "current_goal", shape=(len(goal)))
        # initial position on the prediction step
        p_rob_0 = self._model.set_variable(
            "_tvp", "p_rob_0", shape=(len(goal)))
        # setup
        self._model.setup()

        # MPC controller
        self._mpc = do_mpc.controller.MPC(self._model)
        # controller setup
        setup_mpc = {
            "n_horizon": horizon,
            "t_step": dt,
            "store_full_solution": is_store_robot_predicted_trajectory,
            "nlpsol_opts": {"ipopt.print_level": 0, 
                            "ipopt.sb": "yes", 
                            "print_time": 0,
                            'ipopt.linear_solver': solver}
        }
        self._mpc.set_param(**setup_mpc)
        # horizontally concatenated array of robot current position for all the pedestrians
        p_rob_hcat = casadi.hcat([self._model._x.cat[:2] for _ in range(total_peds)])
        p_peds = self._state_peds[:2, :]
        
        def _get_inversed_sum_of_mahalanobis_distances(p_rob, peds, cov):
            S = 0
            delta = (p_rob - peds) 
            for ped_ind in range(self._total_peds):
                S += 1 / (delta[:, ped_ind].T @ casadi.reshape(cov[:, ped_ind], 2, 2) @ delta[:, ped_ind])
            return S

        # stage cost
        u = self._model._u.cat
        if isinstance(predictor, NeuralPredictor):
            stage_cost = u.T @ R @ u + W * _get_inversed_sum_of_mahalanobis_distances(p_rob_hcat, p_peds, self._covariances_peds)
        elif isinstance(predictor, ConstantVelocityPredictor):
            stage_cost = u.T @ R @ u
        # terminal cost
        p_rob_N = self._model._x.cat[:3]
        delta_p = casadi.norm_2(p_rob_N - current_goal) / casadi.norm_2(p_rob_0 - current_goal)
        #terminal_cost = delta_p.T @ Q @ delta_p
        terminal_cost = Q * delta_p ** 2 # GO-MPC Cost-function

        # set cost
        self._mpc.set_objective(lterm=stage_cost,
                                mterm=terminal_cost)
        if model_type == "unicycle":
            self._mpc.set_rterm(v=1e-2, w=1e-2)
        elif model_type == "unicycle_double_integrator":                      
            self._mpc.set_rterm(u_a=1e-2, u_alpha=1e-2)
        # bounds
        if model_type == "unicycle":
            # lb
            self._mpc.bounds['lower', '_u', 'v'] = lb[0]
            self._mpc.bounds['lower', '_u', 'w'] = lb[1]
            # ub
            self._mpc.bounds['upper', '_u', 'v'] = ub[0]
            self._mpc.bounds['upper', '_u', 'w'] = ub[1]
        elif model_type == "unicycle_double_integrator":
            # lb
            self._mpc.bounds["lower", "_x", "v"] = lb[0]
            self._mpc.bounds["lower", "_x", "w"] = lb[1]
            self._mpc.bounds["lower", "_u", "u_a"] = -3      # We do not care much on acceleration constraints
            self._mpc.bounds["lower", "_u", "u_alpha"] = -3  # Hence, we constraint them with constants in this file
            # ub
            self._mpc.bounds["upper", "_x", "v"] = ub[0]
            self._mpc.bounds["upper", "_x", "w"] = ub[1]
            self._mpc.bounds["upper", "_u", "u_a"] = 3
            self._mpc.bounds["upper", "_u", "u_alpha"] = 3
        # distance to pedestrians
        lb_dist_square = (r_rob + r_ped + min_safe_dist) ** 2
        lb_dists_square = np.array([lb_dist_square for _ in range(total_peds)])
        # inequality constrain for pedestrians
        dx_dy_square = (p_rob_hcat - p_peds) ** 2
        pedestrians_distances_squared = dx_dy_square[0, :] + dx_dy_square[1, :]
        self._mpc.set_nl_cons("dist_to_peds", -pedestrians_distances_squared,
                              ub=-lb_dists_square)

        # time-variable-parameter function for pedestrian
        self._mpc_tvp_fun = self._mpc.get_tvp_template()

        def mpc_tvp_fun(t_now):
            return self._mpc_tvp_fun
        self._mpc.set_tvp_fun(mpc_tvp_fun)
        # setup
        self._mpc.setup()
        # set initial guess
        self._mpc.x0 = np.array(init_state)
        self._mpc.set_initial_guess()
        # set basic goal
        self.set_new_goal(init_state,
                          goal)

    def predict(self,
                ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor
        
        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        predicted_pedestrians_trajectories, predicted_pedestrians_covariances = self._predictor.predict(ground_truth_pedestrians_state)
        
        """ Unfold covarince into a row for MPC
        [[a1, a2
          b1, b2]]  -->  [a1, a2, b1, b2]
        """
        unfolded_predicted_pedestrians_covariances = predicted_pedestrians_covariances.reshape((self._horizon + 1, self._total_peds, 4))

        for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
            self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_pedestrians_trajectories[step].T
            self._mpc_tvp_fun['_tvp', step, 'cov_peds'] = unfolded_predicted_pedestrians_covariances[step].T
        return predicted_pedestrians_trajectories, predicted_pedestrians_covariances

    def get_predicted_robot_trajectory(self) -> List[float]:
        rob_x_pred = self._mpc.data.prediction(('_x', 'x'))[0]
        rob_y_pred = self._mpc.data.prediction(('_x', 'y'))[0]
        array_xy_pred = np.concatenate([rob_x_pred, rob_y_pred], axis=1)
        return np.ndarray.tolist(array_xy_pred)

    def make_step(self, 
                  state: np.ndarray,
                  ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances = self.predict(ground_truth_pedestrians_state)
        control = self._mpc.make_step(state).T[0]
        return control, self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances
    
    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        super().set_new_goal(current_state,
                             new_goal)
        self._mpc_tvp_fun['_tvp', :, 'current_goal'] = self.goal

    @property
    def predictor(self) -> AbstractPredictor:
        return self._predictor
    
    @property
    def init_state(self) -> np.ndarray:
        return self._init_state
    
    @property
    def ghost_tracking_times(self) -> List[int]:
        return self._ghost_tracking_times
    
    @property
    def max_ghost_tracking_time(self) -> int:
        return self._max_ghost_tracking_time

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories