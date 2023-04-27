from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS

from ._controller import AbstractController
from core.predictors import AbstractPredictor, NeuralPredictor, ConstantVelocityPredictor
from core.predictors import PedestrianTracker
import numpy as np
import do_mpc
import casadi
from typing import List, Dict, Optional


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
                 pedestrian_tracker: PedestrianTracker,
                 is_store_robot_predicted_trajectory: bool,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float],
                 solver: str,
                 cost_function: str,
                 constraint_type: str,
                 constraint_value: float) -> None:
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
            constraint_value (float): Minimal safe distance between robot and pedestrian, [m]
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
        self._ped_tracker = pedestrian_tracker
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
            sv = self._model.set_variable("_u", "sv")
            #sW = self._model.set_variable("_u", "sW")
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
        if constraint_type != "ELC":
            self._covariances_peds = self._model.set_variable(
                "_tvp", "cov_peds", shape=(4, total_peds)) # Each covariance matrix is represented a row [a1, a2, b1, b2]
            # invariant covariances
            self._inverse_covariances_peds = self._model.set_variable(
                "_tvp", "inv_cov_peds", shape=(4, total_peds)) # Each covariance matrix is represented a row [a1, a2, b1, b2]
        else:
            # self._ellipsoid_peds = self._model.set_variable(
            #     "_tvp", "ellipsoid_peds", shape=(4, total_peds)
            # )
            self._ellipsoid_axes = self._model.set_variable(
                "_tvp", "ellipsoid_axes", shape=(2, total_peds)
            )
            self._ellipsoid_angles = self._model.set_variable(
                "_tvp", "ellipsoid_angles", shape=total_peds
            )
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

        if constraint_type != "ELC":
            if True:
                R = np.array([[0.005, 0, 0],
                              [0, 0.005, 0],
                              [0, 0, 100000.]])
            else:
                R = np.eye(3)
        else:
            R = np.array([[0.005, 0, 0],
                          [0, 0.005, 0],
                          [0, 0, 100000.]])

        # stage cost
        u = self._model._u.cat
        if cost_function == "GO-MPC":
            stage_cost = u.T @ R @ u
        elif cost_function == "MD-GO-MPC":
            def _get_MD_cost(p_rob, peds, inv_cov):
                S = 0
                delta = (p_rob - peds) 
                for ped_ind in range(self._total_peds):
                    S += 1 / (delta[:, ped_ind].T @ casadi.reshape(inv_cov[:, ped_ind], 2, 2) @ delta[:, ped_ind])
                return S
            stage_cost = u.T @ R @ u + W * _get_MD_cost(p_rob_hcat, p_peds, self._inverse_covariances_peds)
        elif cost_function == "ED-GO-MPC": 
            def _get_ED_cost(p_rob, peds):
                S = 0
                delta = (p_rob - peds) 
                for ped_ind in range(self._total_peds):
                    S += 1 / (delta[:, ped_ind].T @ delta[:, ped_ind])
                return S
            stage_cost = u.T @ R @ u + W * _get_ED_cost(p_rob_hcat, p_peds)

        # terminal cost
        p_rob_N = self._model._x.cat[:3]
        delta_p = casadi.norm_2(p_rob_N[:2] - current_goal[:2]) / casadi.norm_2(p_rob_0[:2] - current_goal[:2])
        terminal_cost = Q * delta_p ** 2 # GO-MPC Cost-function

        # set cost
        self._mpc.set_objective(lterm=stage_cost + terminal_cost,
                                mterm=terminal_cost)
        
        if model_type == "unicycle":
            self._mpc.set_rterm(v=1e-2, w=1e-2, sv=1e-2,)
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

            if constraint_type != "ELC":
                self._mpc.bounds['lower', '_u', 'sv'] = -np.inf
                self._mpc.bounds['upper', '_u', 'sv'] = np.inf
            else:
                self._mpc.bounds['lower', '_u', 'sv'] = 0.
                self._mpc.bounds['upper', '_u', 'sv'] = 1.

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

        # set constraint type
        if constraint_type == "ED":
            # distance to pedestrians
            lb_dist_square = (r_rob + r_ped + constraint_value) ** 2
            lb_dists_square = np.array([lb_dist_square for _ in range(total_peds)])
            # inequality constrain for pedestrians
            dx_dy_square = (p_rob_hcat - p_peds) ** 2
            pedestrians_distances_squared = dx_dy_square[0, :] + dx_dy_square[1, :] + sv
            self._mpc.set_nl_cons("euclidean_dist_to_peds", -pedestrians_distances_squared,
                                  ub=-lb_dists_square)
        elif constraint_type == "MD":
            def _get_MD(p_rob, peds, inv_cov):
                S = 0
                delta = (p_rob - peds)
                array_mahalanobis_distances = casadi.SX(1, total_peds)
                for ped_ind in range(self._total_peds):
                    array_mahalanobis_distances[ped_ind] = delta[:, ped_ind].T @ casadi.reshape(inv_cov[:, ped_ind], 2, 2) @ delta[:, ped_ind] + sv
                return array_mahalanobis_distances 

            pedestrians_mahalanobis_distances = _get_MD(p_rob_hcat, p_peds, self._inverse_covariances_peds)
            
            def _get_determinant(matrix):
                det = (matrix[0, 0] * matrix[1, 1]) - (matrix[1, 0] * matrix[0, 1])
                return det

            def _get_MB_bounds(cov):
                array_mahalanobis_bounds = casadi.SX(1, total_peds)
                V_s = np.pi * 2 * (r_rob + r_ped + 0.3) ** 2
                for ped_ind in range(self._total_peds):
                    deter = 2 * np.pi * _get_determinant(casadi.reshape(cov[:, ped_ind], 2, 2))
                    deter = casadi.sqrt(deter)
                    array_mahalanobis_bounds[ped_ind] = 2 * casadi.log(deter * constraint_value / V_s)
                return array_mahalanobis_bounds
            
            mahalanobis_bounds = _get_MB_bounds(self._covariances_peds)

            self._mpc.set_nl_cons("mahalanobis_dist_to_peds", -pedestrians_mahalanobis_distances-mahalanobis_bounds,
                                  ub=np.zeros(total_peds))

        elif constraint_type == "ELC":
            self._n_sigma = constraint_value
            robot_peds_delta = p_rob_hcat - p_peds
            array_ellipse_bounds = casadi.SX(1, total_peds)
            for ped_ind in range(self._total_peds):
            #     array_ellipse_bounds[ped_ind] = \
            #         robot_peds_delta[:, ped_ind].T @ casadi.reshape(self._ellipsoid_peds[:, ped_ind], 2, 2)\
            #         @ robot_peds_delta[:, ped_ind]
                half_axes = (1 - sv) * self._n_sigma * self._ellipsoid_axes[:, ped_ind] + ROBOT_RADIUS + PEDESTRIAN_RADIUS + 0.3
                diag_matrix = casadi.diag(1 / (half_axes ** 2))
                rotation_matrix = casadi.blockcat(casadi.cos(self._ellipsoid_angles[ped_ind]),
                                                  -casadi.sin(self._ellipsoid_angles[ped_ind]),
                                                  casadi.sin(self._ellipsoid_angles[ped_ind]),
                                                  casadi.cos(self._ellipsoid_angles[ped_ind]))
                ellipse_matrix = rotation_matrix.T @ diag_matrix @ rotation_matrix
                array_ellipse_bounds[ped_ind] = casadi.bilin(ellipse_matrix,
                                                             robot_peds_delta[:, ped_ind],
                                                             robot_peds_delta[:, ped_ind])

            self._mpc.set_nl_cons("ellipse_bounds", -array_ellipse_bounds,
                                  ub=-1.)

        elif constraint_type == "None":
            pass
        self._constraint_type = constraint_type

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
                observation: Dict[int, np.ndarray]) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor
        
        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        tracked_predictions = self._ped_tracker.get_predictions()
        predicted_trajectories = np.tile(np.array([10000., 10000., 0., 0.]), (self._horizon + 1, self._total_peds, 1))
        predicted_covs = np.tile(np.array([[0.01, 0.0], [0., 0.01]]), (self._horizon + 1, self._total_peds, 1, 1))
        for k in tracked_predictions.keys():
            predicted_trajectories[:, k, :2] = tracked_predictions[k][0]
            predicted_covs[:, k, :] = tracked_predictions[k][1]

        predicted_covs_inv = np.linalg.inv(predicted_covs)
        predicted_covs_flatten = predicted_covs.reshape((self._horizon + 1, self._total_peds, 4))

        if self._constraint_type != "ELC":
            predicted_covs_inv_flatten = predicted_covs_inv.reshape((self._horizon + 1, self._total_peds, 4))
            for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
                self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_trajectories[step].T
                self._mpc_tvp_fun['_tvp', step, 'cov_peds'] = predicted_covs_flatten[step].T
                self._mpc_tvp_fun['_tvp', step, 'inv_cov_peds'] = predicted_covs_inv_flatten[step].T
        else:
            lambdas, es = np.linalg.eig(predicted_covs)
            half_axes = self._n_sigma * np.sqrt(lambdas)
            phis = -np.arctan2(es[:, :, 1, 0], es[:, :, 0, 0])
            # half_axes = half_axes + ROBOT_RADIUS + PEDESTRIAN_RADIUS
            # ellipses = np.zeros_like(predicted_covs)
            # for i in range(ellipses.shape[0]):
            #     for j in range(ellipses.shape[1]):
            #         rot_matrix = np.array([[np.cos(phis[i, j]), -np.sin(phis[i, j])],
            #                                [np.sin(phis[i, j]), np.cos(phis[i, j])]])
            #         ellipses[i, j] = rot_matrix.T @ np.diag((1. / half_axes[i, j]) ** 2) @ rot_matrix
            # ellipses_flatten = ellipses.reshape((self._horizon + 1, self._total_peds, 4))

            for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
                self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_trajectories[step].T
                self._mpc_tvp_fun['_tvp', step, 'ellipsoid_axes'] = half_axes[step].T
                self._mpc_tvp_fun['_tvp', step, 'ellipsoid_angles'] = phis[step]
        return predicted_trajectories, predicted_covs


    def get_predicted_robot_trajectory(self) -> List[float]:
        rob_x_pred = self._mpc.data.prediction(('_x', 'x'))[0]
        rob_y_pred = self._mpc.data.prediction(('_x', 'y'))[0]
        array_xy_pred = np.concatenate([rob_x_pred, rob_y_pred], axis=1)
        return np.ndarray.tolist(array_xy_pred)

    def make_step(self, 
                  state: np.ndarray,
                  observation: Dict[int, np.ndarray],
                  robot_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances = self.predict(observation)
        control = self._mpc.make_step(state).T[0]
        #print(control)
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
