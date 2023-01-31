from ._controller import AbstractController
from planners.core.predictors import AbstractPredictor
import numpy as np
import do_mpc
import casadi
from typing import List, Set


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
                 lb: List[float],
                 ub: List[float],
                 r_rob: float,
                 r_ped: float,
                 min_safe_dist: float,
                 predictor: AbstractPredictor,
                 is_store_robot_predicted_trajectory: bool,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float]) -> None:
        """Initiazition of the class instance

        Args:
            init_state (np.ndarray): Initial state of the system
            goal (np.ndarray): Goal of the system
            horizon (int): Prediction horizon of the controller, [steps]
            dt (float): Time delta, [s]
            model_type (str): Type of the model, ["unicycle", "unicycle double integrator"]
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
                         dt)
        self._predictor = predictor
        self._is_store_robot_predicted_trajectory = is_store_robot_predicted_trajectory
        self._previously_detected_pedestrians: Set[int] = set()
        self._ghost_tracking_times : List[int]= [0] * total_peds
        self._max_ghost_tracking_time = max_ghost_tracking_time
        self._state_dummy_ped = state_dummy_ped

        # Architecture requires at least one dummy pedestrian in the system
        if total_peds == 0:
            total_peds = 1

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
        elif model_type == "unicycle double integrator":
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
        # goal
        current_goal = self._model.set_variable(
            "_tvp", "current_goal", shape=(len(goal)))
        # setup
        self._model.setup()

        # MPC controller
        self._mpc = do_mpc.controller.MPC(self._model)
        # controller setup
        setup_mpc = {
            "n_horizon": horizon,
            "t_step": dt,
            "store_full_solution": is_store_robot_predicted_trajectory,
            "nlpsol_opts": {"ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": 0}
        }
        self._mpc.set_param(**setup_mpc)
        # stage cost
        u = self._model._u.cat
        stage_cost = u.T @ R @ u
        # terminal cost
        p_rob_N = self._model._x.cat[:3]
        terminal_cost = (p_rob_N - current_goal).T @ Q @ (p_rob_N - current_goal)

        # set cost
        self._mpc.set_objective(lterm=stage_cost,
                                mterm=terminal_cost)
        if model_type == "unicycle":
            self._mpc.set_rterm(v=1e-2, w=1e-2)
        elif model_type == "unicycle double integrator":                      
            self._mpc.set_rterm(u_a=1e-2, u_alpha=1e-2)
        # bounds
        if model_type == "unicycle":
            # lb
            self._mpc.bounds['lower', '_u', 'v'] = lb[0]
            self._mpc.bounds['lower', '_u', 'w'] = lb[1]
            # ub
            self._mpc.bounds['upper', '_u', 'v'] = ub[0]
            self._mpc.bounds['upper', '_u', 'w'] = ub[1]
        elif model_type == "unicycle double integrator":
            # lb
            self._mpc.bounds["lower", "_x", "v"] = lb[0]
            self._mpc.bounds["lower", "_x", "w"] = lb[1]
            self._mpc.bounds["lower", "_u", "u_a"] = lb[2]
            self._mpc.bounds["lower", "_u", "u_alpha"] = lb[3]
            # ub
            self._mpc.bounds["upper", "_x", "v"] = ub[0]
            self._mpc.bounds["upper", "_x", "w"] = ub[1]
            self._mpc.bounds["upper", "_u", "u_a"] = ub[2]
            self._mpc.bounds["upper", "_u", "u_alpha"] = ub[3]
        # distance to pedestrians
        lb_dist_square = (r_rob + r_ped + min_safe_dist) ** 2
        lb_dists_square = np.array([lb_dist_square for _ in range(total_peds)])
        # inequality constrain for pedestrians
        # Horizontally concatenated array of robot current position for all the pedestrians
        p_rob_hcat = casadi.hcat([self._model._x.cat[:2]
                                 for _ in range(total_peds)])
        p_peds = self._state_peds[:2, :]
        dx_dy_square = (p_rob_hcat - p_peds) ** 2
        peds_dists_square = dx_dy_square[0, :] + dx_dy_square[1, :]
        self._mpc.set_nl_cons("dist_to_peds", -peds_dists_square,
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

    def get_ref_direction(self,
                          state: np.ndarray,
                          goal: np.ndarray) -> np.ndarray:
        """Get direction angle of reference position

        This helps MPC to find solution when the reference point is located behind the robot.

        Args:
            state (np.ndarray): initial robot state vector (for more details see config file)
            goal (np.ndarray): reference robot position (for more details see config file)

        Returns:
            np.ndarray: goal (np.ndarray): reference robot position with a changed direction
        """
        x_goal_polar = goal[0] - state[0]
        y_goal_polar = goal[1] - state[1]
        return np.array([goal[0], goal[1], np.arctan2(y_goal_polar, x_goal_polar)])

    def predict(self,
                ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor
        
        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        predicted_pedestrians_trajectories: np.ndarray = self._predictor.predict(ground_truth_pedestrians_state,
                                                                                 self.dt,
                                                                                 self.horizon)
        for step in range(len(self._mpc_tvp_fun['_tvp', :, 'p_peds'])):
            self._mpc_tvp_fun['_tvp', step, 'p_peds'] = predicted_pedestrians_trajectories[step].T
        return predicted_pedestrians_trajectories

    def get_predicted_robot_trajectory(self) -> List[float]:
        rob_x_pred = self._mpc.data.prediction(('_x', 'x'))[0]
        rob_y_pred = self._mpc.data.prediction(('_x', 'y'))[0]
        array_xy_pred = np.concatenate([rob_x_pred, rob_y_pred], axis=1)
        return np.ndarray.tolist(array_xy_pred)

    def make_step(self, 
                  state: np.ndarray,
                  ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        self._predicted_pedestrians_trajectories = self.predict(ground_truth_pedestrians_state)
        control = self._mpc.make_step(state).T[0]
        return control, self._predicted_pedestrians_trajectories

    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        new_goal = self.get_ref_direction(current_state,
                                          new_goal)
        self._mpc_tvp_fun['_tvp', :, 'current_goal'] = new_goal
        self.goal = new_goal

    def get_pedestrains_ghosts_states(self,
                                      ground_truth_pedestrians_state: np.ndarray,
                                      undetected_pedestrian_indices: List[int]) -> np.ndarray:
        """Methods returns states for the pedestrians taking into account the ghosts.

        Ghost is a pedestrian that was previously seen and dissepeared from the vision field.
        This method helps the controller to take into account stored trajectories of the ghosts

        Args:
            ground_truth_pedestrians_state (np.ndarray): This is the structure of ground truth states. Here every ghost has a dummy pedestrian state.
            undetected_pedestrian_indices (List[int]): Structure to keep undetected pedestrians at this time step.

        Returns:
            np.ndarray: Structure with ground truth states of detected pedestrians, memorized states of the missed but previously seen pedestrians and dummy states for previously not seen pedestrians.
        """
        pedestrains_ghosts_states = np.copy(ground_truth_pedestrians_state)
        for undetected_pedestrian in undetected_pedestrian_indices:
            if undetected_pedestrian in self._previously_detected_pedestrians:
                pedestrains_ghosts_states[undetected_pedestrian] = self._predicted_pedestrians_trajectories[1, undetected_pedestrian]
                self._ghost_tracking_times[undetected_pedestrian] += 1
                if self._ghost_tracking_times[undetected_pedestrian] > self._max_ghost_tracking_time:
                    self._ghost_tracking_times[undetected_pedestrian] = 0
                    pedestrains_ghosts_states[undetected_pedestrian] = self._state_dummy_ped
                    self._previously_detected_pedestrians.remove(undetected_pedestrian)
        return pedestrains_ghosts_states

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
    def previously_detected_pedestrians(self) -> Set[int]:
        return self._previously_detected_pedestrians

    @previously_detected_pedestrians.setter
    def previously_detected_pedestrians(self,
                                        new_detected_pedestrians: Set[int]) -> Set[int]:
        self._previously_detected_pedestrians = new_detected_pedestrians
    
    @property
    def max_ghost_tracking_time(self) -> int:
        return self._max_ghost_tracking_time

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories