from ._controller import AbstractController
from core.predictors import AbstractPredictor
import numpy as np
from typing import List

class DWAController(AbstractController):
    
    def __init__(self, 
                 init_state: np.ndarray, 
                 goal: np.ndarray,  
                 dt: float,
                 horizon: int,
                 predictor: AbstractPredictor,
                 cost_function: str,
                 v_res: float,
                 w_res: float,
                 lb: List[float],
                 ub: List[float],
                 weights: List[float],
                 total_peds: int,
                 state_dummy_ped: List[float],
                 max_ghost_tracking_time: int) -> None:
        assert cost_function == "original" or cost_function == "euclidean", f"Provide correct name for a cost function: ['original', 'euclidean'], given: {cost_function}"
        super().__init__(init_state, 
                         goal, 
                         horizon, 
                         dt,
                         state_dummy_ped,
                         max_ghost_tracking_time)
        self._cost_function = cost_function
        self._predictor = predictor
        self._weights = weights
        self._v_range, self._w_range = self.get_dynamic_window(lb, 
                                                               ub, 
                                                               v_res, 
                                                               w_res)
        self._alpha, self._beta, self._gamma = weights
        self._ghost_tracking_times : List[int]= [0] * total_peds

    def get_dynamic_window(self, 
                           lb: List[float],
                           ub: List[float],
                           v_res: float,
                           w_res: float) -> np.ndarray:
        v_range = np.arange(lb[0], ub[0], v_res)
        if 0 not in v_range: # To be sure that 0 velocity is the possible velocity
            np.append(v_range, 0)
        w_range = np.arange(lb[1], ub[1], w_res)
        if 0 not in w_range: # To be sure that 0 angular velocity is the possible velocity
            np.append(w_range, 0)
        return v_range, w_range

    def _calculate_robot_trajectory(self,
                                    state: np.ndarray,
                                    v: float,
                                    w: float) -> np.ndarray:
        """Get robot trajectory along prediction horizon with selected pair of velocities

        Args:
            X_rob_cur (_type_): current robot state vector
            v (_type_): selected velocity v
            w (_type_): selected velocity w

        Returns:
            np.ndarray: predicted robot trajectory
        """
        predicted_robot_trajectory = np.zeros([self.horizon, 3])
        predicted_robot_trajectory[0, :] = state
        for i in range(1, self.horizon):
            new_x = predicted_robot_trajectory[i - 1, 0] + v * \
                np.cos(predicted_robot_trajectory[i - 1, 2]) * self.dt
            new_y = predicted_robot_trajectory[i - 1, 1] + v * \
                np.sin(predicted_robot_trajectory[i - 1, 2]) * self.dt
            new_phi = predicted_robot_trajectory[i - 1, 2] + w * self.dt
            new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi
            predicted_robot_trajectory[i, :] = np.array([new_x, new_y, new_phi])
        return predicted_robot_trajectory

    def get_predicted_robot_trajectory(self) -> List[float]:
        return np.ndarray.tolist(self._predicted_robot_trajectory[:, :2])

    def make_step(self, 
                  state: np.ndarray,
                  ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        
        min_cost = np.inf
        control = [0, 0]
        self._predicted_pedestrians_trajectories, _ = self._predictor.predict(ground_truth_pedestrians_state)
        for v in self._v_range:
            for w in self._w_range:
                predicted_robot_trajectory = self._calculate_robot_trajectory(state,
                                                                              v,
                                                                              w)
                if self._cost_function == "original":
                    cost = self._alpha * self._weighted_heading_to_goal(predicted_robot_trajectory) + \
                        self._beta * self._weighted_distance_to_pedestrians(predicted_robot_trajectory,
                                                                            self._predicted_pedestrians_trajectories) + \
                        self._gamma * (max(self._v_range) - v)

                elif self._cost_function == "euclidean":
                    cost = self._alpha * self._weighted_distance_to_goal(predicted_robot_trajectory) + \
                        self._beta * self._weighted_distance_to_pedestrians(predicted_robot_trajectory,
                                                                            self._predicted_pedestrians_trajectories)
                if cost < min_cost:
                    min_cost = cost
                    control = [v, w]
                    self._predicted_robot_trajectory = predicted_robot_trajectory
        return control, self._predicted_pedestrians_trajectories

    def _weighted_heading_to_goal(self,
                                  predicted_robot_trajectory: np.ndarray) -> float:
        sum_cost_heading = 0
        for pred_rob_pos in predicted_robot_trajectory:
            dx = self.goal[0] - pred_rob_pos[0]
            dy = self.goal[1] - pred_rob_pos[1]
            error_angle = np.arctan2(dy, dx)
            overall_deviation_angle = error_angle - pred_rob_pos[2]
            overall_deviation_angle = (
                overall_deviation_angle + np.pi) % (2 * np.pi) - np.pi
            cost = abs(overall_deviation_angle) / np.pi
            sum_cost_heading += cost
        return sum_cost_heading / self.horizon

    def _weighted_distance_to_pedestrians(self,
                                          predicted_robot_trajectory: np.ndarray,
                                          predicted_pedestrians_trajectories: np.ndarray) -> float:
        """Get normalized distance among each pedestrian at each timestep

        Args:
            predicted_robot_trajectory (np.ndarray): predicted robot trajectory with selected pair of velocities
            predicted_pedestrians_trajectories (np.ndarray): predicted pedestrian trajectories

        Returns:
            float: normalized distance to all the pedestrians at each time step: [0, 1]
        """
        sum_cost_dist = 0
        for step in range(self.horizon):
            ox = predicted_pedestrians_trajectories[step, :, 0]
            oy = predicted_pedestrians_trajectories[step, :, 1]
            dx = predicted_robot_trajectory[step, 0] - ox[:, None]
            dy = predicted_robot_trajectory[step, 1] - oy[:, None]
            r = np.hypot(dx, dy)
            min_r = np.min(r)
            cost = 1.0 / min_r
            sum_cost_dist += cost
        normalized_sum = sum_cost_dist / self.horizon
        return normalized_sum

    def _weighted_distance_to_goal(self,
                                   predicted_robot_trajectory: np.ndarray) -> float:
        """Get normalized distance to the goal after prediction propagation

        Args:
            pred_rob_traj (np.ndarray): predicted robot trajectory with selected pair of velocities

        Returns:
            float: normalized distance: [0, 1]
        """
        dx_0 = predicted_robot_trajectory[0, 0] - self.goal[0]
        dy_0 = predicted_robot_trajectory[0, 1] - self.goal[1]
        dist_at_beginning = np.sqrt(dx_0 ** 2 + dy_0 ** 2)

        dx_f = predicted_robot_trajectory[-1, 0] - self.goal[0]
        dy_f = predicted_robot_trajectory[-1, 1] - self.goal[1]
        dist_after_pred = np.sqrt(dx_f ** 2 + dy_f ** 2)

        normalized_dist = dist_after_pred / dist_at_beginning

        return normalized_dist