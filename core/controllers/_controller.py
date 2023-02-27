from abc import ABC, abstractmethod
import numpy as np
from typing import Set, List


class AbstractController(ABC):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float,
                 state_dummy_ped: List[float],
                 max_ghost_tracking_time: int) -> None:
        assert len(goal) == 3, f"Goal should be provided with a vector [x, y, theta] and has a length of 3, provided length is {len(goal)}"
        self._init_state = init_state
        self._goal = goal
        self._horizon = horizon
        self._dt = dt
        self._previously_detected_pedestrians: Set[int] = set()
        self._max_ghost_tracking_time = max_ghost_tracking_time
        self._state_dummy_ped = state_dummy_ped

    @abstractmethod
    def make_step(self, 
                  state: np.ndarray,
                  ground_truth_pedestrians_state: np.ndarray) -> np.ndarray:
        """Method calculates control input for the specified system

        Args:
            state (np.ndarray): State of the system
            propagation_ped_traj (np.ndarray): Predicted pedestrian trajectories [prediction step, pedestrian, [x, y, vx, vy]]
            
        Raises:
            NotImplementedError: Abstract method was not implemented

        Returns:
            np.ndarray: Control input
        """
        raise NotImplementedError()

    @property
    def init_state(self) -> np.ndarray:
        return self._init_state

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self,
             new_goal: np.ndarray) -> np.ndarray:
        self._goal = new_goal

    @property
    def horizon(self) -> np.ndarray:
        return self._horizon

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def previously_detected_pedestrians(self) -> Set[int]:
        return self._previously_detected_pedestrians

    @previously_detected_pedestrians.setter
    def previously_detected_pedestrians(self,
                                        new_detected_pedestrians: Set[int]) -> Set[int]:
        self._previously_detected_pedestrians = new_detected_pedestrians
    
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
    
    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        new_goal = self.get_ref_direction(current_state,
                                          new_goal)
        self.goal = new_goal

