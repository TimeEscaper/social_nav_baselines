from pyminisim.core import Simulation
from typing import List
import numpy as np

class Statistics():
    def __init__(self,
                 simulator: Simulation,
                 scene_config_path: str,
                 controller_config_path: str) -> None:
        self._simulator = simulator
        self._simulation_ticks: int = 0
        self._total_collisions : int = 0
        self._previous_step_collision_indices = simulator._get_world_state().robot_to_pedestrians_collisions
        self._current_step_collision_indices = simulator._get_world_state().robot_to_pedestrians_collisions
        self._failure = False
        self._scene_config_path = scene_config_path
        self._controller_config_path = controller_config_path
        self._collisions_per_subgoall_array: List[int] = []
        # trajectories
        self._ground_truth_pedestrian_trajectories: List[np.ndarray] = []
        self._ground_truth_robot_trajectory: List[np.ndarray] = []
        self._predicted_pedestrians_trajectories: List[np.ndarray] = []
        self._predicted_robot_trajectory: List[np.ndarray] = []
        self._predicted_pedestrians_covariances: List = []
        self._subgoals_trajectory = []

    @property
    def ground_truth_pedestrian_trajectories(self) -> np.ndarray:
        return np.asarray(self._ground_truth_pedestrian_trajectories)
    
    @property
    def ground_truth_robot_trajectory(self) -> np.ndarray:
        return np.asarray(self._ground_truth_robot_trajectory)
    
    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return np.asarray(self._predicted_pedestrians_trajectories)

    @property
    def predicted_pedestrians_covariances(self) -> np.ndarray:
        return np.asarray(self._predicted_pedestrians_covariances)

    @property
    def predicted_robot_trajectory(self) -> np.ndarray:
        return np.asarray(self._predicted_robot_trajectory)

    @property
    def subgoals_trajectory(self) -> np.ndarray:
        return np.asarray(self._subgoals_trajectory)

    def append_current_subgoal(self,
                               current_subgoal: np.ndarray) -> None:
        self._subgoals_trajectory.append(current_subgoal.tolist())

    def set_failure_flag(self) -> None:
        self._failure = True

    def track_collisions(self) -> None:
        self._current_step_collision_indices = self._simulator._get_world_state().robot_to_pedestrians_collisions
        delta = len(set(self._previous_step_collision_indices) & (set(self._previous_step_collision_indices) - set(self._current_step_collision_indices)))
        if delta > 0:
            self._total_collisions += delta
        self._previous_step_collision_indices = self._current_step_collision_indices

    def track_simulation_ticks(self):
        self._simulation_ticks += 1

    def append_collision_per_subgoal(self):
        if len(self._collisions_per_subgoall_array) == 0:
            self._collisions_per_subgoall_array.append(self._total_collisions)
        else:
            self._collisions_per_subgoall_array.append(self._total_collisions - self._collisions_per_subgoall_array[-1])
    
    @property
    def failure(self) -> int:
        return self._failure

    @property
    def simulation_ticks(self) -> int:
        return self._simulation_ticks
    
    @property
    def total_collisions(self) -> int:
        return self._total_collisions

    @property
    def scene_config_path(self) -> str:
        return self._scene_config_path
    
    @property
    def controller_config_path(self) -> str:
        return self._controller_config_path
    
    @property
    def collisions_per_subgoall_array(self) -> List[int]:
        return self._collisions_per_subgoall_array