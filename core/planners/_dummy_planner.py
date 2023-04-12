import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List
from core.planners import AbstractPlanner
from core.controllers import AbstractController
from core.predictors import PedestrianTracker
from core.statistics import Statistics

class DummyPlanner(AbstractPlanner):

    def __init__(self,
                 global_goal: np.ndarray,
                 controller: AbstractController,
                 subgoal_reach_threshold: float,
                 subgoal_to_goal_threshold: float,
                 pedestrian_tracker: PedestrianTracker,
                 dist_bounds: Tuple[float, float] = (1.2, 2.),
                 angle_bounds: Tuple[float, float] = (np.deg2rad(-32.), np.deg2rad(32.)),
                 statistics_module: Optional[Statistics] = None):
        super(DummyPlanner, self).__init__(global_goal=global_goal,
                                            controller=controller,
                                            subgoal_reach_threshold=subgoal_reach_threshold,
                                            subgoal_to_goal_threshold=subgoal_to_goal_threshold,
                                            statistics_module=statistics_module)
        self._pedestrian_tracker = pedestrian_tracker
        self._dist_bounds = dist_bounds
        self._angle_bounds = angle_bounds

    def generate_subgoal(self, state: np.ndarray, observation: Any, robot_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        return self.global_goal[:2]

    def make_step(self, state: np.ndarray, observation: Any, robot_velocity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        self._pedestrian_tracker.update(observation)
        new_subgoal, global_goal_reached = self.update_subgoal(state, observation, robot_velocity)
        if new_subgoal:
            self._controller.set_new_goal(state, self._current_subgoal)
        return self._controller.make_step(state, observation, robot_velocity)
