import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from core.planners import AbstractPlanner
from core.controllers import AbstractController
from core.predictors import PedestrianTracker


class RandomPlanner(AbstractPlanner):

    def __init__(self,
                 global_goal: np.ndarray,
                 controller: AbstractController,
                 subgoal_reach_threshold: float,
                 subgoal_to_goal_threshold: float,
                 pedestrian_tracker: PedestrianTracker):
        super(RandomPlanner, self).__init__(global_goal=global_goal,
                                            controller=controller,
                                            subgoal_reach_threshold=subgoal_reach_threshold,
                                            subgoal_to_goal_threshold=subgoal_to_goal_threshold)
        self._pedestrian_tracker = pedestrian_tracker

    def generate_subgoal(self, state: np.ndarray, observation: Any) -> np.ndarray:
        return self._global_goal - 1

    def make_step(self, state: np.ndarray, observation: Any) -> Tuple[np.ndarray, Any]:
        self._pedestrian_tracker.update(observation)
        new_subgoal, global_goal_reached = self.update_subgoal(state, observation)
        if new_subgoal:
            self._controller.set_new_goal(state, self._current_subgoal)
        return self._controller.make_step(state, observation)
