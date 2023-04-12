import numpy as np

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional
from core.controllers import AbstractController
from core.statistics import Statistics


class AbstractPlanner(ABC):

    def __init__(self,
                 global_goal: np.ndarray,
                 controller: AbstractController,
                 subgoal_reach_threshold: float,
                 subgoal_to_goal_threshold: float,
                 statistics_module: Optional[Statistics] = None,
                 max_steps_to_subgoal: Optional[int] = None):
        self._global_goal = global_goal
        self._controller = controller
        self._subgoal_reach_threshold = subgoal_reach_threshold
        self._subgoal_to_goal_threshold = subgoal_to_goal_threshold
        self._statistics_module = statistics_module

        self._current_subgoal = None
        self._subgoals_history = []

        if max_steps_to_subgoal is not None and max_steps_to_subgoal == 0:
            max_steps_to_subgoal = None
        self._max_steps_to_subgoal = max_steps_to_subgoal
        self._subgoal_step_cnt = 0

    @abstractmethod
    def generate_subgoal(self, state: np.ndarray, observation: Any,
                         robot_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def make_step(self, state: np.ndarray, observation: Any) -> Tuple[np.ndarray, Any]:
        raise NotImplementedError()

    @property
    def global_goal(self) -> np.ndarray:
        return self._global_goal

    @property
    def current_subgoal(self) -> Optional[np.ndarray]:
        return self._current_subgoal

    @property
    def subgoals_history(self) -> np.ndarray:
        return np.array(self._subgoals_history)

    @property
    def controller(self) -> AbstractController:
        return self._controller

    def update_subgoal(self, state: np.ndarray, observation: Any,
                       robot_velocity: Optional[np.ndarray] = None) -> Tuple[bool, bool]:
        position = state[:2]
        if np.linalg.norm(position - self._global_goal[:2]) <= self._subgoal_reach_threshold:
            if self._statistics_module:
                self._statistics_module.append_collision_per_subgoal()
            return False, True

        subgoal_updated = False
        goal_reached = False

        self._subgoal_step_cnt += 1

        if self._current_subgoal is None:
            self._current_subgoal = self.generate_subgoal(state, observation, robot_velocity)
            self._subgoals_history.append(self._current_subgoal)
            subgoal_updated = True
        else:
            if np.linalg.norm(position - self._current_subgoal[:2]) <= self._subgoal_reach_threshold:
                if np.linalg.norm(position - self._global_goal[:2]) <= self._subgoal_to_goal_threshold:
                    self._current_subgoal = self._global_goal.copy()
                else:
                    self._current_subgoal = self.generate_subgoal(state, observation, robot_velocity)
                self._subgoals_history.append(self._current_subgoal)
                subgoal_updated = True
                if self._statistics_module:
                    self._statistics_module.append_collision_per_subgoal()

            else:
                if self._max_steps_to_subgoal is not None and self._subgoal_step_cnt >= self._max_steps_to_subgoal:
                    self._current_subgoal = self.generate_subgoal(state, observation, robot_velocity)
                    subgoal_updated = True

        if subgoal_updated:
            self._subgoal_step_cnt = 0

        return subgoal_updated, goal_reached

    def set_new_global_goal(self, goal: np.ndarray):
        self._global_goal = goal.copy()
        self._current_subgoal = None
        self._subgoals_history = []
        self._subgoal_step_cnt = 0

    # if self._current_subgoal is None or np.linalg.norm(self._current_subgoal - position) <= self._subgoal_reach_threshold:
        #     if self._current_subgoal is not None:
        #