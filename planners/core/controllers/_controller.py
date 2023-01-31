from abc import ABC, abstractmethod
import numpy as np


class AbstractController(ABC):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float) -> None:
        assert len(goal) == 3, f"Goal should be provided with a vector [x, y, theta] and has a length of 3, provided length is {len(goal)}"
        self._init_state = init_state
        self._goal = goal
        self._horizon = horizon
        self._dt = dt

    @abstractmethod
    def make_step(self, state: np.ndarray) -> np.ndarray:
        """Method calculates control input for the specified system

        Args:
            state (np.ndarray): State of the system

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
