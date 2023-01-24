from abc import ABC, abstractmethod
from typing import Optional

import numpy as np


class AbstractController(ABC):

    @abstractmethod
    def make_step(self) -> np.ndarray:
        """
        
        """
        raise NotImplementedError()