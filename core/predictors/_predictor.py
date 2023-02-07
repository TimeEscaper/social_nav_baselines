from abc import ABC, abstractmethod
import numpy as np

class AbstractPredictor(ABC):
    
    def __init__(self,
                 dt: float,
                 total_peds: int,
                 horizon: int) -> None:
        assert dt > 0, f"Time delta should be > 0, given {dt}"
        assert total_peds > 0, f"Amount of pedestrians should be positive, given {total_peds}"
        assert horizon > 0, f"Prediction horizon should be positive, given {horizon}"
        self._dt = dt
        self._total_peds = total_peds
        self._horizon = horizon
    

    @abstractmethod
    def predict_one_step(self)-> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def predict(self)-> np.ndarray: 
        raise NotImplementedError()

    @property
    def dt(self) -> float:
        return self._dt
    
    @property
    def total_peds(self) -> float:
        return self._total_peds