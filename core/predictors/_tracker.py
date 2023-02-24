import numpy as np
from typing import Dict, Optional


class PedestrianTracker:

    _STATE_DIM = 4  # x, y, v_x, v_y

    def __init__(self,
                 track_history_length: int,
                 return_dummy: bool = True):
        self._track_history_length = track_history_length
        self._return_dummy = return_dummy
        self._tracks = {}

    def update(self, observation: Dict[int, np.ndarray]):
        tracked_peds = set(observation.keys())
        observed_peds = set(observation.keys())

        updated_peds = tracked_peds.intersection(observed_peds)
        for k in updated_peds:
            history = self._tracks[k]
            new_history = np.zeros_like(history)
            new_history[:-1, :] = history[1:, :]
            new_history[-1, :] = observation
            self._tracks[k] = new_history

        new_peds = observed_peds.difference(tracked_peds)
        for k in new_peds:
            self._tracks[k] = np.tile(observation, (self._track_history_length, 1))

        lost_peds = tracked_peds.difference(observed_peds)
        for k in lost_peds:
            self._tracks[k] = None

    @property
    def joint_track(self) -> Optional[np.ndarray]:
        # Return: (history_length, n_pedestrians, state_dim)
        if len(self._tracks) != 0:
            return np.concatenate([v for v in self._tracks.values()], axis=1)
        return np.tile(np.array([1000., 1000., 0., 0.]), (self._track_history_length, 1, 1))

    @property
    def track_dict(self) -> Dict[int, np.ndarray]:
        return {k: v.copy() for k, v in self._tracks.copy()}



