import numpy as np
from typing import Dict, Optional, Tuple

from ._predictor import AbstractPredictor


_STATE_DIM = 4


class _ShiftBuffer:

    def __init__(self, size: int, dims: Tuple[int, ...]):
        self._size = size
        self._dims = dims
        self._buffer = np.zeros((size,) + dims)

    def put(self, value: np.ndarray):
        self._buffer[:-1, :] = self._buffer[1:, :]
        self._buffer[-1, :] = value

    def fill(self, value: np.ndarray):
        assert value.shape == self._dims
        self._buffer = np.tile(value, (self._size, 1))

    @property
    def all(self) -> np.ndarray:
        return self._buffer

    @all.setter
    def all(self, value: np.ndarray):
        assert value.shape == (self._size,) + self._dims
        self._buffer = value

    @property
    def first(self) -> np.ndarray:
        return self._buffer[0]

    @property
    def last(self) -> np.ndarray:
        return self._buffer[-1]


class _PedestrianTrack:

    _STATE_DIM = 4  # x, y, v_x, v_y
    _POSE_DIM = 2  # x, y

    def __init__(self, history_length: int, prediction_length: int,
                 initial_state: np.ndarray):
        self._states = _ShiftBuffer(history_length, (_PedestrianTrack._STATE_DIM,))
        self._predicted_poses = np.zeros((prediction_length, _PedestrianTrack._POSE_DIM))
        self._predicted_covs = np.zeros((prediction_length, _PedestrianTrack._POSE_DIM, _PedestrianTrack._POSE_DIM))

        if len(initial_state.shape) == 1:
            self._states.fill(initial_state)
        else:
            self._states.all = initial_state

    def update_states(self, state: np.ndarray):
        self._states.put(state)

    def update_predictions(self, pred_poses: np.ndarray, pred_covs: np.ndarray):
        self._predicted_poses = pred_poses
        self._predicted_covs = pred_covs

    @property
    def history(self) -> np.ndarray:
        return self._states.all

    @property
    def current_pose(self) -> np.ndarray:
        return self._states.last

    @property
    def predicted_poses(self) -> np.ndarray:
        return self._predicted_poses

    @property
    def predicted_covs(self) -> np.ndarray:
        return self._predicted_covs


class _GhostTrack:

    def __init__(self,
                 states: np.ndarray,
                 predicted_poses: np.ndarray,
                 predicted_covs: np.ndarray):
        self._states = _ShiftBuffer(states.shape[0], states.shape[1:])
        self._states.all = states
        self._predicted_poses = _ShiftBuffer(predicted_poses.shape[0], predicted_poses.shape[1:])
        self._predicted_poses.all = predicted_poses
        self._predicted_covs = _ShiftBuffer(predicted_covs.shape[0], predicted_covs.shape[1:])
        self._predicted_covs.all = predicted_covs
        self._tracking_time = 0
        self._last_vel = self._states.last[2:]

    def update(self):
        self._states.put(np.concatenate((self._predicted_poses.first, self._last_vel)))
        self._predicted_poses.put(self._predicted_poses.last)
        self._predicted_covs.put(self._predicted_covs.last)
        self._tracking_time += 1

    @property
    def tracking_time(self) -> int:
        return self._tracking_time

    @property
    def history(self) -> np.ndarray:
        return self._states.all

    @property
    def current_pose(self) -> np.ndarray:
        return self._states.last

    @property
    def predicted_poses(self) -> np.ndarray:
        return self._predicted_poses.all

    @property
    def predicted_covs(self) -> np.ndarray:
        return self._predicted_covs.all


class PedestrianTracker:

    _STATE_DIM = 4  # x, y, v_x, v_y

    def __init__(self,
                 predictor: AbstractPredictor,
                 max_ghost_tracking_time: int,
                 return_dummy: bool = True):
        self._predictor = predictor
        self._track_history_length = predictor.history_length
        self._max_ghost_tracking_time = max_ghost_tracking_time
        self._return_dummy = return_dummy

        self._tracks: Dict[int, _PedestrianTrack] = {}
        self._ghosts: Dict[int, _GhostTrack] = {}

        # # Actively tracked pedestrians for which we do predictions
        # # Item: (history_len, state_dim)
        # self._tracks = {}
        #
        # # Pedestrians that were not observed, but we continue tracking them in a background
        # # Item: tuple of poses (history_len, state_dim), covariances (history_len, 2, 2) and tracking times
        # self._ghosts = {}
        #
        # self._track_predictions = {}
        # self._ghost_predictions = {}
        # self._ghosts_times = {}

    def update(self, observation: Dict[int, np.ndarray]):
        tracked_peds = set(self._tracks.keys())
        observed_peds = set(observation.keys())

        # Pedestrians that were tracked and present observation - just continue tracking them
        updated_peds = tracked_peds.intersection(observed_peds)
        for k in updated_peds:
            self._tracks[k].update_states(observation[k])

        # Pedestrians that are observed, but not present in current tracks - they are "returned ghosts" or new pedestrians
        new_peds = observed_peds.difference(tracked_peds)
        for k in new_peds:
            if k in self._ghosts:
                # Pedestrians is a ghost that was "pseudo-tracked"
                pedestrian = _PedestrianTrack(self._track_history_length,
                                              self._predictor.horizon,
                                              self._ghosts[k].history)
                pedestrian.update_states(observation[k])
                self._tracks[k] = pedestrian
                self._ghosts.pop(k)
            else:
                # Pedestrian was simply not observed
                self._tracks[k] = _PedestrianTrack(self._track_history_length,
                                                   self._predictor.horizon,
                                                   observation[k])

        # Pedestrians that were tracked but not observed now - they become "ghosts"
        lost_peds = tracked_peds.difference(observed_peds)
        if self._max_ghost_tracking_time != 0:
            for k in lost_peds:
                pedestrian = self._tracks[k]
                ghost = _GhostTrack(pedestrian.history,
                                    pedestrian.predicted_poses,
                                    pedestrian.predicted_covs)
                self._ghosts[k] = ghost
                self._tracks.pop(k)

        # Finally, do the predictions
        self._do_predictions_for_tracks()
        self._update_ghosts()

    def get_predictions(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        preds = {}
        preds.update({k: (v.predicted_poses.copy(), v.predicted_covs.copy()) for k, v in self._tracks.items()})
        preds.update({k: (v.predicted_poses.copy(), v.predicted_covs.copy()) for k, v in self._ghosts.items()})
        return preds

    def get_current_poses(self) -> Dict[int, np.ndarray]:
        result = {}
        result.update({k: v.current_pose[:2] for k, v in self._tracks.items()})
        result.update({k: v.current_pose[:2] for k, v in self._ghosts.items()})
        return result

    @property
    def joint_track(self) -> Optional[np.ndarray]:
        # Return: (history_length, n_pedestrians, state_dim)
        if len(self._tracks) != 0:
            return np.concatenate([v.history[:, np.newaxis, :] for v in self._tracks.values()], axis=1)
        if self._return_dummy:
            return np.tile(np.array([1000., 1000., 0., 0.]), (self._track_history_length, 1, 1))
        return None

    @property
    def track_dict(self) -> Dict[int, np.ndarray]:
        return {k: v.history.copy() for k, v in self._tracks.items()}

    def _do_predictions_for_tracks(self):
        if len(self._tracks) == 0:
            return {}

        joint_history = np.concatenate([v.history[:, np.newaxis, :] for v in self._tracks.values()], axis=1)
        pred_poses, pred_covs = self._predictor.predict(joint_history)
        ped_ids = list(self._tracks.keys())
        for i in range(len(self._tracks)):
            ped_id = ped_ids[i]
            self._tracks[ped_id].update_predictions(pred_poses[:, i, :], pred_covs[:, i, :])

    def _update_ghosts(self):
        for k in set(self._ghosts.keys()):
            ghost = self._ghosts[k]
            if ghost.tracking_time > self._max_ghost_tracking_time:
                self._ghosts.pop(k)
            else:
                ghost.update()
