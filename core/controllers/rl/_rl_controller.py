import gym
import numpy as np
import torch
import gym
import pkg_resources

from stable_baselines3 import PPO

from typing import List, Optional, Dict
from core.controllers._controller import AbstractController
from core.predictors import PedestrianTracker


class RLController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 pedestrian_tracker: PedestrianTracker,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float],
                 total_peds: int,
                 horizon: int,
                 weights: str = "ppo_model_v1",
                 device: str = "cuda"):
        super(RLController, self).__init__(init_state=init_state,
                                           goal=goal,
                                           horizon=horizon,
                                           dt=0.1,
                                           max_ghost_tracking_time=max_ghost_tracking_time,
                                           state_dummy_ped=state_dummy_ped)
        self._ped_tracker = pedestrian_tracker
        self._total_peds = total_peds
        self._peds_padding = 8
        self._rl_horizon = 5

        weights = pkg_resources.resource_filename("core.controllers.rl", f"weights/{weights}.zip")
        self._rl_model = PPO.load(weights, device=device)
        self._states = None

    def _get_output_predictions(self) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor

        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        tracked_predictions = self._ped_tracker.get_predictions()
        predicted_trajectories = np.tile(np.array([10000., 10000., 0., 0.]), (self._horizon + 1, self._total_peds, 1))
        predicted_covs = np.tile(np.array([[0.01, 0.0], [0., 0.01]]), (self._horizon + 1, self._total_peds, 1, 1))
        for k in tracked_predictions.keys():
            predicted_trajectories[:, k, :2] = tracked_predictions[k][0]
            predicted_covs[:, k, :] = tracked_predictions[k][1]

        return predicted_trajectories, predicted_covs

    def _build_model_obs(self, robot_pose: np.ndarray, robot_vel: np.ndarray):
        current_poses = self._ped_tracker.get_current_poses()
        predictions = {k: v[0] for k, v in self._ped_tracker.get_predictions().items()}

        obs_ped_traj = np.ones((self._peds_padding, self._rl_horizon + 1, 2)) * 100.
        obs_peds_ids = current_poses.keys()
        obs_peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
        for k in obs_peds_ids:
            obs_ped_traj[k, 0, :] = current_poses[k] - robot_pose[:2]
            obs_ped_traj[k, 1:, :] = predictions[k][:self._rl_horizon, :] - robot_pose[:2]
            obs_peds_vis[k] = True

        # TODO: Should we make soring optional?
        distances = np.linalg.norm(obs_ped_traj[:, 0, :], axis=1)
        sorted_indices = np.argsort(distances)
        obs_ped_traj = obs_ped_traj[sorted_indices]
        obs_peds_vis = obs_peds_vis[sorted_indices]

        robot_obs = np.array([np.linalg.norm(self._goal[:2] - robot_pose[:2]),
                         self._goal[0] - robot_pose[0],
                         self._goal[1] - robot_pose[1],
                         robot_pose[2],
                         robot_vel[0],
                         robot_vel[1],
                         robot_vel[2]]).astype(np.float32)

        return {
            "peds_traj": obs_ped_traj[np.newaxis, :, :, :],
            "peds_visibility": obs_peds_vis[np.newaxis, :],
            "robot_state": robot_obs[np.newaxis, :]
        }

    def make_step(self, state: np.ndarray, observation: Dict[int, np.ndarray], robot_velocity: np.ndarray) -> np.ndarray:
        self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances = self._get_output_predictions()
        obs = self._build_model_obs(state, robot_velocity)

        with torch.no_grad():
            actions, states = self._rl_model.predict(obs, self._states, deterministic=True)
            self._states = states

        return actions[0], self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances

    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        super().set_new_goal(current_state, new_goal)
        self._states = None

    def get_predicted_robot_trajectory(self):
        return np.array([[10000, 10000]])
