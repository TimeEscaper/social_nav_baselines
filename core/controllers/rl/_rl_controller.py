import gym
import numpy as np
import torch
import gym
import pkg_resources
from pyminisim.core import ROBOT_RADIUS, PEDESTRIAN_RADIUS
from pyminisim.util import wrap_angle

from stable_baselines3 import PPO

from typing import List, Optional, Dict, Union
from core.controllers._controller import AbstractController
from core.predictors import PedestrianTracker


def unnormalize_symmetric(action: np.ndarray,
                          lb: np.ndarray,
                          ub: np.ndarray) -> np.ndarray:
    deviation = (ub - lb) / 2.
    shift = (ub + lb) / 2.
    action = (action * deviation) + shift
    return action


class RLController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 pedestrian_tracker: PedestrianTracker,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float],
                 total_peds: int,
                 horizon: int,
                 weights: str = "ppo_model_v2",
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
        self._model_id = weights

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
        goal = self.goal[:2]

        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        robot_vel = rotation_matrix @ robot_vel[:2]

        d_g = np.linalg.norm(robot_pose[:2] - goal)
        phi_goal = wrap_angle(robot_pose[2] - np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0]))
        robot_obs = np.array([d_g, phi_goal, robot_vel[0], robot_vel[1], ROBOT_RADIUS])

        peds_obs = np.tile(np.array([-10., -10., 0., 0., 0., 0., 100, ROBOT_RADIUS]), (self._peds_padding, 1))
        peds_vis = np.zeros(self._peds_padding, dtype=np.bool)
        detections = self._ped_tracker.get_current_poses(return_velocities=True)
        for i, (k, v) in enumerate(detections.items()):
            ped_pose = v[:2]
            ped_vel = v[2:]
            ped_pose = ped_pose - robot_pose[:2]
            ped_pose = rotation_matrix @ ped_pose
            ped_phi = np.arctan2(ped_pose[1], ped_pose[0])
            ped_vel = rotation_matrix @ ped_vel
            d = np.linalg.norm(ped_pose)
            peds_obs[i] = np.array([ped_pose[0], ped_pose[1], ped_phi, ped_vel[0], ped_vel[1], PEDESTRIAN_RADIUS,
                                    d, PEDESTRIAN_RADIUS + ROBOT_RADIUS])
            peds_vis[i] = True

        return {
            "robot": robot_obs[np.newaxis],
            "peds": peds_obs[np.newaxis],
            "visibility": peds_vis[np.newaxis]
        }

    def _build_model_obs_v3(self, robot_pose: np.ndarray, robot_vel: np.ndarray):
        goal = self.goal[:2]

        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        robot_vel = rotation_matrix @ robot_vel[:2]
        d_g = np.linalg.norm(robot_pose[:2] - goal)
        phi_goal = wrap_angle(robot_pose[2] - np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0]))
        obs_robot = np.array([d_g, phi_goal, robot_vel[0], robot_vel[1], ROBOT_RADIUS])

        current_poses = self._ped_tracker.get_current_poses()
        obs_pred_mean = np.tile(np.array([-10., -10.]), (self._peds_padding, 6, 1))
        obs_pred_cov = np.tile(np.eye(2) * 0.001, (self._peds_padding, 5, 1, 1))
        peds_vis = np.zeros(self._peds_padding, dtype=np.bool)

        for i, (k, v) in enumerate(self._ped_tracker.get_predictions().items()):
            obs_pred_mean[i, :, :] = np.concatenate((current_poses[k][np.newaxis], v[0][:5]), axis=0)
            obs_pred_cov[i] = v[1][:5]
            peds_vis[i] = True

        obs_pred_mean = np.einsum("ij,mnj->mni", rotation_matrix, obs_pred_mean)
        obs_pred_cov = np.einsum("ij,mnjk->mnik", rotation_matrix, obs_pred_cov)
        obs_pred_cov = np.einsum("mnij,jk->mnik", obs_pred_cov, rotation_matrix.T)

        return {
            "robot": obs_robot[np.newaxis],
            "pred_mean_rl": obs_pred_mean[np.newaxis],
            "pred_cov_rl": obs_pred_cov[np.newaxis],
            "visibility": peds_vis[np.newaxis],
        }

    def _build_model_obs_v4(self, robot_pose: np.ndarray, robot_vel: np.ndarray):
        goal = self.goal[:2]

        rotation_matrix = np.array([[np.cos(robot_pose[2]), -np.sin(robot_pose[2])],
                                    [np.sin(robot_pose[2]), np.cos(robot_pose[2])]])
        robot_vel = rotation_matrix @ robot_vel[:2]
        d_g = np.linalg.norm(robot_pose[:2] - goal)
        phi_goal = wrap_angle(robot_pose[2] - np.arctan2(goal[1] - robot_pose[1], goal[0] - robot_pose[0]))
        obs_robot = np.array([d_g, phi_goal, robot_vel[0], robot_vel[1], ROBOT_RADIUS])

        current_poses = self._ped_tracker.get_current_poses()
        pred_mean = np.tile(np.array([-10., -10.]), (self._peds_padding, 8, 1))
        pred_cov = np.tile(np.eye(2) * 0.001, (self._peds_padding, 7, 1, 1))
        pred_vis = np.zeros(self._peds_padding, dtype=np.bool)

        for i, (k, v) in enumerate(self._ped_tracker.get_predictions().items()):
            pred_mean[i, :, :] = np.concatenate((current_poses[k][np.newaxis], v[0][:7]), axis=0) - robot_pose[:2]
            pred_cov[i] = v[1][:7]
            pred_vis[i] = True

        if pred_vis.any():
            pred_mean[pred_vis] = np.einsum("ij,mnj->mni", rotation_matrix, pred_mean[pred_vis])
            pred_cov[pred_vis] = np.einsum("ij,mnjk->mnik", rotation_matrix, pred_cov[pred_vis])
            pred_cov[pred_vis] = np.einsum("mnij,jk->mnik", pred_cov[pred_vis], rotation_matrix.T)

        return {
            "robot": obs_robot[np.newaxis],
            "pred_mean_rl": pred_mean[np.newaxis],
            "pred_cov_rl": pred_cov[np.newaxis],
            "visibility": pred_vis[np.newaxis],
        }


    def make_step(self, state: np.ndarray, observation: Dict[int, np.ndarray], robot_velocity: np.ndarray) -> np.ndarray:
        self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances = self._get_output_predictions()
        if self._model_id == "ppo_model_v3":
            obs = self._build_model_obs_v3(state, robot_velocity)
        elif self._model_id == "ppo_model_v4":
            obs = self._build_model_obs_v4(state, robot_velocity)
        else:
            obs = self._build_model_obs(state, robot_velocity)

        with torch.no_grad():
            actions, states = self._rl_model.predict(obs, self._states, deterministic=True)
            actions = actions[0]
            actions = unnormalize_symmetric(actions,
                                            lb=np.array([0., -2.]),
                                            ub=np.array([2., 2.]))
            self._states = states

        return actions, self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances

    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        super().set_new_goal(current_state, new_goal)
        self._states = None

    def get_predicted_robot_trajectory(self):
        return np.array([[10000, 10000]])
