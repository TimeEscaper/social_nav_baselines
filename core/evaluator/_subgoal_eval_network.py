import torch
import torch.nn as nn

from typing import List


class SubgoalEvalNetwork(nn.Module):

    def __init__(self,
                 point_embedding_dim: int = 32,
                 covariance_embedding_dim: int = 32,
                 rnn_hidden_dim: int = 256,
                 fusion_dim: int = 512):
        super(SubgoalEvalNetwork, self).__init__()

        self._subgoal_embedding = nn.Linear(2, point_embedding_dim)
        self._robot_embedding = nn.Linear(2, point_embedding_dim)
        self._pedestrian_previous_trajectory_embedding = nn.Linear(2, point_embedding_dim)
        self._pedestrian_predicted_trajectory_embedding = nn.Linear(2, point_embedding_dim)
        self._pedestrian_predicted_covariance_embedding = nn.Linear(3, covariance_embedding_dim)

        self._robot_trajectory_encoder = nn.LSTM(input_size=point_embedding_dim,
                                                 hidden_size=rnn_hidden_dim,
                                                 batch_first=True)
        self._pedestrian_previous_trajectory_encoder = nn.LSTM(input_size=point_embedding_dim,
                                                      hidden_size=rnn_hidden_dim,
                                                      batch_first=True)
        self._pedestrian_predicted_trajectory_encoder = nn.LSTM(input_size=point_embedding_dim,
                                                                hidden_size=rnn_hidden_dim,
                                                                batch_first=True)
        self._pedestrian_predicted_cov_encoder = nn.LSTM(input_size=covariance_embedding_dim,
                                                         hidden_size=rnn_hidden_dim,
                                                         batch_first=True)

        joint_dim = rnn_hidden_dim * 4 + point_embedding_dim

        self._fusion_layers = nn.Sequential(nn.Linear(joint_dim, fusion_dim),
                                            nn.ReLU(),
                                            nn.LayerNorm(fusion_dim),
                                            nn.Linear(fusion_dim, fusion_dim // 2),
                                            nn.ReLU(),
                                            nn.LayerNorm(fusion_dim // 2),
                                            nn.Linear(fusion_dim // 2, 1),
                                            nn.Sigmoid())

    def forward(self,
                subgoal: torch.Tensor,
                previous_robot_trajectory: torch.Tensor,
                previous_pedestrians_trajectories: torch.Tensor,
                predicted_pedestrians_trajectories: torch.Tensor,
                predicted_pedestrians_covs: torch.Tensor) -> torch.Tensor:
        # subgoal: (bs, 2)
        # previous_robot_trajectory: (bs, hist_len, 2)
        # previous_pedestrians_trajectories: [(bs, hist_len, 2)], len = n_previous_agents
        # predicted_pedestrians_trajectories: [(bs, pred_len, 2)], len = n_observed_agents
        # predicted_pedestrians_covs: [(bs, pred_len, 2)], len = n_observed_agents
        batch_size = subgoal.shape[0]
        n_previous_agents = previous_pedestrians_trajectories.shape[1]
        n_predicted_agents = predicted_pedestrians_trajectories.shape[1]
        hist_len = previous_robot_trajectory.shape[1]
        pred_len = predicted_pedestrians_trajectories.shape[2]

        previous_pedestrians_trajectories = previous_pedestrians_trajectories.reshape(n_previous_agents * batch_size, hist_len, 2)

        predicted_pedestrians_trajectories = predicted_pedestrians_trajectories.reshape(n_predicted_agents * batch_size, pred_len, 2)
        predicted_pedestrians_covs = predicted_pedestrians_covs.reshape(n_predicted_agents * batch_size, pred_len, 2, 2)
        predicted_pedestrians_covs = predicted_pedestrians_covs.reshape(n_predicted_agents * batch_size, pred_len, 4)[:, :, :3]

        subgoal_emb = self._subgoal_embedding.forward(subgoal)
        robot_emb = self._robot_embedding.forward(previous_robot_trajectory)
        ped_hist_emb = self._pedestrian_previous_trajectory_embedding.forward(previous_pedestrians_trajectories)
        ped_pred_emb = self._pedestrian_predicted_trajectory_embedding.forward(predicted_pedestrians_trajectories)
        ped_pred_cov_emb = self._pedestrian_predicted_covariance_embedding.forward(predicted_pedestrians_covs)

        robot_enc = self._robot_trajectory_encoder.forward(robot_emb)[0][:, -1, :]
        ped_hist_enc = self._pedestrian_previous_trajectory_encoder.forward(ped_hist_emb)[0][:, -1, :]
        ped_pred_enc = self._pedestrian_predicted_trajectory_encoder.forward(ped_pred_emb)[0][:, -1, :]
        ped_pred_cov_enc = self._pedestrian_predicted_cov_encoder.forward(ped_pred_cov_emb)[0][:, -1, :]

        ped_hist_enc = ped_hist_enc.reshape(batch_size, n_previous_agents, -1).sum(dim=1)
        ped_pred_enc = ped_pred_enc.reshape(batch_size, n_predicted_agents, -1).sum(dim=1)
        ped_pred_cov_enc = ped_pred_cov_enc.reshape(batch_size, n_predicted_agents, -1).sum(dim=1)

        joint_feature = torch.cat([subgoal_emb, robot_enc, ped_hist_enc, ped_pred_enc, ped_pred_cov_enc], dim=1)

        score = self._fusion_layers.forward(joint_feature)

        return score
