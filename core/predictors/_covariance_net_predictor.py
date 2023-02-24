import numpy as np
import torch

from typing import Optional, Tuple

from ._predictor import AbstractPredictor
from ._covariance_net._model import CovarianceNet
from ._covariance_net._model_utils import get_rotation_translation_matrix


class NeuralPredictor(AbstractPredictor):
    # Network was trained with those parameters
    _MODEL_HORIZON = 25
    _DT = 0.1
    _HISTORY_LENGTH = 8

    def __init__(self,
                 total_peds: int,
                 horizon: int = 25,
                 device: str = "cpu") -> None:
        super().__init__(dt=NeuralPredictor._DT,
                         total_peds=total_peds,
                         horizon=horizon)

        self._model = CovarianceNet(input_size=2,
                                    hidden_size=64,
                                    prediction_steps=NeuralPredictor._MODEL_HORIZON)
        self._model.load_state_dict(torch.load("/home/sibirsky/model_6_2023-02-14_12-08-44.pth",
                                               map_location=device))
        _ = self._model.to(device)
        self._model = self._model.eval()
        self._device = device

    def predict(self, joint_history: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # joint_history: (history, n_neighbours, state_dim)
        n_agents = joint_history.shape[1]
        if n_agents == 1:
            ego_agent = joint_history[:, 0, :2]
            ego_vel = joint_history[-1, 0, :]
            neighbours_stub = np.ones((joint_history.shape[0], 1, 2)) * 1000.
            pred, cov = self._predict_ego_agent(ego_agent, ego_vel, neighbours_stub)
            return pred[np.newaxis], cov[np.newaxis]

        preds = []
        covs = []
        for i in range(n_agents):
            ego_agent = joint_history[:, i, :2]
            ego_vel = joint_history[-1, i, :]
            neighbours = joint_history[:, [j for j in range(n_agents) if j != i], :2]
            pred, cov = self._predict_ego_agent(ego_agent, ego_vel, neighbours)
            preds.append(pred)
            covs.append(pred)
        preds = np.array(preds)
        covs = np.array(covs)
        return preds, covs

        # for i in range(joint_history.shape[1]):
        #     ego_history =

    def predict_one_step(self) -> np.ndarray:
        return None

    def _predict_ego_agent(self,
                           ego_history: np.ndarray,
                           ego_vel: np.ndarray,
                           neighbours_history: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        # ego_history: (history, pose_dim)
        # ego_vel: (velocity_dim,)
        # neighbours_history: (history, n_neighbours, pose_dim)
        rt_matrix = get_rotation_translation_matrix(ego_history[7, 0], ego_history[7, 1],
                                                    ego_vel[0], ego_vel[1])

        ego_history = np.matmul(ego_history + rt_matrix[:2, 2], rt_matrix[:2, :2].T)
        neighbours_history = np.matmul(neighbours_history + rt_matrix[:2, 2], rt_matrix[:2, :2].T)

        with torch.no_grad():
            pred, cov = self._model(torch.Tensor(ego_history).unsqueeze(0).to(self._device),
                                    ego_vel[np.newaxis],
                                    torch.Tensor(neighbours_history).unsqueeze(0).to(self._device))
            pred = pred.clone().detach().cpu().numpy()[0]
            cov = cov.clone().detach().cpu().numpy()[0]
        pred = np.matmul(pred[:, :2], np.linalg.inv(rt_matrix[:2, :2].T)) - rt_matrix[:2, 2]

        return pred, cov
