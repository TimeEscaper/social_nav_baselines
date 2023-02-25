import torch
import torch.nn as nn
import numpy as np
from typing import Optional
from ._model_utils import nll_loss, rollout_gru, get_cov_mat
from ._cons_vel_model import batched_constant_velocity_model


class CovarianceNet(nn.Module):
    def __init__(self, input_size, hidden_size, prediction_steps=12):
        super(CovarianceNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.output_size = output_size
        self.dropout = nn.Dropout(0.1)
        # history encoder
        self.history_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.history_fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

        # future encoder
        # self.future_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        # self.future_fc = nn.Linear(hidden_size, output_size)

        # neighbor encoder
        self.n_lin = nn.Sequential(
            nn.Linear(input_size, hidden_size // 2),
            nn.ReLU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.ReLU()
        )
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=4, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.n_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

        # decoder:
        self.predictions_to_hidden = nn.Linear(input_size, hidden_size)
        self.gru = torch.nn.GRUCell(hidden_size, hidden_size)
        self.out_l = torch.nn.Linear(hidden_size, 3)  # stable generation of cov mat
        self.criterion = nll_loss
        self.prediction_steps = prediction_steps

    def forward(self, agent_hist: torch.Tensor, agent_vel: np.ndarray, neighbor_hist: torch.Tensor):
        # agent_hist: (batch_size, seq_len, input_size)
        # neighbor_hist: (batch_size, num_neighbors, seq_len, input_size)
        # agent_future: (batch_size, seq_len, input_size)
        # output: (batch_size, seq_len, output_size)

        # history encoder
        agent_hist_encoded, _ = self.history_lstm(agent_hist)  # bs, times, datalen
        # agent_hist = self.history_fc(agent_hist)
        # agent_hist = self.relu(agent_hist)
        # agent_hist = self.dropout(agent_hist)

        # future encoder
        # if agent_future_gt is not None:
        #     agent_future, _ = self.future_lstm(agent_future_gt)
        # agent_future = self.future_fc(agent_future)
        # agent_future = self.relu(agent_future)
        # agent_future = self.dropout(agent_future)

        # neighbor encoder
        # transformer encoder
        num_neighb = neighbor_hist.shape[1]
        bs = neighbor_hist.shape[0]
        neighbor_hist = self.n_lin(neighbor_hist)  # bs, neighb num, times, datalen
        neighbor_hist = neighbor_hist.view(bs * num_neighb, 8, -1)
        anchor = torch.zeros(neighbor_hist.shape[0], 1, neighbor_hist.shape[2]).to(neighbor_hist.device)
        neighbor_hist = self.transformer_decoder(anchor, neighbor_hist).view(bs, num_neighb, -1)
        neighbor_hist, _ = self.n_lstm(neighbor_hist)

        # if agent_future_gt is None:
        #     scene = torch.cat((agent_hist, neighbor_hist), dim=2)

        # scene = torch.cat((agent_hist[:,-1], neighbor_hist[:,-1]), dim=-1)
        scene = agent_hist_encoded[:, -1] + neighbor_hist[:, -1]
        # decoder_input: (seq_len, batch_size, output_size)
        predictions_xy = batched_constant_velocity_model(agent_hist.cpu().detach().numpy(), agent_vel,
                                                         self.prediction_steps)
        predictions_xy = torch.from_numpy(predictions_xy).to(agent_hist_encoded.device).float()
        predictions = self.predictions_to_hidden(predictions_xy)

        gru_out = rollout_gru(self.gru, scene, predictions, pred_steps=predictions.shape[1])
        gru_out = self.out_l(gru_out)

        covariances = get_cov_mat(gru_out[:, :, :1], gru_out[:, :, 1:2], gru_out[:, :, 2:3], predictions)

        return predictions_xy, covariances

    def loss(self, pred, cov, gt):
        return self.criterion(pred, cov, gt).mean()
