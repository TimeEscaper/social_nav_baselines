import json
import numpy as np
import torch

from typing import Optional, List
from torch.utils.data import IterableDataset
from tqdm import tqdm


class SubgoalEvalDataset(IterableDataset):

    def __init__(self,
                 method_name: str,
                 trajectory_path: str,
                 statistics_path: str,
                 trials: Optional[List[int]] = None):
        super(SubgoalEvalDataset, self).__init__()

        print("Loading trajectory...")
        with open(trajectory_path) as f:
            traj_data = json.load(f)
        print("Loaded!")

        print("Loading stats...")
        with open(statistics_path) as f:
            stat_data = json.load(f)
        print("Loaded!")

        self._traj_data = traj_data
        self._stat_data = stat_data
        self._method_name = method_name
        self._trials = [str(trial) for trial in trials] if trials is not None else None

    def __iter__(self):
        traj_data = self._traj_data
        stat_data = self._stat_data
        method_name = self._method_name

        for scene_name, scene_data in traj_data.items():
            for run_n, run_data in scene_data[method_name]["7"].items():
                if self._trials is not None:
                    if run_n not in self._trials:
                        continue
                n_steps = len(run_data["subgoals_trajectory"])
                previous_subgoal = None
                subgoal_cnt = -1
                for i in range(n_steps):
                    subgoal = np.array(run_data["subgoals_trajectory"][i])[:2]
                    if previous_subgoal is not None and np.allclose(subgoal, previous_subgoal):
                        continue
                    previous_subgoal = subgoal
                    subgoal_cnt += 1

                    preds = np.array(run_data["predicted_pedestrians_trajectories"])[i].transpose(1, 0, 2)
                    preds_covs = np.array(run_data["predicted_pedestrians_covariances"])[i].transpose(1, 0, 2, 3)
                    indicies = []
                    for j in range(preds.shape[0]):
                        if preds[j, 0, 0] <= 1000:
                            indicies.append(j)
                    if len(indicies) != 0:
                        preds = preds[indicies, :, :]
                        preds_covs = preds_covs[indicies, :, :, :]
                    else:
                        continue

                    hist_len = 4
                    ped_history = np.tile(run_data["ground_truth_pedestrian_trajectories"][i], (hist_len, 1, 1))
                    robot_history = np.tile(run_data["ground_truth_robot_trajectory"][i][:2], (hist_len, 1))
                    for j in range(1, hist_len):
                        if i - j < 0:
                            break
                        ped_history[hist_len - 1 - j, :, :] = np.array(run_data["ground_truth_pedestrian_trajectories"][i - j])
                        robot_history[hist_len - 1 - j, :] = np.array(
                            run_data["ground_truth_robot_trajectory"][i - j][:2])
                    ped_history = ped_history[:, indicies, :]
                    ped_history = ped_history.transpose(1, 0, 2)

                    if subgoal_cnt >= len(stat_data[scene_name][method_name]["7"][run_n]["collisions_per_subgoal"]):
                        continue
                    collision = stat_data[scene_name][method_name]["7"][run_n]["collisions_per_subgoal"][subgoal_cnt]
                    collision = float(min(1, collision))
                    collision = torch.Tensor([collision])

                    item = {
                        "subgoal": torch.Tensor(subgoal),
                        "robot": torch.Tensor(robot_history),
                        "peds": torch.Tensor(ped_history),
                        "peds_pred": torch.Tensor(preds),
                        "peds_covs": torch.Tensor(preds_covs)
                    }
                    yield item, collision
