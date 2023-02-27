import numpy as np
import torch

from ._controller import AbstractController
from core.predictors import AbstractPredictor, NeuralPredictor, ConstantVelocityPredictor
from core.predictors import PedestrianTracker
import do_mpc
import casadi
from pytorch_mppi import MPPI
from typing import List, Dict, Optional


class _StageControlCost:

    def __init__(self, R: np.ndarray, device: str):
        self._R = torch.tensor(R).float().to(device)

    def __call__(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return 0.
        # cost = torch.einsum("ni,ji->nj", action, self._R)
        # cost = torch.einsum("ni,ni->n", action, cost)
        # return cost


class _FullTrajectoryCost:

    def __init__(self, Q: float, stage_weight: float, r_robot: float,
                 r_ped: float, device: str):
        self._Q = Q
        self._stage_weight = stage_weight
        self._r_robot = r_robot
        self._r_ped = r_ped
        self._device = device
        self._goal = None
        self._initial_state = None
        self._ped_samples = None

    def update(self, goal: np.ndarray, initial_state: np.ndarray,
               ped_samples: Optional[torch.Tensor]):
        self._goal = torch.tensor(goal).float().to(self._device)
        self._initial_state = torch.tensor(initial_state).float().to(self._device)
        if ped_samples is not None:
            self._ped_samples = torch.tensor(ped_samples).float().to(self._device)
        else:
            self._ped_samples = None

    def __call__(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cost = self._calculate_goal_cost(states)
        cost = cost + self._stage_weight * self._calculate_social_cost(states)
        return cost

    def _calculate_goal_cost(self, states: torch.Tensor) -> torch.Tensor:
        final_state = states[:, :, -1, :]
        initial_dist = torch.linalg.norm(self._initial_state - self._goal)
        final_dist = torch.linalg.norm(states[:, :, -1, :] - self._goal, dim=-1)
        cost = self._Q * (final_dist / initial_dist) ** 2
        return cost

    def _calculate_social_cost(self, states: torch.Tensor) -> torch.Tensor:
        if self._ped_samples is None:
            return 0.
        robot_samples = states[0][:, :, :2]
        peds_poses = self._ped_samples.permute(0, 2, 1, 3)

        n_ped_samples = peds_poses.shape[0]
        n_peds = peds_poses.shape[1]
        distances = torch.zeros((n_ped_samples, n_peds, robot_samples.shape[0], robot_samples.shape[1], robot_samples.shape[2]))
        for i in range(n_ped_samples):
            for j in range(n_peds):
                distances[i, j, :, :, :] = robot_samples - peds_poses[i, j, :, :]
        distances = torch.linalg.norm(distances, dim=-1)

        costs = 1 / distances
        costs = torch.mean(costs, dim=0)
        costs = torch.sum(costs, dim=0)
        costs = torch.sum(costs, dim=1)
        costs = costs.unsqueeze(0)
        return costs


class _UnicycleDynamics:

    def __init__(self, dt: float):
        self._dt = dt

    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        new_state = torch.zeros_like(state)
        new_state[:, 0] = state[:, 0] + action[:, 0] * torch.cos(state[:, 2]) * self._dt
        new_state[:, 1] = state[:, 1] + action[:, 0] * torch.sin(state[:, 2]) * self._dt
        new_state[:, 2] = state[:, 2] + action[:, 1] * self._dt
        return new_state

    def rollout_dynamics(self, start_state: np.ndarray, actions: np.ndarray) -> np.ndarray:
        traj = np.zeros((actions.shape[0], 3))
        for i in range(actions.shape[0]):
            if i != 0:
                previous_state = traj[i - 1]
            else:
                previous_state = start_state
            traj[i, 0] = previous_state[0] + actions[i, 0] * np.cos(previous_state[2]) * self._dt
            traj[i, 1] = previous_state[1] + actions[i, 0] * np.sin(previous_state[2]) * self._dt
            traj[i, 2] = previous_state[2] + actions[i, 1] * self._dt
        return traj


class MPPIController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float,
                 model_type: str,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 W: float,
                 noise_sigma: np.ndarray,
                 lb: List[float],
                 ub: List[float],
                 r_rob: float,
                 r_ped: float,
                 pedestrian_tracker: PedestrianTracker,
                 max_ghost_tracking_time: int,
                 cost_function: str,
                 num_samples: int = 50,
                 lambda_: float = 1.,
                 cost_n_samples: int = 20,
                 device: str = "cpu") -> None:
        """Initiazition of the class instance

        Args:
            init_state (np.ndarray): Initial state of the system
            goal (np.ndarray): Goal of the system
            horizon (int): Prediction horizon of the controller, [steps]
            dt (float): Time delta, [s]
            model_type (str): Type of the model, ["unicycle", "unicycle_double_integrator"]
            total_peds (int): Amount of pedestrians in the system, >= 0
            Q (np.ndarray): State weighted matrix, [state_dim * state_dim]
            R (np.ndarray): Control weighted matrix, [control_dim * control_dim]
            lb (List[float]): Lower boundaries for system states or/and controls
            ub (List[float]): Upper boundaries for system states or/and controls
            r_rob (float): Robot radius, [m]
            r_ped (float): Pedestrian radius, [m]
            constraint_value (float): Minimal safe distance between robot and pedestrian, [m]
            predictor (AbstractPredictor): Predictor, [Constant Velocity Predictor, Neural Predictor]
            # TODO

        Returns:
            _type_: None
        """
        super().__init__(init_state,
                         self.get_ref_direction(init_state,
                                                goal),
                         horizon,
                         dt,
                         [],
                         max_ghost_tracking_time)
        self._ped_tracker = pedestrian_tracker
        self._total_peds = total_peds
        self._device = device
        self._cost_n_samples = cost_n_samples
        self._full_traj_cost = _FullTrajectoryCost(Q=Q,
                                                   stage_weight=W,
                                                   r_robot=r_rob,
                                                   r_ped=r_ped,
                                                   device=device)

        if model_type == "unicycle":
            dynamics = _UnicycleDynamics(dt=dt)
        else:
            raise ValueError("Only unicycle model for now implemented for MPPI")
        self._dynamics = dynamics
        stage_cost = _StageControlCost(R=R, device=device)

        if len(noise_sigma.shape) == 1:
            noise_sigma = np.array([[noise_sigma[0], 0.],
                                    [0., noise_sigma[1]]])
        elif len(noise_sigma.shape) == 2:
            assert noise_sigma[0, 1] == noise_sigma[1, 0], "noise_sigma must be proper covariance matrix"
        else:
            raise ValueError("noise_sigma must be a vector (with sigmas for each action dim) or proper covariancew matrix")

        self._mppi = MPPI(dynamics=dynamics,
                          running_cost=stage_cost,
                          nx=3,
                          noise_sigma=torch.tensor(noise_sigma).float().to(device),
                          num_samples=num_samples,
                          horizon=horizon+1,
                          device=device,
                          terminal_state_cost=self._full_traj_cost,
                          lambda_=lambda_,
                          u_min=torch.tensor(lb, dtype=torch.double, device=device),
                          u_max=torch.tensor(ub, dtype=torch.double, device=device),
                          u_per_command=horizon+1)
        self._planned_traj = np.zeros((horizon+1, 3))

    def predict(self,
                observation: Dict[int, np.ndarray]) -> np.ndarray:
        """Method predicts pedestrian trajectories with specified predictor

        Args:
            ground_truth_pedestrians_state (np.ndarray): Current state of the pedestrians, [2-d numpy array]
        """
        self._ped_tracker.update(observation)
        tracked_predictions = self._ped_tracker.get_predictions()

        predicted_trajectories = np.tile(np.array([1000., 1000., 0., 0.]), (self._horizon + 1, self._total_peds, 1))
        predicted_covs = np.tile(np.array([[0.01, 0.0], [0., 0.01]]), (self._horizon + 1, self._total_peds, 1, 1))
        for k in tracked_predictions.keys():
            predicted_trajectories[:, k, :2] = tracked_predictions[k][0]
            predicted_covs[:, k, :] = tracked_predictions[k][1]

        self._predicted_pedestrians_trajectories = predicted_trajectories
        self._predicted_pedestrians_covariances = predicted_covs

        # For MPPI, number of pedestrians is not a problem unlike for Casadi, so we handle them separately
        if len(tracked_predictions) == 0:
            return None
        tracked_trajs = np.stack([v[0] for v in tracked_predictions.values()], axis=1)
        tracked_covs = np.stack([v[1] for v in tracked_predictions.values()], axis=1)

        tracked_trajs = torch.tensor(tracked_trajs).float().to(self._device)
        tracked_covs = torch.tensor(tracked_covs).float().to(self._device)

        distribution = torch.distributions.MultivariateNormal(loc=tracked_trajs,
                                                              covariance_matrix=tracked_covs)
        ped_samples = distribution.sample((self._cost_n_samples,))

        return ped_samples

    def make_step(self,
                  state: np.ndarray,
                  observation: Dict[int, np.ndarray]) -> np.ndarray:
        ped_samples = self.predict(observation)
        self._full_traj_cost.update(self._goal, state, ped_samples)
        controls = self._mppi.command(state).clone().detach().cpu().numpy()
        self._planned_traj = self._dynamics.rollout_dynamics(state, controls)[:, :2]
        control = controls[0]
        return control, self._predicted_pedestrians_trajectories, self._predicted_pedestrians_covariances

    def set_new_goal(self,
                     current_state: np.ndarray,
                     new_goal: np.ndarray) -> None:
        super().set_new_goal(current_state,
                             new_goal)

    def get_predicted_robot_trajectory(self) -> List[float]:
        return self._planned_traj

    @property
    def init_state(self) -> np.ndarray:
        return self._init_state

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories
