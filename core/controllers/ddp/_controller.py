import numpy as np
import do_mpc
import casadi
import core.controllers.ddp as ddp

from typing import List, Tuple, Dict, Any
from core.predictors import AbstractPredictor, NeuralPredictor, ConstantVelocityPredictor
from core.controllers import AbstractController


class DDPController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float,
                 model_type: str,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 q_b: float,
                 q_m: float,
                 r_rob: float,
                 r_ped: float,
                 constraint_value: float,
                 n_iterations: int,
                 predictor: AbstractPredictor,
                 is_store_robot_predicted_trajectory: bool,
                 max_ghost_tracking_time: int,
                 state_dummy_ped,
                 alpha: float = 1.) -> None:
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
                         state_dummy_ped,
                         max_ghost_tracking_time)
        self._predictor = predictor
        self._is_store_robot_predicted_trajectory = is_store_robot_predicted_trajectory
        self._ghost_tracking_times: List[int] = [0] * total_peds

        if model_type == "unicycle":
            self._state_dim = 4
        elif model_type == "double_integrator":
            self._state_dim = 5
        else:
            raise ValueError(f"Unknown model type {model_type}")
        self._model_type = model_type

        self._Q = Q.copy()
        self._R = R.copy()
        self._q_b = q_b
        self._q_m = q_m
        self._barrier_dist = constraint_value + r_ped + r_rob
        self._n_iterations = n_iterations
        self._alpha = alpha

        self._planned_robot_traj = np.zeros((horizon, 2))

    def get_predicted_robot_trajectory(self) -> List[float]:
        return self._planned_robot_traj.tolist()

    def make_step(self,
                  state: np.ndarray,
                  ground_truth_pedestrians_state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        predicted_peds_traj, predicted_peds_covs = self._predictor.predict(ground_truth_pedestrians_state)

        if self._model_type == "unicycle":
            dynamics = ddp.UnicycleBarrierDynamics(self._dt,
                                                   predicted_peds_traj[:, :, :2],
                                                   self._barrier_dist)
        elif self._model_type == "double_integrator":
            dynamics = ddp.DoubleIntegratorBarrierDynamics(self._dt,
                                                           predicted_peds_traj[:, :, :2],
                                                           self._barrier_dist)
        else:
            raise ValueError()

        stage_cost = ddp.ControlBarrierMahalanobisCasadiCost(R=self._R,
                                                             q=self._q_b,
                                                             w=self._q_m,
                                                             state_dim=self._state_dim,
                                                             ped_poses=predicted_peds_traj[:, :, :2],
                                                             ped_covs=predicted_peds_covs)
        final_cost = ddp.GoalCasadiCost(goal=self._goal,
                                        Q=self._Q,
                                        state_dim=self._state_dim)

        x_nominal = np.tile(np.array([state[0], state[1], state[2], 0.]), (self._horizon, 1))
        u_nominal = np.zeros((self._horizon, 2))
        for i in range(self._horizon):
            x_nominal[i, -1] = dynamics.eval_barrier(x_nominal[i, :-1], i)

        x_opt, u_opt = ddp.ddp_solve(x_nominal,
                                     u_nominal,
                                     dynamics,
                                     stage_cost,
                                     final_cost,
                                     self._n_iterations,
                                     alpha=self._alpha)

        self._planned_robot_traj = x_opt[:, :2]
        control = u_opt[0]

        print(u_opt)
        self._predicted_pedestrians_trajectories = predicted_peds_traj

        return control, predicted_peds_traj, predicted_peds_covs
        # return control, {"ped_poses": predicted_peds_traj,
        #                  "ped_covs": predicted_peds_covs}

    @property
    def predictor(self) -> AbstractPredictor:
        return self._predictor

    @property
    def ghost_tracking_times(self) -> List[int]:
        return self._ghost_tracking_times

    @property
    def max_ghost_tracking_time(self) -> int:
        return self._max_ghost_tracking_time

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories
