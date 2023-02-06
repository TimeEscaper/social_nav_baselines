from ._predictor import AbstractPredictor
import numpy as np

class ConstantVelocityPredictor(AbstractPredictor):

    def __init__(self,
                 dt: float,
                 total_peds: int,
                 horizon: int) -> None:
        super().__init__(dt,
                         total_peds,
                         horizon)

    def predict_one_step(self,
                         state_peds_k: np.ndarray,
                         dt: float) -> np.ndarray:
        """Method returns next state for the specified pedestrians

        Propagation model with constant_velocity

        Args:
            state_peds_k (np.ndarray): state vector of all pedestrians: [x_init,  y_init, vx, vy], [m, m, m/s, m/s]

        Return:
            state_peds_k+1 (np.ndarray): state vector at k + 1
        """
        state_peds_k_1 = state_peds_k[:]
        x = state_peds_k_1[0, :]
        y = state_peds_k_1[1, :]
        vx = state_peds_k_1[2, :]
        vy = state_peds_k_1[3, :]
        state_peds_k_1[0, :] = x + vx * dt
        state_peds_k_1[1, :] = y + vy * dt
        return state_peds_k_1

    def predict(self,
                state_peds_k: np.ndarray,
                dt: float,
                horizon: int) -> np.ndarray:
        """Methods predicts trajectories for all pedestrians across all prediction horizon

        Args:
            state_peds_k (np.ndarray): Current pedestrian states, [tota_peds, state_dim]
            dt (float): Time delta, [s]
            horizon (int): Horizon, [steps]

        Returns:
            np.ndarray: Coordinates of the pedestrians trajectories, [horizon_dim, tota_peds, state_dim]
        """
        states_peds = np.zeros([horizon + 1, *np.shape(state_peds_k)])
        states_peds[0, :, :] = state_peds_k
        # copy velocities
        states_peds[1:, :, 2:] = states_peds[0, :, 2:]
        # propagate coordinates
        for step in range(1, horizon + 1):
            states_peds[step, :, :2] = states_peds[step-1, :, :2] + states_peds[step-1, :, 2:] * dt
        return states_peds