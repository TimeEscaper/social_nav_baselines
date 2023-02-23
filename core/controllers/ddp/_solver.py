import numpy as np
from typing import Tuple

from ._dynamics import AbstractCasadiDynamics
from ._cost import AbstractCasadiCost


def _ddp_backward(nominal_trajectory_x: np.ndarray,
                  nominal_trajectory_u: np.ndarray,
                  dynamics: AbstractCasadiDynamics,
                  stage_cost: AbstractCasadiCost,
                  final_cost: AbstractCasadiCost) -> Tuple[np.ndarray, np.ndarray]:
    horizon = nominal_trajectory_x.shape[0]

    l_x, l_u = final_cost.evaluate_jacobians(nominal_trajectory_x[horizon - 1],
                                             nominal_trajectory_u[horizon - 1],
                                             horizon - 1)
    l_xx, l_xu, l_ux, l_uu = final_cost.evaluate_hessians(nominal_trajectory_x[horizon - 1],
                                                          nominal_trajectory_u[horizon - 1],
                                                          horizon - 1)
    b = l_x
    A = l_xx

    Ks = []
    js = []

    for i in reversed(range(horizon - 1)):
        l_x, l_u = stage_cost.evaluate_jacobians(nominal_trajectory_x[i],
                                                 nominal_trajectory_u[i], i)
        l_xx, l_xu, l_ux, l_uu = stage_cost.evaluate_hessians(nominal_trajectory_x[i],
                                                              nominal_trajectory_u[i], i)

        f_x, f_u = dynamics.evaluate_jacobians(nominal_trajectory_x[i],
                                               nominal_trajectory_u[i], i)
        # f_xx, f_xu, f_ux, f_uu = self._dynamics.evaluate_hessians(self._current_trajectory_x[i],
        #                                              self._current_trajectory_u[i], i)

        Q_x = l_x + f_x.T @ b
        Q_u = l_u + f_u.T @ b
        Q_xx = l_xx + f_x.T @ A @ f_x  # + (f_xx.T @ b).T  # + np.einsum("i,ijk->jk", b, f_xx)
        Q_uu = l_uu + f_u.T @ A @ f_u  # + (f_uu.T @ b).T # + np.einsum("i,ijk->jk", b, f_uu)
        Q_ux = l_ux + f_u.T @ A @ f_x  # + (f_ux.T @ b).T  #+ np.einsum("i,ijk->jk", b, f_ux)

        Q_uu_inv = np.linalg.inv(Q_uu)
        K = -Q_uu_inv @ Q_ux
        j = -Q_uu_inv @ Q_u

        A = Q_xx + K.T @ Q_uu @ K + Q_ux.T @ K + K.T @ Q_ux
        b = Q_x + K.T @ Q_uu @ j + Q_ux.T @ j + K.T @ Q_u

        Ks.append(K)
        js.append(j)

    Ks = np.stack(Ks[::-1], axis=0)
    js = np.stack(js[::-1], axis=0)

    return Ks, js


def _ddp_forward(nominal_trajectory_x: np.ndarray,
                 nominal_trajectory_u: np.ndarray,
                 dynamics: AbstractCasadiDynamics,
                 Ks: np.ndarray,
                 js: np.ndarray,
                 alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    new_traj_x = []
    new_traj_u = []
    x_k_new = nominal_trajectory_x[0]
    horizon = nominal_trajectory_x.shape[0]

    for i in range(horizon - 1):
        K = Ks[i]
        j = js[i]
        x_k = nominal_trajectory_x[i]
        u_k = nominal_trajectory_u[i]

        u_k_new = u_k + K @ (x_k_new - x_k) + alpha * j

        new_traj_x.append(x_k_new)
        new_traj_u.append(u_k_new)
        x_k_new = dynamics(x_k_new, u_k_new, i)

    new_traj_x.append(x_k_new)
    new_traj_u.append(np.zeros_like(nominal_trajectory_u[0]))

    new_traj_x = np.array(new_traj_x)
    new_traj_u = np.array(new_traj_u)

    return new_traj_x, new_traj_u


def ddp_solve(nominal_trajectory_x: np.ndarray,
              nominal_trajectory_u: np.ndarray,
              dynamics: AbstractCasadiDynamics,
              stage_cost: AbstractCasadiCost,
              final_cost: AbstractCasadiCost,
              n_iterations: int,
              alpha: float = 1.) -> Tuple[np.ndarray, np.ndarray]:
    assert len(nominal_trajectory_x.shape) == 2, "Nominal state trajectory shape must be (horizon, state_dim)"
    assert len(nominal_trajectory_u.shape) == 2, "Nominal control trajectory shape must be (horizon, control_dim)"
    assert nominal_trajectory_x.shape[0] == nominal_trajectory_u.shape[0], \
        "State and control trajectories must have same lenght"
    assert 0. <= alpha <= 1., "alpha must be 0 <= alpha <= 1"
    assert n_iterations > 0, "Number of iterations must be greater than 0"

    optimized_traj_x = nominal_trajectory_x
    optimized_traj_u = nominal_trajectory_u
    for _ in range(n_iterations):
        Ks, js = _ddp_backward(optimized_traj_x,
                               optimized_traj_u,
                               dynamics,
                               stage_cost,
                               final_cost)
        optimized_traj_x, optimized_traj_u = _ddp_forward(optimized_traj_x,
                                                          optimized_traj_u,
                                                          dynamics,
                                                          Ks,
                                                          js,
                                                          alpha)
    return optimized_traj_x, optimized_traj_u
