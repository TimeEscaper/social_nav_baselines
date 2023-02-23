import numpy as np
import casadi as cs

from abc import ABC, abstractmethod
from typing import Tuple


class AbstractCasadiCost(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()

    @abstractmethod
    def evaluate_hessians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raise NotImplementedError()


class GoalCasadiCost(AbstractCasadiCost):

    def __init__(self, goal: np.ndarray, Q: np.ndarray, state_dim: int):
        super().__init__()
        x_r_dim = Q.shape[0]
        x = cs.SX.sym("x", state_dim)
        l_f = (x[:x_r_dim] - goal).T @ Q @ (x[:x_r_dim] - goal)
        l_f_x = cs.jacobian(l_f, x)
        l_f_xx = cs.jacobian(l_f_x, x)

        self._l_f = cs.Function("cost_final",
                                {"x": x, "l_f": l_f},
                                ["x"], ["l_f"])
        self._l_f_x = cs.Function("cost_final_x",
                                  {"x": x, "l_f_x": l_f_x},
                                  ["x"], ["l_f_x"])
        self._l_f_xx = cs.Function("cost_final_xx",
                                   {"x": x, "l_f_xx": l_f_xx},
                                   ["x"], ["l_f_xx"])

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._l_f(x).toarray().reshape(1)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        l_f_x = self._l_f_x(x).toarray().reshape(x.shape[0])
        l_f_u = np.zeros_like(u)
        return l_f_x, l_f_u

    def evaluate_hessians(self, x: np.ndarray, u: np.ndarray,
                          step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        l_f_xx = self._l_f_xx(x).toarray()
        l_f_xu = np.zeros((x.shape[0], u.shape[0]))
        l_f_ux = np.zeros((u.shape[0], x.shape[0]))
        l_f_uu = np.zeros((u.shape[0], u.shape[0]))
        return l_f_xx, l_f_xu, l_f_ux, l_f_uu


class ControlCasadiCost(AbstractCasadiCost):

    def __init__(self, R_diag: np.ndarray):
        super().__init__()
        R = np.diag(R_diag)
        u = cs.SX.sym("u", R_diag.shape[0])
        l_s = u.T @ R @ u
        l_s_u = cs.jacobian(l_s, u)
        l_s_uu = cs.jacobian(l_s_u, u)

        self._l_s = cs.Function("cost_stage",
                                {"u": u, "l_s": l_s},
                                ["u"], ["l_s"])
        self._l_s_u = cs.Function("cost_stage_u",
                                  {"u": u, "l_s_u": l_s_u},
                                  ["u"], ["l_s_u"])
        self._l_s_uu = cs.Function("cost_stage_uu",
                                   {"u": u, "l_s_uu": l_s_uu},
                                   ["u"], ["l_s_uu"])

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._l_s(u).toarray().reshape(1)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        l_s_x = np.zeros_like(x)
        l_s_u = self._l_s_u(u).toarray().reshape(u.shape[0])
        return l_s_x, l_s_u

    def evaluate_hessians(self, x: np.ndarray, u: np.ndarray,
                          step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        l_s_xx = np.zeros((x.shape[0], x.shape[0]))
        l_s_xu = np.zeros((x.shape[0], u.shape[0]))
        l_s_ux = np.zeros((u.shape[0], x.shape[0]))
        l_s_uu = self._l_s_uu(u).toarray()
        return l_s_xx, l_s_xu, l_s_ux, l_s_uu


class ControlBarrierMahalanobisCasadiCost(AbstractCasadiCost):

    def __init__(self, R: np.ndarray, q: float, w: float, state_dim: int,
                 ped_poses: np.ndarray, ped_covs: np.ndarray):
        super().__init__()
        self._ped_poses = ped_poses.copy()
        self._ped_covs_inv = np.linalg.inv(ped_covs)
        u = cs.SX.sym("u", R.shape[0])
        x = cs.SX.sym("x", state_dim)

        n_peds = ped_poses.shape[1]
        self._n_peds = n_peds
        x_r = x[:2]
        peds = [cs.SX.sym(f"p_{i}", 2) for i in range(n_peds)]
        peds_covs_inv = [cs.SX.sym(f"p_cov_inv_{i}", 2, 2) for i in range(n_peds)]

        l_base = u.T @ R @ u + x[-1] * q * x[-1]

        l_m = (x_r - peds[0]).T @ peds_covs_inv[0] @ (x_r - peds[0])
        for i in range(1, n_peds):
            l_m = l_m + (x_r - peds[i]).T @ peds_covs_inv[i] @ (x_r - peds[i])
        l_s = l_base - w * cs.log(l_m)

        l_s_x = cs.jacobian(l_s, x)
        l_s_u = cs.jacobian(l_s, u)
        l_s_xx = cs.jacobian(l_s_x, x)
        l_s_xu = cs.jacobian(l_s_x, u)
        l_s_ux = cs.jacobian(l_s_u, x)
        l_s_uu = cs.jacobian(l_s_u, u)

        var_dict = {"x": x, "u": u}
        var_dict.update({f"p_{i}": peds[i] for i in range(n_peds)})
        var_dict.update({f"p_cov_inv_{i}": peds_covs_inv[i] for i in range(n_peds)})
        input_vars = ["x", "u"] + [f"p_{i}" for i in range(n_peds)] + [f"p_cov_inv_{i}" for i in range(n_peds)]

        var_dict_l_s = var_dict.copy()
        var_dict_l_s.update({"l_s": l_s})
        self._l_s = cs.Function("stage_cost",
                                var_dict_l_s,
                                input_vars, ["l_s"])

        var_dict_l_s_x = var_dict.copy()
        var_dict_l_s_x.update({"l_s_x": l_s_x})
        self._l_s_x = cs.Function("stage_cost_x",
                                  var_dict_l_s_x,
                                  input_vars, ["l_s_x"])

        var_dict_l_s_u = var_dict.copy()
        var_dict_l_s_u.update({"l_s_u": l_s_u})
        self._l_s_u = cs.Function("stage_cost_u",
                                  var_dict_l_s_u,
                                  input_vars, ["l_s_u"])

        var_dict_l_s_xx = var_dict.copy()
        var_dict_l_s_xx.update({"l_s_xx": l_s_xx})
        self._l_s_xx = cs.Function("stage_cost_xx",
                                   var_dict_l_s_xx,
                                   input_vars, ["l_s_xx"])

        var_dict_l_s_xu = var_dict.copy()
        var_dict_l_s_xu.update({"l_s_xu": l_s_xu})
        self._l_s_xu = cs.Function("stage_cost_xu",
                                   var_dict_l_s_xu,
                                   input_vars, ["l_s_xu"])

        var_dict_l_s_ux = var_dict.copy()
        var_dict_l_s_ux.update({"l_s_ux": l_s_ux})
        self._l_s_ux = cs.Function("stage_cost_ux",
                                   var_dict_l_s_ux,
                                   input_vars, ["l_s_ux"])

        var_dict_l_s_uu = var_dict.copy()
        var_dict_l_s_uu.update({"l_s_uu": l_s_uu})
        self._l_s_uu = cs.Function("stage_cost_uu",
                                   var_dict_l_s_uu,
                                   input_vars, ["l_s_uu"])

    def __call__(self, x: np.ndarray, u: np.ndarray, step: int) -> np.ndarray:
        args = self._get_args(x, u, step)
        return self._l_s(*args).toarray().reshape(1)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        args = self._get_args(x, u, step)
        l_s_x = self._l_s_x(*args).toarray().reshape(x.shape[0])
        l_s_u = self._l_s_u(*args).toarray().reshape(u.shape[0])
        return l_s_x, l_s_u

    def evaluate_hessians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        args = self._get_args(x, u, step)
        l_s_xx = self._l_s_xx(*args).toarray()
        l_s_xu = self._l_s_xu(*args).toarray()
        l_s_ux = self._l_s_ux(*args).toarray()
        l_s_uu = self._l_s_uu(*args).toarray()
        return l_s_xx, l_s_xu, l_s_ux, l_s_uu

    def _get_args(self, x: np.ndarray, u: np.ndarray, step: int):
        args = [x, u] + [self._ped_poses[step, i, :] for i in range(self._n_peds)] + \
               [self._ped_covs_inv[step, i, :, :] for i in range(self._n_peds)]
        return args
