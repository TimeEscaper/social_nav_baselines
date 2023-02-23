import numpy as np
import casadi as cs

from abc import ABC, abstractmethod
from typing import Tuple


class AbstractCasadiDynamics(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        raise NotImplementedError()

    @abstractmethod
    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError()


class DoubleIntegratorCasadiDynamics(AbstractCasadiDynamics):

    def __init__(self, dt: float):
        fo = dt
        so = (dt * dt) / 2
        self._A = np.array([[1., 0., fo, 0.],
                            [0., 1., 0., fo],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.]])
        self._B = np.array([[so, 0.],
                            [0., so],
                            [fo, 0.],
                            [0., fo]])
        x = cs.SX.sym("x", 4)
        u = cs.SX.sym("u", 2)
        f = self._A @ x + self._B @ u
        self._f = cs.Function("f_dyn",
                              {"x": x, "u": u, "f": f},
                              ["x", "u"], ["f"])
        self._f_x = cs.Function("f_dyn_x",
                                {"x": x, "u": u, "f_x": cs.jacobian(f, x)},
                                ["x", "u"], ["f_x"])
        self._f_u = cs.Function("f_dyn_u",
                                {"x": x, "u": u, "f_u": cs.jacobian(f, u)},
                                ["x", "u"], ["f_u"])

    def __call__(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        return self._f(x, u).toarray().reshape(4)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        f_x = self._f_x(x, u).toarray()
        f_u = self._f_u(x, u).toarray()
        return f_x, f_u


class DoubleIntegratorBarrierDynamics(AbstractCasadiDynamics):

    def __init__(self, dt: float, ped_traj: np.ndarray, safe_threshold: float):
        super().__init__()
        self._ped_traj = ped_traj.copy()
        self._safe_threshold = safe_threshold
        n_peds = ped_traj.shape[1]
        self._n_peds = n_peds

        fo = dt
        so = (dt * dt) / 2
        A = np.array([[1., 0., fo, 0.],
                      [0., 1., 0., fo],
                      [0., 0., 1., 0.],
                      [0., 0., 0., 1.]])
        B = np.array([[so, 0.],
                      [0., so],
                      [fo, 0.],
                      [0., fo]])

        x = cs.SX.sym("x", 5)
        u = cs.SX.sym("u", 2)
        x_r = x[:4]

        peds = [cs.SX.sym(f"p_{i}", 2) for i in range(n_peds)]

        x_r_next = A @ x_r + B @ u
        w_next = 1. / ((x_r_next[:2] - peds[0]).T @ (x_r_next[:2] - peds[0]) - safe_threshold ** 2)
        for i in range(1, len(peds)):
            w_next = w_next + 1. / (
                        (x_r_next[:2] - peds[i]).T @ (x_r_next[:2] - peds[i]) - safe_threshold ** 2)
        f = cs.vertcat(x_r_next, w_next)

        var_dict = {"x": x, "u": u}
        var_dict.update({f"p_{i}": peds[i] for i in range(n_peds)})
        input_vars = ["x", "u"] + [f"p_{i}" for i in range(n_peds)]

        var_dict_f = var_dict.copy()
        var_dict_f.update({"f": f})
        self._f = cs.Function("f_dyn",
                              var_dict_f,
                              input_vars, ["f"])

        var_dict_f_x = var_dict.copy()
        var_dict_f_x.update({"f_x": cs.jacobian(f, x)})
        self._f_x = cs.Function("f_dyn_x",
                                var_dict_f_x,
                                input_vars, ["f_x"])

        var_dict_f_u = var_dict.copy()
        var_dict_f_u.update({"f_u": cs.jacobian(f, u)})
        self._f_u = cs.Function("f_dyn_u",
                                var_dict_f_u,
                                input_vars, ["f_u"])

    def __call__(self, x: np.ndarray, u: np.ndarray, step: int) -> np.ndarray:
        args = [x, u] + [self._ped_traj[step, i, :] for i in range(self._n_peds)]
        return self._f(*args).toarray().reshape(5)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        args = [x, u] + [self._ped_traj[step, i, :] for i in range(self._n_peds)]
        f_x = self._f_x(*args).toarray()
        f_u = self._f_u(*args).toarray()
        return f_x, f_u

    def eval_barrier(self, x_r: np.ndarray, step: int) -> float:
        w = 0.
        for i in range(self._n_peds):
            x_p = self._ped_traj[step, i, :]
            diff = x_r[:2] - x_p
            w = w + 1. / ((diff.T @ diff) - self._safe_threshold ** 2)
        return w


class UnicycleDynamics(AbstractCasadiDynamics):

    def __init__(self, dt: float):
        super().__init__()

        x = cs.SX.sym("x", 3)
        u = cs.SX.sym("u", 2)

        f = cs.vertcat(x[0] + u[0] * cs.cos(x[2]) * dt,
                              x[1] + u[0] * cs.sin(x[2]) * dt,
                              x[2] + u[1] * dt)

        f_x = cs.jacobian(f, x)
        f_u = cs.jacobian(f, u)
        f_xx = cs.jacobian(f_x, x)
        f_xu = cs.jacobian(f_x, u)
        f_ux = cs.jacobian(f_u, x)
        f_uu = cs.jacobian(f_u, u)

        self._f = cs.Function("f_dyn",
                              {"x": x, "u": u, "f": f},
                              ["x", "u"], ["f"])
        self._f_x = cs.Function("f_dyn_x",
                                {"x": x, "u": u, "f_x": f_x},
                                ["x", "u"], ["f_x"])
        self._f_u = cs.Function("f_dyn_u",
                                {"x": x, "u": u, "f_u": f_u},
                                ["x", "u"], ["f_u"])
        self._f_xx = cs.Function("f_dyn_xx",
                                {"x": x, "u": u, "f_xx": f_xx},
                                ["x", "u"], ["f_xx"])
        self._f_xu = cs.Function("f_dyn_xu",
                                {"x": x, "u": u, "f_xu": f_xu},
                                ["x", "u"], ["f_xu"])
        self._f_ux = cs.Function("f_dyn_ux",
                                {"x": x, "u": u, "f_ux": f_ux},
                                ["x", "u"], ["f_ux"])
        self._f_uu = cs.Function("f_dyn_uu",
                                {"x": x, "u": u, "f_uu": f_uu},
                                ["x", "u"], ["f_uu"])

        f_x = cs.jacobian(f, x)
        f_xu = cs.jacobian(f_x, u)
        print(f_x.shape)
        print(f_xu)

    def __call__(self, x: np.ndarray, u: np.ndarray, step: int) -> np.ndarray:
        return self._f(x, u).toarray().reshape(3)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        f_x = self._f_x(x, u).toarray()
        f_u = self._f_u(x, u).toarray()
        return f_x, f_u

    def evaluate_hessians(self, x: np.ndarray, u: np.ndarray, step: int) -> \
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f_xx = self._f_xx(x, u).toarray().reshape(3, 3, 3)
        f_xu = self._f_xu(x, u).toarray().reshape(3, 3, 2)
        f_ux = self._f_ux(x, u).toarray().reshape(3, 2, 3)
        f_uu = self._f_uu(x, u).toarray().reshape(3, 2, 2)

        return f_xx, f_xu, f_ux, f_uu


class UnicycleBarrierDynamics(AbstractCasadiDynamics):

    def __init__(self, dt: float, ped_traj: np.ndarray, safe_threshold: float):
        super().__init__()
        self._ped_traj = ped_traj.copy()
        self._safe_threshold = safe_threshold
        n_peds = ped_traj.shape[1]
        self._n_peds = n_peds

        x = cs.SX.sym("x", 4)
        u = cs.SX.sym("u", 2)
        x_r = x[:3]

        peds = [cs.SX.sym(f"p_{i}", 2) for i in range(n_peds)]

        # (theta + np.pi) % (2 * np.pi) - np.pi
        x_r_next = cs.vertcat(x_r[0] + u[0] * cs.cos(x_r[2]) * dt,
                              x_r[1] + u[0] * cs.sin(x_r[2]) * dt,
                              x_r[2] + u[1] * dt)
        w_next = 1. / ((x_r_next[:2] - peds[0]).T @ (x_r_next[:2] - peds[0]) - safe_threshold ** 2)
        for i in range(1, len(peds)):
            w_next = w_next + 1. / (
                        (x_r_next[:2] - peds[i]).T @ (x_r_next[:2] - peds[i]) - safe_threshold ** 2)
        f = cs.vertcat(x_r_next, w_next)

        var_dict = {"x": x, "u": u}
        var_dict.update({f"p_{i}": peds[i] for i in range(n_peds)})
        input_vars = ["x", "u"] + [f"p_{i}" for i in range(n_peds)]

        var_dict_f = var_dict.copy()
        var_dict_f.update({"f": f})
        self._f = cs.Function("f_dyn",
                              var_dict_f,
                              input_vars, ["f"])

        var_dict_f_x = var_dict.copy()
        var_dict_f_x.update({"f_x": cs.jacobian(f, x)})
        self._f_x = cs.Function("f_dyn_x",
                                var_dict_f_x,
                                input_vars, ["f_x"])

        var_dict_f_u = var_dict.copy()
        var_dict_f_u.update({"f_u": cs.jacobian(f, u)})
        self._f_u = cs.Function("f_dyn_u",
                                var_dict_f_u,
                                input_vars, ["f_u"])

    def __call__(self, x: np.ndarray, u: np.ndarray, step: int) -> np.ndarray:
        args = [x, u] + [self._ped_traj[step, i, :] for i in range(self._n_peds)]
        return self._f(*args).toarray().reshape(4)

    def evaluate_jacobians(self, x: np.ndarray, u: np.ndarray, step: int) -> Tuple[np.ndarray, np.ndarray]:
        args = [x, u] + [self._ped_traj[step, i, :] for i in range(self._n_peds)]
        f_x = self._f_x(*args).toarray()
        f_u = self._f_u(*args).toarray()
        return f_x, f_u

    def eval_barrier(self, x_r: np.ndarray, step: int) -> float:
        w = 0.
        for i in range(self._n_peds):
            x_p = self._ped_traj[step, i, :]
            diff = x_r[:2] - x_p[:2]
            w = w + 1. / ((diff.T @ diff) - self._safe_threshold ** 2)
        return w
