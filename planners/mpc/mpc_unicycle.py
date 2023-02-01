from pyminisim.world_map import EmptyWorld
from pyminisim.visual import CircleDrawing, Renderer
from pyminisim.sensors import (PedestrianDetector, PedestrianDetectorConfig,
                               PedestrianDetectorNoise)
from pyminisim.robot import UnicycleRobotModel
from pyminisim.pedestrians import (HeadedSocialForceModelPolicy,
                                   RandomWaypointTracker)
from pyminisim.core import Simulation
import yaml
import pygame
import numpy as np
import do_mpc
import casadi
from typing import *
import time
import random
import os
import sys

sys.path.append('..')

import pathlib
planners_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
sys.path.append(planners_path)
from planners.utils import graphic_tools

class DoMPCController:
    def __init__(self, dt: float,
                 p_rob_init: np.ndarray,
                 p_rob_ref: np.ndarray,
                 sim: Simulation,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 lb: List[float],
                 ub: List[float],
                 r_rob: float,
                 r_ped: float,
                 min_safe_dist: float,
                 n_horizon: int,
                 simulation_time: float) -> None:
        # assignment
        self._dt = dt
        self._p_rod_init = p_rob_init.copy()
        self._p_rob_ref = self.get_ref_direction(p_rob_init, p_rob_ref)
        self._sim = sim
        self._total_peds = total_peds
        self._Q = Q
        self._R = R
        self._lb = lb
        self._ub = ub
        self._r_rob = r_rob
        self._r_ped = r_ped
        self._min_safe_dist = min_safe_dist
        self._n_horizon = n_horizon
        self._simulation_time = simulation_time
        # Second order unicycle robot model
        self._model = do_mpc.model.Model('discrete')
        # state variables
        self.x = self._model.set_variable('_x', 'x')
        self.y = self._model.set_variable('_x', 'y')
        self.phi = self._model.set_variable('_x', 'phi')
        # control input variables
        self.v = self._model.set_variable('_u', 'v')
        self.w = self._model.set_variable('_u', 'w')
        # discrete equations of motion/en/latest/project_structure.html
        self._model.set_rhs('x', self.x + self.v * casadi.cos(self.phi) * dt)
        self._model.set_rhs('y', self.y + self.v * casadi.sin(self.phi) * dt)
        self._model.set_rhs('phi', self.phi + self.w * dt)
        # pedestrians
        self.p_peds = self._model.set_variable(
            '_tvp', 'p_peds', shape=(4, self._total_peds))
        # setup
        self._model.setup()
        # MPC controller
        self._mpc = do_mpc.controller.MPC(self._model)
        # controller setup
        setup_mpc = {
            'n_horizon': self._n_horizon,
            't_step': self._dt,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.print_level': 0, 'ipopt.sb': 'yes', 'print_time': 0}
        }
        self._mpc.set_param(**setup_mpc)
        # stage cost
        self.u = self._model._u.cat
        self.stage_cost = self.u.T @ self._R @ self.u
        # terminal cost
        self.p_rob_N = self._model._x.cat[:3]
        self.terminal_cost = (
            self.p_rob_N - self._p_rob_ref).T @ self._Q @ (self.p_rob_N - self._p_rob_ref)
        # set cost
        self._mpc.set_objective(lterm=self.stage_cost,
                                mterm=self.terminal_cost)
        # TODO: Вот этот терм для обычной unicycle модели важен, так по сути будет сглаживать скорости и делать движение плавнее
        self._mpc.set_rterm(v=1e-2, w=1e-2)
        # bounds
        # lb
        self._mpc.bounds['lower', '_u', 'v'] = self._lb[0]
        self._mpc.bounds['lower', '_u', 'w'] = self._lb[1]
        # ub
        self._mpc.bounds['upper', '_u', 'v'] = self._ub[0]
        self._mpc.bounds['upper', '_u', 'w'] = self._ub[1]
        # distance to pedestrians
        self.ub_cons = (self._r_rob + self._r_ped + self._min_safe_dist) ** 2
        self.ub_cons_vect = np.array(
            [self.ub_cons for _ in range(self._total_peds)])
        # inequality constrain for pedestrian
        self.xy_rob = casadi.hcat([self._model._x.cat[:2]
                                  for _ in range(self._total_peds)])
        self.xy_peds = self.p_peds[:2, :]
        self.sq = (self.xy_rob - self.xy_peds) ** 2
        peds_dist = self.sq[0, :] + self.sq[1, :]
        self._mpc.set_nl_cons('dist_to_peds', -peds_dist,
                              ub=-self.ub_cons_vect)
        # time-variable-parameter function for pedestrian
        self.mpc_ped_tvp_fun = self._mpc.get_tvp_template()

        def mpc_tvp_fun(t_now):
            return self.mpc_ped_tvp_fun
        self._mpc.set_tvp_fun(mpc_tvp_fun)
        # setup
        self._mpc.setup()

    def get_ref_direction(self,
                          p_rob_init: np.ndarray,
                          p_rob_ref: np.ndarray) -> np.ndarray:
        """ Get direction angle of reference position

        This helps MPC to find solution when the reference point is located behind the robot.

        Args:
            p_rob_init (np.ndarray): initial robot state vector (for more details see config file)
            p_rob_ref (np.ndarray): reference robot position (for more details see config file)

        Returns:
            np.ndarray: p_rob_ref (np.ndarray): reference robot position with a changed direction
        """
        x_goal_polar = p_rob_ref[0] - p_rob_init[0]
        y_goal_polar = p_rob_ref[1] - p_rob_init[1]
        return np.array([p_rob_ref[0], p_rob_ref[1], np.arctan2(y_goal_polar, x_goal_polar)])

    def propagate_all_pedestrians(self,
                                  p_peds_k: np.ndarray,
                                  dt: float):
        """ Function returns next state for the specified pedestrians

        Propagation model with constant_velocity

        Args:
            p_peds_k (np.ndarray): state vector of all pedestrians: [x_init,  y_init, vx, vy], [m, m, m/s, m/s]
            dt (float): time step

        Return:
            p_peds_k+1 (np.ndarray): state vector at k + 1
        """
        p_peds_k_1 = p_peds_k[:]
        x = p_peds_k_1[0, :]
        y = p_peds_k_1[1, :]
        vx = p_peds_k_1[2, :]
        vy = p_peds_k_1[3, :]
        p_peds_k_1[0, :] = x + vx * dt
        p_peds_k_1[1, :] = y + vy * dt
        return p_peds_k_1


def create_sim(p_rob_init: np.ndarray,
               dt: float,
               total_peds: int,
               ped_dect_range,
               ped_dect_fov,
               misdetection_prob,
               is_robot_visible: bool) -> Tuple[Simulation, Renderer]:
    robot_model = UnicycleRobotModel(initial_pose=p_rob_init,
                                     initial_control=np.array([0., np.deg2rad(0.)]))
    tracker = RandomWaypointTracker(world_size=(10.0, 15.0))
    if total_peds > 0:
        pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=total_peds,
                                                     waypoint_tracker=tracker,
                                                     robot_visible=is_robot_visible)
    else:
        pedestrians_model = None
    pedestrian_detector_config = PedestrianDetectorConfig(ped_dect_range,
                                                          ped_dect_fov)
    sensors = [PedestrianDetector(config=pedestrian_detector_config,
                                  noise=PedestrianDetectorNoise(0, 0, 0, 0, misdetection_prob))]
    sim = Simulation(sim_dt=dt,
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     sensors=sensors)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(700, 700))
    return sim, renderer


def main() -> None:
    # Assign file paths
    file_name = os.path.basename(__file__)
    file_path = os.path.relpath(__file__).replace(file_name, '')
    config_path = f'{file_path}configs'
    results_path = f'{file_path}results'
    file_name = file_name.split('.')[0]

    # Load config
    with open(rf'{config_path}/{file_name}_config.yaml') as f:
        config = yaml.safe_load(f)
        # General
        p_rob_init: np.ndarray = np.array(config['X_rob_init'])
        p_rob_ref: np.ndarray = np.array(config['p_rob_ref'])
        dt: float = config['dt']
        n_horizon: int = config['n_horizon']
        Q: np.ndarray = np.array(config['Q'])
        R: np.ndarray = np.array(config['R'])
        r_rob: float = config['r_rob']
        r_ped: float = config['r_ped']
        simulation_time: float = config['simulation_time']
        lb: List[float] = config['lb']
        ub: List[float] = config['ub']
        min_safe_dist: float = config['min_safe_dist']
        total_peds: int = config['total_peds']
        ped_dect_range: float = config['ped_dect_range']
        ped_dect_fov: float = np.deg2rad(config['ped_dect_fov'])
        misdetection_prob: float = config['misdetection_prob']
        x_dummy_ped: np.ndarray = config['x_dummy_ped']
        is_robot_visible: bool = config['is_robot_visible']
        # Graphics
        col_hex: List[str] = config['col_hex']
        if len(col_hex) < total_peds:
            col_hex.append('#' + "%06x" % random.randint(0, 0xFFFFFF))
        col_rgb: List[Tuple[int]] = [
            graphic_tools.hex_to_rgb(col) for col in col_hex]

    # Initialization
    sim, renderer = create_sim(p_rob_init,
                               dt/10,
                               total_peds,
                               ped_dect_range,
                               ped_dect_fov,
                               misdetection_prob,
                               is_robot_visible)
    renderer.initialize()
    controller = DoMPCController(dt,
                                 p_rob_init,
                                 p_rob_ref,
                                 sim,
                                 total_peds,
                                 Q,
                                 R,
                                 lb,
                                 ub,
                                 r_rob,
                                 r_ped,
                                 min_safe_dist,
                                 n_horizon,
                                 simulation_time)
    renderer.draw("goal",
                  CircleDrawing(p_rob_ref[:2],
                                0.1,
                                (255, 0, 0),
                                0))

    # Main loop
    running = True
    sim.step()  # First step can take some time due to Numba compilation
    start_time = time.time()
    end_time = start_time
    n_frames = 0
    u_pred = np.array([0., 0.])
    hold_time = sim.sim_dt
    controller._mpc.x0 = p_rob_init
    controller._mpc.set_initial_guess()
    pred_peds_traj = np.zeros(
        [int(simulation_time/dt), total_peds, n_horizon+1, 2])
    gt_peds_traj = np.zeros([int(simulation_time/dt), total_peds*2])
    pred_rob_traj = np.zeros([int(simulation_time/dt), n_horizon + 1, 2])
    sim_step = 0
    while running:
        renderer.render()
        n_frames += 1
        if hold_time >= dt:  # Every ten-th step of simulation we calculate control
            X_rob_cur = sim.current_state.world.robot.state
            detect_ped_ind = sim.current_state.sensors['pedestrian_detector'].reading.pedestrians.keys(
            )
            undetect_ped_ind = list(
                set(range(total_peds)) - set(detect_ped_ind))
            gt_ped_data = np.concatenate([sim.current_state.world.pedestrians.poses[:, :2],
                                         sim.current_state.world.pedestrians.velocities[:, :2]], axis=1)
            gt_ped_data[undetect_ped_ind] = x_dummy_ped
            # save for matplotlib
            gt_peds_traj[sim_step] = sim.current_state.world.pedestrians.poses[:, :2].flatten()
            # propagation of pedestrians inside horizon
            for k in range(n_horizon + 1):
                if k == 0:
                    controller.mpc_ped_tvp_fun['_tvp',
                                               0, 'p_peds'] = gt_ped_data.T
                else:
                    controller.mpc_ped_tvp_fun['_tvp', k, 'p_peds'] = controller.propagate_all_pedestrians(
                        controller.mpc_ped_tvp_fun['_tvp', k-1, 'p_peds'], dt)
            # render pedestrian predictions
            for i in range(total_peds):
                for k in range(0, n_horizon + 1):
                    pred_ped__xy_at_step_k = np.array(
                        controller.mpc_ped_tvp_fun['_tvp', k, 'p_peds']).T[i][:2]
                    renderer.draw(f"ped_{i}_{k}", CircleDrawing(
                        pred_ped__xy_at_step_k, 0.05, col_rgb[i], 0))
                    pred_peds_traj[sim_step, i, k, :] = pred_ped__xy_at_step_k
            # Calculate action
            u_pred = controller._mpc.make_step(X_rob_cur).T[0]
            # render robot predicted trajectory
            x_pred = controller._mpc.data.prediction(('_x', 'x'))[0]
            y_pred = controller._mpc.data.prediction(('_x', 'y'))[0]
            array_xy_pred = np.concatenate([x_pred, y_pred], axis=1)
            pred_rob_traj[sim_step] = array_xy_pred
            sim_step += 1
            for i in range(n_horizon+1):
                renderer.draw(
                    f"{1000+i}", CircleDrawing(array_xy_pred[i], 0.03, (255, 100, 0), 0))
            hold_time = 0.
        # Simulate
        sim.step(u_pred)
        current_time = time.time()
        hold_time += sim.sim_dt
        if current_time - start_time >= simulation_time:
            end_time = current_time
            break
    fps = int(n_frames / (end_time - start_time))
    print("FPS: ", fps)
    # Done! Time to quit.
    pygame.quit()

    # Graphics
    # get data
    mpc_graphics = do_mpc.graphics.Graphics(controller._mpc.data)
    # robot trajectory
    y_rob_plt = mpc_graphics.data._x[:, 0]
    x_rob_plt = mpc_graphics.data._x[:, 1]
    phi_rob_plt = mpc_graphics.data._x[:, 2]
    # pedestrians trajectories
    y_peds_plt = gt_peds_traj[:, 0::2]
    x_peds_plt = gt_peds_traj[:, 1::2]
    graphic_tools.make_animation(y_rob_plt,
                                 x_rob_plt,
                                 phi_rob_plt,
                                 y_peds_plt,
                                 x_peds_plt,
                                 pred_peds_traj,
                                 pred_rob_traj,
                                 'Robot Trajectory with MPC',
                                 results_path,
                                 config_path,
                                 file_name,
                                 col_hex,
                                 config)
    # graphic_tools.mpc_step_response(controller._mpc.data,
    #                                results_path,
    #                                file_name)


if __name__ == '__main__':
    main()
