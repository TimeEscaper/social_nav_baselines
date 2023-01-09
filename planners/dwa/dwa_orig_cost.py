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
from typing import *
import time
import random
import math
import os
import sys

sys.path.append('../')

import pathlib
planners_path = str(pathlib.Path(__file__).parent.parent.parent.resolve())
sys.path.append(planners_path)
from planners.utils import graphic_tools

def propagate_all_pedestrians(p_peds_k: np.ndarray,
                              dt: float):
    """ Function returns next state for the specified pedestrians

    Propagation model with constant velocity

    Args:
        p_peds_k (np.ndarray): state vector of all pedestrians: [x_init,  y_init, vx, vy], [m, m, m/s, m/s]
        dt (float): time step

    Return:
        p_peds_k+1 (np.ndarray): state vector at k + 1
    """
    p_peds_k_1 = p_peds_k
    x = p_peds_k_1[:, 0]
    y = p_peds_k_1[:, 1]
    vx = p_peds_k_1[:, 2]
    vy = p_peds_k_1[:, 3]
    p_peds_k_1[:, 0] = x + vx * dt
    p_peds_k_1[:, 1] = y + vy * dt
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
    pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=total_peds,
                                                     waypoint_tracker=tracker,
                                                     robot_visible=is_robot_visible)
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


def hex_to_rgb(value: str) -> Tuple[int]:
    """Function convert HEX format color to RGB

    Args:
        value (str): HEX color: '#f2a134'

    Returns:
        Tuple[int]: RGB color: (242, 161, 52)
    """
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def dwa_make_step(X_rob_cur: np.ndarray,
                  propagation_ped_traj: np.ndarray,
                  p_rob_ref: np.ndarray,
                  n_horizon: int,
                  dt: float,
                  config) -> np.ndarray:
    """Function generates control input according to the minimal cost

    Args:
        X_rob_cur (np.ndarray): Current robot state vector: [x, y, phi]
        propagation_ped_traj (np.ndarray): Predicted pedestrian trajectories [prediction step, pedestrian, [x, y, vx, vy]]

    Returns:
        np.ndarray: Control input
    """
    minG = math.inf

    sigma = config['weights'][0]
    alpha = config['weights'][1]
    betta = config['weights'][2]
    gamma = config['weights'][3]

    v_max = config['ub'][0]

    best_u = [0.0, 0.0]

    best_pred_rob_traj = np.zeros([n_horizon, 3])

    # посчитаем допустимые скорости для движения
    v_range = np.arange(config['lb'][0], config['ub']
                        [0], config['v_resolution'])
    if 0 not in v_range:
        np.append(v_range, 0)
    w_range = np.arange(config['lb'][1], config['ub']
                        [1], config['w_resolution'])
    if 0 not in w_range:
        np.append(w_range, 0)

    # Будем выбирать такую скорость, что функция стоимости была минимальной
    for v in v_range:
        for w in w_range:  # Нужно чтобы обязательно была возможность сделать нулевую скорость, чтобы у робота был шанс остановиться
            # Посчитаем предсказанную траекторию робота с выбранными скоростями
            pred_rob_traj = get_pred_rob_traj(X_rob_cur, v, w, n_horizon, dt)
            # Посчитаем стоимость такой комбинации
            G = sigma * (alpha * heading(pred_rob_traj, p_rob_ref, n_horizon) + \
                         betta * dist(pred_rob_traj, propagation_ped_traj, n_horizon) + \
                         gamma * (v_max - v))

            # А что если я буду минимизировать расстояние до цели
            #G = alpha * dist_to_goal(pred_rob_traj, p_rob_ref) + \
            #    betta * dist(pred_rob_traj, propagation_ped_traj, n_horizon)

            if G < minG:
                minG = G
                best_u = [v, w]
                best_pred_rob_traj = pred_rob_traj

    return best_u, best_pred_rob_traj


def get_pred_rob_traj(X_rob_cur, v, w, n_horizon, dt) -> np.ndarray:
    # propagate rob trajectory along horizon
    pred_trajectory = np.zeros([n_horizon, 3])
    pred_trajectory[0, :] = X_rob_cur
    for i in range(1, n_horizon):
        new_x = pred_trajectory[i - 1, 0] + v * \
            np.cos(pred_trajectory[i - 1, 2]) * dt
        new_y = pred_trajectory[i - 1, 1] + v * \
            np.sin(pred_trajectory[i - 1, 2]) * dt
        new_phi = pred_trajectory[i - 1, 2] + w * dt
        new_phi = (new_phi + np.pi) % (2 * np.pi) - np.pi
        pred_trajectory[i, :] = np.array([new_x, new_y, new_phi])
    return pred_trajectory


def dist_to_goal(pred_rob_traj, p_rob_ref):
    # будем оптимизировать дистанцию до конечной цели
    # чем ближе к цели -> тем меньше кост
    dx_0 = pred_rob_traj[0, 0] - p_rob_ref[0]
    dy_0 = pred_rob_traj[0, 1] - p_rob_ref[1]
    dist_at_beginning = math.sqrt(dx_0 ** 2 + dy_0 ** 2)

    dx_f = pred_rob_traj[-1, 0] - p_rob_ref[0]
    dy_f = pred_rob_traj[-1, 1] - p_rob_ref[1]
    dist_after_pred = math.sqrt(dx_f ** 2 + dy_f ** 2)

    normalized_dist = dist_after_pred / dist_at_beginning

    # print(normalized_dist)

    return normalized_dist


def heading(pred_rob_traj, p_rob_ref, n_horizon) -> float:
    # is the sum of heading errors along defined path
    sum_cost_heading = 0
    for pred_rob_pos in pred_rob_traj:
        dx = p_rob_ref[0] - pred_rob_pos[0]
        dy = p_rob_ref[1] - pred_rob_pos[1]
        error_angle = math.atan2(dy, dx)
        overall_deviation_angle = error_angle - pred_rob_pos[2]
        overall_deviation_angle = (
            overall_deviation_angle + np.pi) % (2 * np.pi) - np.pi
        # TODO: Это я типа нормализацию придумал подобно статье
        cost = abs(overall_deviation_angle) / np.pi
        sum_cost_heading += cost
    return sum_cost_heading / n_horizon


def dist(pred_rob_traj, propagation_ped_traj, n_horizon):
    # функция стоимости дистанции до каждого пешехода
    # в каждый момент времени будем искать ближайшего пешехода, при этом сами пешеходы тоже идут
    # считать стоимость до него и суммировать в финалиную стоимость
    sum_cost_dist = 0
    for step in range(n_horizon):
        ox = propagation_ped_traj[step, :, 0]
        oy = propagation_ped_traj[step, :, 1]
        dx = pred_rob_traj[step, 0] - ox[:, None]  # TODO: Проверить в дебагере
        dy = pred_rob_traj[step, 1] - oy[:, None]
        r = np.hypot(dx, dy)
        min_r = np.min(r)
        cost = 1.0 / min_r
        sum_cost_dist += cost

    normalized_sum = sum_cost_dist / n_horizon
    # print(normalized_sum)
    return normalized_sum


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
        X_rob_init: np.ndarray = np.array(config['X_rob_init'])
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
        min_ref_dist: float = config['min_ref_dist']
        # Graphics
        col_hex: List[str] = config['col_hex']
        if len(col_hex) < total_peds:
            col_hex.append('#' + "%06x" % random.randint(0, 0xFFFFFF))
        col_rgb: List[Tuple[int]] = [hex_to_rgb(col) for col in col_hex]

    # Initialization
    sim, renderer = create_sim(X_rob_init,
                               dt/10,
                               total_peds,
                               ped_dect_range,
                               ped_dect_fov,
                               misdetection_prob,
                               is_robot_visible)
    renderer.initialize()
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
    pred_peds_traj = np.zeros(
        [int(simulation_time/dt), total_peds, n_horizon, 2])
    gt_peds_traj = np.zeros([int(simulation_time/dt), total_peds*2])
    pred_rob_traj = np.zeros([int(simulation_time/dt), n_horizon, 2])
    sim_step = 0

    # pedestrian propagation array
    propagation_ped_traj = np.zeros([n_horizon, total_peds, 4])

    # robot trajectory
    res_rob_traj = np.zeros([int(simulation_time/dt), 3])

    while running:
        renderer.render()
        n_frames += 1
        if hold_time >= dt:  # Every ten-th step of simulation we calculate control
            X_rob_cur = sim.current_state.world.robot.state

            cur_dist_to_goal = math.sqrt(
                (X_rob_cur[0] - p_rob_ref[0]) ** 2 + (X_rob_cur[1] - p_rob_ref[1]) ** 2)
            if cur_dist_to_goal <= min_ref_dist:
                end_time = current_time
                break

            res_rob_traj[sim_step, :] = X_rob_cur

            detect_ped_ind = sim.current_state.sensors['pedestrian_detector'].reading.pedestrians.keys(
            )
            undetect_ped_ind = list(
                set(range(total_peds)) - set(detect_ped_ind))
            gt_ped_data = np.concatenate([sim.current_state.world.pedestrians.poses[:, :2],
                                         sim.current_state.world.pedestrians.velocities[:, :2]], axis=1)
            gt_ped_data[undetect_ped_ind] = x_dummy_ped
            # save for matplotlib
            gt_peds_traj[sim_step] = sim.current_state.world.pedestrians.poses[:, :2].flatten(
            )

            # propagation of pedestrians inside horizonbest_u
            for k in range(n_horizon):
                if k == 0:
                    propagation_ped_traj[k, :, :] = gt_ped_data
                else:
                    propagation_ped_traj[k, :, :] = propagate_all_pedestrians(
                        propagation_ped_traj[k-1, :, :], dt)
            # render pedestrian predictions
            for k in range(0, n_horizon):
                for i in range(total_peds):
                    renderer.draw(f"ped_{i}_{k}", CircleDrawing(
                        propagation_ped_traj[k, i, :2], 0.05, col_rgb[i], 0))
                    pred_peds_traj[sim_step, i, k,
                                   :] = propagation_ped_traj[k, i, :2]

            # Calculate action
            u_pred, rob_traj = dwa_make_step(
                X_rob_cur, propagation_ped_traj, p_rob_ref, n_horizon, dt, config)
            pred_rob_traj[sim_step, :] = rob_traj[:, :2]
            sim_step += 1
            # print(pred_rob_traj)
            for i in range(n_horizon):
                renderer.draw(
                    f"{1000+i}", CircleDrawing(rob_traj[i, :2], 0.03, (255, 100, 0), 0))
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
    # robot trajectory
    y_rob_plt = res_rob_traj[:sim_step+1, 0]
    x_rob_plt = res_rob_traj[:sim_step+1, 1]
    phi_rob_plt = res_rob_traj[:sim_step+1, 2]
    # pedestrians trajectories
    y_peds_plt = gt_peds_traj[:sim_step+1, 0::2]
    x_peds_plt = gt_peds_traj[:sim_step+1, 1::2]
    graphic_tools.make_animation(y_rob_plt,
                                 x_rob_plt,
                                 phi_rob_plt,
                                 y_peds_plt,
                                 x_peds_plt,
                                 pred_peds_traj,
                                 pred_rob_traj,
                                 'Robot Trajectory with DWA',
                                 results_path,
                                 config_path,
                                 file_name,
                                 col_hex,
                                 config)


if __name__ == '__main__':
    main()
