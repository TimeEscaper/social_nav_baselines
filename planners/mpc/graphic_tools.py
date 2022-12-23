# Graphic tools for planners visualization
# Kashrin Aleksandr

import os
import shutil
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from typing import *

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

def make_animation(y_rob_plt: np.ndarray, 
                   x_rob_plt: np.ndarray, 
                   phi_rob_plt: np.ndarray,
                   y_peds_plt: np.ndarray, 
                   x_peds_plt: np.ndarray,
                   pred_peds_traj: np.ndarray,
                   pred_rob_traj: np.ndarray,
                   title: str,
                   results_path: str,
                   config_path: str,
                   file_name: str,
                   col_hex: List[str],
                   config: Dict[str, Any]) -> None:
    """Function makes animation and stores it in the specified path

    Args:
        y_rob_plt (np.ndarray): y position [m] of the robot, 1-D numpy array
        x_rob_plt (np.ndarray): x position [m] of the robot, 1-D numpy array
        phi_rob_plt (np.ndarray): phi angle [rad] of the robot, 1-D numpy array
        y_peds_plt (np.ndarray): y position [m] of the pedestrians, 2-D array like [[y0_ped1, y0_ped2, y0_ped3], 
                                                                                    [y1_ped1, y1_ped2, y1_ped3]]
        x_peds_plt (np.ndarray): x position [m] of the pedestrians, 2-D array like [[x0_ped1, x0_ped2, x0_ped3], 
                                                                                    [x1_ped1, x1_ped2, x1_ped3]]
        pred_peds_traj (np.ndarray): predicted pedestrian trajectories at each simulation step, 4-D array like [simulation step, pedestrian i, ped_x, ped_y]
        pred_rob_traj (np.ndarray): predicted robot trajectory at each simulation step, 3-D array like [simulation step, rob_x, rob_y]
        title (str): animation plot title
        results_path (str): pathdir to results folder
        config_path (str): pathdir to config folder
        file_name (str): file name
        col_hex (List[str]): list of generated hex colors
        config (Dict[str, Any]): configuration file
    """

    # config parse
    # General
    X_rob_init: np.ndarray = np.array(config['X_rob_init'])
    X_rob_init[0], X_rob_init[1] = X_rob_init[1], X_rob_init[0]
    p_rob_ref: np.ndarray = np.array(config['p_rob_ref'])
    p_rob_ref[0], p_rob_ref[1] = p_rob_ref[1], p_rob_ref[0]
    dt: float = config['dt']
    r_rob: float = config['r_rob']
    r_ped: float = config['r_ped']
    total_peds: int = config['total_peds']
    # Graphics
    annotation_offset: np.ndarray = np.array(config['annotation_offset'])
    animation_format: str = config['animation_format']
    
    # Customizing Matplotlib:
    mpl.rcParams['font.size'] = 18
    mpl.rcParams['lines.linewidth'] = 3
    mpl.rcParams['axes.grid'] = True
   
    # set figure
    fig, ax = plt.subplots(figsize=[16, 16], facecolor='white')
    ax.set_aspect('equal', adjustable='box')
    fig.suptitle(title, fontsize=35)
    
    # animation function
    cnt = 0

    def plot_pedestrian(x_ped_plt, y_ped_plt, cnt, i) -> None:
        # plot pedestrian i position
        ax.plot(x_ped_plt[:cnt], y_ped_plt[:cnt], linewidth=3, color=col_hex[i], label=f'Pedestrian {i+1}')
        # plot pedestrian i area
        ped1_radius_plot = plt.Circle((x_ped_plt[cnt], y_ped_plt[cnt]), r_ped, fill=False, linewidth=5, color=col_hex[i])
        ax.add_patch(ped1_radius_plot)
        # annotate pedestrian i
        ped_coord = (round(x_ped_plt[cnt], 2), (round(y_ped_plt[cnt], 2)))
        ax.annotate(f'Pedestrian {i+1}: {ped_coord}', ped_coord + np.array([0, r_ped]) + annotation_offset,  ha='center')
        # plot pedestrian prediction
        ax.plot(pred_peds_traj[cnt, i, :, 1], pred_peds_traj[cnt, i, :, 0], color=col_hex[i], linestyle='dashed', linewidth=1)
        
    # find max, min for plots
    max_x_plt = np.max(x_rob_plt)
    min_x_plt = np.min(x_rob_plt)
    max_y_plt = np.max(y_rob_plt)
    min_y_plt = np.min(y_rob_plt)

    def animate(i) -> None:
        nonlocal cnt
        ax.clear()
        # plot robot position
        ax.plot(x_rob_plt[:cnt], y_rob_plt[:cnt], linewidth=3, color='blue', label='Robot')
        ax.set_xlim([min_x_plt - r_rob - 0.5, max_x_plt + r_rob + 0.5])
        ax.set_ylim([min_y_plt - r_rob - 0.5, max_y_plt + r_rob + 0.5])
        # plot robot area
        robot_radius_plot = plt.Circle((x_rob_plt[cnt], y_rob_plt[cnt]), r_rob, fill=False, linewidth=5, color='blue')
        robot_fill= plt.Circle((x_rob_plt[cnt], y_rob_plt[cnt]), r_rob, fill=True, color='r', alpha=0.3)
        ax.add_patch(robot_radius_plot)
        ax.add_patch(robot_fill)
        # annotate robot
        robot_coord = (round(x_rob_plt[cnt], 2), (round(y_rob_plt[cnt], 2)))
        ax.annotate(f'Robot: {robot_coord}', robot_coord + np.array([0, r_rob]) + annotation_offset,  ha='center')
        # plot robot goal
        ax.plot(float(p_rob_ref[0]), float(p_rob_ref[1]), 'y*', markersize=10)
        # annotate robot goal
        goal_coord = (round(float(p_rob_ref[0]), 2), round(float(p_rob_ref[1]), 2))
        ax.annotate(f'Goal: {goal_coord}', goal_coord - np.array([0, r_rob]) - annotation_offset,  ha='center')
        # plot robot start
        ax.plot(float(X_rob_init[0]), float(X_rob_init[1]), 'yo', markersize=10)
        # annotate robot start
        start_coord = (round(float(X_rob_init[0]), 2), round(float(X_rob_init[1]), 2))
        ax.annotate(f'Start: {start_coord}', start_coord - np.array([0, r_rob]) - annotation_offset,  ha='center')
        # plot robot direction arrow
        plt.arrow(x_rob_plt[cnt], y_rob_plt[cnt], np.sin(phi_rob_plt[cnt])*r_rob,  np.cos(phi_rob_plt[cnt])*r_rob, color='b', width=r_rob/5)
        # plot predicted robot trajectory
        ax.plot(pred_rob_traj[cnt, :, 1], pred_rob_traj[cnt, :, 0], linewidth=1, color='blue', linestyle='dashed')
        
        for i in range(total_peds):
            plot_pedestrian(x_peds_plt[:, i], y_peds_plt[:, i], cnt, i)
        
        # legend
        ax.set_xlabel('$y$ [m]')
        ax.set_ylabel('$x$ [m]')
        ax.legend()
        # increment counter
        cnt = cnt + 1
    
    print('Animation: Start!')  
    time_start = time.time() 
    frames = len(x_rob_plt)-2
    anim = FuncAnimation(fig, animate, frames=frames, interval=dt, repeat=False)
    
    # automatic result saving
    result_name = file_name + '_result_'
    dirs = [i for i in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,i)) and result_name in i]
    if dirs:
        ind = max([int(line.split('_')[-1]) for line in dirs])
    else:
        ind = 0
    
    result_directory = rf'{results_path}/{result_name}{ind+1}'
    os.mkdir(result_directory)
    anim.save(rf'{result_directory}/{result_name}{ind+1}{animation_format}', 'ffmpeg', 10)
    shutil.copy(rf'{config_path}/{file_name}_config.yaml', rf'{result_directory}/{file_name}_config_{ind+1}.yaml')
    time_end = time.time() 
    print(f'Animation: Finished! Elapsed time: {round(time_end - time_start, 2)}s')
    
def mpc_step_response(mpc_data, 
                      results_path, 
                      file_name) -> None:
    
     # Step response
    time_plt = mpc_data._time
    x_plt = mpc_data['_x', 'x', :]
    y_plt = mpc_data['_x', 'y', :]
    phi_plt = mpc_data['_x', 'phi', :]
    v_plt = mpc_data['_x', 'v', :]
    w_plt = mpc_data['_x', 'w', :]
    u_a_plt = mpc_data['_u', 'u_a', :]
    u_alpha_plt = mpc_data['_u', 'u_alpha', :]
    # set figure
    fig, ax = plt.subplots(3, sharex=True, figsize=(16,9), facecolor='white')
    fig.align_ylabels()
    fig.suptitle('Robot Step Response', fontsize=35)
    ax[0].plot(time_plt, x_plt, color='#1f77b4', linestyle='solid', label=r'$x, [m]$')
    ax[0].plot(time_plt, y_plt, color='#ff7f0e', linestyle='solid', label=r'$y, [m]$')
    ax[0].plot(time_plt, phi_plt, color='#2ca02c', linestyle='solid', label=r'$\phi, [rad]$')
    ax[1].plot(time_plt, v_plt, color='#d62728', linestyle='solid', label=r'$v, [\frac{m}{s}]$')
    ax[1].plot(time_plt, w_plt, color='#9467bd', linestyle='solid', label=r'$\omega, [\frac{rad}{s^2}]$')
    ax[2].plot(time_plt, u_a_plt, color='#1f77b4', linestyle='solid', label=r'$u_a, [\frac{m}{s^2}]$')
    ax[2].plot(time_plt, u_alpha_plt, color='#ff7f0e', linestyle='solid', label=r'$\omega, [\frac{rad}{s^2}]$')
    
    ax[0].set_ylabel(r'$Pose$')
    ax[0].legend()
    ax[1].set_ylabel(r'$Velocity$')
    ax[1].legend()
    ax[2].set_ylabel(r'$Control$')
    ax[2].legend()
    ax[2].set_xlabel(r'$time, [s]$')
    format_of_plot = '.png'
    
     # automatic result saving
    result_name = file_name + '_result_'
    dirs = [i for i in os.listdir(results_path) if os.path.isdir(os.path.join(results_path,i)) and result_name in i]
    ind = max([int(line.split('_')[-1]) for line in dirs])
    
    result_directory = rf'{results_path}/{result_name}{ind}'
    
    plt.savefig(rf'{result_directory}/step_response_{ind}{format_of_plot}')