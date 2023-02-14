import random
from typing import Dict, List, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from pyminisim.visual import CircleDrawing, Renderer

DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

class Visualizer():
    def __init__(self,
                 total_peds: int,
                 renderer: Optional[Renderer] = None) -> None:
        self._ground_truth_pedestrian_trajectories: List[np.ndarray] = []
        self._ground_truth_robot_trajectory: List[np.ndarray] = []
        self._predicted_pedestrians_trajectories: List[np.ndarray] = []
        self._predicted_robot_trajectory: List[np.ndarray] = []
        self._renderer = renderer
        self._total_peds = total_peds
        self._palette_hex = DEFAULT_COLOR_HEX_PALETTE
        self._palette_rgb: List[int] = []
        self._set_of_goals: List[np.ndarray] = []
        if len(self._palette_hex) < total_peds:
            self._palette_hex.append('#' + "%06x" % random.randint(0, 0xFFFFFF))
        for color_hex in self._palette_hex:
            self._palette_rgb.append(Visualizer.hex_to_rgb(color_hex))

    @property
    def ground_truth_pedestrian_trajectories(self) -> np.ndarray:
        return np.asarray(self._ground_truth_pedestrian_trajectories)
    
    @property
    def ground_truth_robot_trajectory(self) -> np.ndarray:
        return np.asarray(self._ground_truth_robot_trajectory)
    
    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return np.asarray(self._predicted_pedestrians_trajectories)

    @property
    def predicted_robot_trajectory(self) -> np.ndarray:
        return np.asarray(self._predicted_robot_trajectory)

    @property
    def renderer(self) -> Renderer:
        return self._renderer

    @property
    def palette_hex(self) -> List[str]:
        return self._palette_hex
    
    @property
    def palette_rgb(self) -> List[str]:
        return self._palette_rgb

    def append_ground_truth_pedestrians_pose(self,
                                             ground_truth_pedestrians_pose: np.ndarray) -> None:
        """Append ground truth pedestrians pose

        Args:
            ground_truth_pedestrians_pose (np.ndarray): Two-dimensional numpy array of pedestrians poses, [[x0, y0],
                                                                                                           [x1, y1],
                                                                                                           ...
                                                                                                           [xn, yn]]
        """
        self._ground_truth_pedestrian_trajectories.append(ground_truth_pedestrians_pose)

    def append_ground_truth_robot_state(self,
                                        ground_truth_robot_state: List[float]) -> None:
        """Append ground truth robot state

        Args:
            ground_truth_robot_state (List[float]): State of the robot, [x, y, theta]
        """
        self._ground_truth_robot_trajectory.append(ground_truth_robot_state)
        # Goal propagation 
        self._set_of_goals.append(self._set_of_goals[-1])

    def append_predicted_pedestrians_trajectories(self,
                                           predicted_pedestrians_poses: List[List[float]]) -> None:
        """Append predicted pedestrians trajectories

        Args:
            predicted_pedestrians_poses (List[List[float]]): _description_
        """
        self._predicted_pedestrians_trajectories.append(predicted_pedestrians_poses)

    def append_predicted_robot_trajectory(self,
                                          predicted_robot_trajectory: List[List[float]]) -> None:
        self._predicted_robot_trajectory.append(predicted_robot_trajectory)
    
    def visualize_predicted_robot_trajectory(self,
                                             predicted_robot_trajectory: List[List[float]]) -> None:
        assert self._renderer, f"You should provide renderer instance of the pyminisim to use this method!"
        for i in range(len(predicted_robot_trajectory)):
            self._renderer.draw(f"{1000+i}", CircleDrawing(predicted_robot_trajectory[i], 0.03, (255, 100, 0), 0))
    
    def visualize_predicted_pedestrians_trajectories(self,
                                                     predicted_pedestrians_trajectories: List[List[float]]) -> None:
        assert self._renderer, f"You should provide renderer instance of the pyminisim to use this method!"
        for step in range(len(predicted_pedestrians_trajectories)):
            for pedestrian in range(len(predicted_pedestrians_trajectories[0])):
                self._renderer.draw(f"ped_{step}_{pedestrian}", CircleDrawing(
                        predicted_pedestrians_trajectories[step][pedestrian], 0.05, self._palette_rgb[pedestrian], 0))

    @staticmethod
    def hex_to_rgb(value: str) -> List[int]:
        """Function convert HEX format color to RGB

        Args:
            value (str): HEX color: '#f2a134'

        Returns:
            Tuple[int]: RGB color: (242, 161, 52)
        """
        value = value.lstrip('#')
        lv = len(value)
        return [int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3)]
    
    def visualize_goal(self,
                       new_goal: np.ndarray) -> None:
        self._renderer.draw("goal",
                            CircleDrawing(new_goal[:2],
                                          0.1,
                                          (255, 0, 0),
                                          0))
        self._set_of_goals.append(new_goal)

    def make_animation(self,
                       title: str,
                       result_path: str,
                       config: Dict) -> None:
        # General
        init_state: np.ndarray = np.array(config['init_state'])
        init_state[0], init_state[1] = init_state[1], init_state[0]
        dt: float = config['dt']
        r_rob: float = config['r_rob']
        r_ped: float = config['r_ped']
        # Graphics
        annotation_offset: np.ndarray = np.array([0, 0.2])

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
            ax.plot(x_ped_plt[:cnt], y_ped_plt[:cnt], linewidth=3,
                    color=self._palette_hex[i], label=f'Pedestrian {i+1}')
            # plot pedestrian i area
            ped1_radius_plot = plt.Circle(
                (x_ped_plt[cnt], y_ped_plt[cnt]), r_ped, fill=False, linewidth=5, color=self._palette_hex[i])
            ax.add_patch(ped1_radius_plot)
            # annotate pedestrian i
            ped_coord = (round(x_ped_plt[cnt], 2), (round(y_ped_plt[cnt], 2)))
            ax.annotate(f'Pedestrian {i+1}: {ped_coord}', ped_coord +
                        np.array([0, r_ped]) + annotation_offset,  ha='center')
            # plot pedestrian prediction
            ax.plot(self.predicted_pedestrians_trajectories[cnt, :, i, 1], self.predicted_pedestrians_trajectories[cnt, :,
                    i, 0], color=self._palette_hex[i], linestyle='dashed', linewidth=1)

        # Data parsing
        x_rob_plt = self.ground_truth_robot_trajectory[:, 1]
        y_rob_plt = self.ground_truth_robot_trajectory[:, 0]
        phi_rob_plt = self.ground_truth_robot_trajectory[:, 2]
        x_peds_plt = self.ground_truth_pedestrian_trajectories[:, :, 1]
        y_peds_plt = self.ground_truth_pedestrian_trajectories[:, :, 0]

        # find max, min for plots
        max_x_plt = np.max(x_rob_plt)
        min_x_plt = np.min(x_rob_plt)
        max_y_plt = np.max(y_rob_plt)
        min_y_plt = np.min(y_rob_plt)

        def animate(i) -> None:
            nonlocal cnt
            ax.clear()
            # plot robot position
            ax.plot(x_rob_plt[:cnt], y_rob_plt[:cnt],
                    linewidth=3, color='blue', label='Robot')
            ax.set_xlim([-5, 5])
            ax.set_ylim([-5, 5])
            #ax.set_xlim([min_x_plt - r_rob - 0.5, max_x_plt + r_rob + 0.5])
            #ax.set_ylim([min_y_plt - r_rob - 0.5, max_y_plt + r_rob + 0.5])
            # plot robot area
            robot_radius_plot = plt.Circle(
                (x_rob_plt[cnt], y_rob_plt[cnt]), r_rob, fill=False, linewidth=5, color='blue')
            robot_fill = plt.Circle(
                (x_rob_plt[cnt], y_rob_plt[cnt]), r_rob, fill=True, color='r', alpha=0.3)
            ax.add_patch(robot_radius_plot)
            ax.add_patch(robot_fill)
            # annotate robot
            robot_coord = (round(x_rob_plt[cnt], 2), (round(y_rob_plt[cnt], 2)))
            ax.annotate(f'Robot: {robot_coord}', robot_coord +
                        np.array([0, r_rob]) + annotation_offset,  ha='center')
            # plot robot goal
            goal: np.ndarray = self._set_of_goals[cnt]
            ax.plot(float(goal[1]), float(goal[0]), 'y*', markersize=10)
            # annotate robot goal
            goal_coord = (round(float(goal[1]), 2), round(
                float(goal[0]), 2))
            ax.annotate(f'Goal: {goal_coord}', goal_coord +
                        np.array([0, r_rob]) + annotation_offset,  ha='center')
            # plot robot start
            ax.plot(float(init_state[0]), float(
                init_state[1]), 'yo', markersize=10)
            # annotate robot start
            start_coord = (round(float(init_state[0]), 2), round(
                float(init_state[1]), 2))
            ax.annotate(f'Start: {start_coord}', start_coord -
                        np.array([0, r_rob]) - annotation_offset,  ha='center')
            # plot robot direction arrow
            plt.arrow(x_rob_plt[cnt], y_rob_plt[cnt], np.sin(
                phi_rob_plt[cnt])*r_rob,  np.cos(phi_rob_plt[cnt])*r_rob, color='b', width=r_rob/5)
            # plot predicted robot trajectory
            ax.plot(self.predicted_robot_trajectory[cnt, :, 1], self.predicted_robot_trajectory[cnt, :, 0], linewidth=1, color='blue', linestyle='dashed')

            for i in range(self._total_peds):
                plot_pedestrian(x_peds_plt[:, i], y_peds_plt[:, i], cnt, i)

            # legend
            ax.set_xlabel('$y$ [m]')
            ax.set_ylabel('$x$ [m]')
            ax.legend()
            # increment counter
            cnt = cnt + 1

        print("make_animation: Start")
        frames = len(x_rob_plt)-2
        anim = FuncAnimation(fig, animate, frames=frames, interval=dt, repeat=False)
        anim.save(result_path, 'pillow', frames)
        print("make_animation: Done")