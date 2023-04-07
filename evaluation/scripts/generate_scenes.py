import yaml
import numpy as np
import random
import pathlib
import matplotlib.pyplot as plt
from typing import List

PEDESTRIAN_RANGE = [7]
TOTAL_SCENARIOS = 3
ROBOT_VISION_RANGE = 5
INFLATION_RADIUS = 0.8 # robot_radius + pedestrian_radius + safe_distances
BORDERS_X = [-4, 4]
BORDERS_Y = [-4, 4]
DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

def visualize_scenario(pedestrians_initial_positions: np.ndarray, 
                       pedestrians_goal_positions: np.ndarray,
                       robot_initial_position: np.ndarray,
                       robot_goal_position: np.ndarray) -> None:
    """ Plot scenario

    Args:
        pedestrians_initial_positions (np.ndarray): pedestrians initial positions
        pedestrians_goal_positions (np.ndarray): pedestrians goal positions
        robot_initial_position (np.ndarray): robot initial position
        robot_goal_position (np.ndarray): robot goal position
    """
    fig, ax = plt.subplots(figsize=[16, 16], facecolor='white')

    total_peds = pedestrians_goal_positions.shape[0]
    # Plot pedestrians
    for i in range(total_peds):
        ax.scatter(pedestrians_initial_positions[i,0], pedestrians_initial_positions[i, 1], marker='*', s=80, color=DEFAULT_COLOR_HEX_PALETTE[i])
        ax.scatter(pedestrians_goal_positions[i,0], pedestrians_goal_positions[i, 1], color=DEFAULT_COLOR_HEX_PALETTE[i])  # noqa: E501
        ax.plot([pedestrians_initial_positions[i,0], pedestrians_goal_positions[i,0]], [pedestrians_initial_positions[i,1], pedestrians_goal_positions[i,1]], color=DEFAULT_COLOR_HEX_PALETTE[i])
        circle_plot = plt.Circle(pedestrians_initial_positions[i, :2], 0.3, fill=False, linewidth=3, color=DEFAULT_COLOR_HEX_PALETTE[i])
        ax.add_patch(circle_plot)
        circle_plot = plt.Circle(pedestrians_goal_positions[i, :2], 0.3, fill=False, linewidth=3, color=DEFAULT_COLOR_HEX_PALETTE[i])
        ax.add_patch(circle_plot)
    # Plot robot
    ax.scatter(robot_initial_position[0], robot_initial_position[1], marker='*', s=80, color="blue", label="Robot start position")
    ax.scatter(robot_goal_position[0], robot_goal_position[1], color="blue")
    ax.plot([robot_initial_position[0], robot_goal_position[0]], [robot_initial_position[1], robot_goal_position[1]], 'b')
    circle_plot = plt.Circle(robot_initial_position[:2], 0.3, fill=False, linewidth=3, color="blue")
    ax.add_patch(circle_plot)
    circle_plot = plt.Circle(robot_goal_position[:2], 0.3, fill=False, linewidth=3, color="blue")
    ax.add_patch(circle_plot)
    ax.set_xlabel("x, [m]")
    ax.set_ylabel("y, [m]")
    ax.set_title("Scenario Generator")
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def is_point_belongs_circle(point: np.ndarray,
                            circle_center: np.ndarray,
                            circle_radius: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(point[:2] - circle_center[:2])
    return dist <= circle_radius

def sample_robot_final_position(robot_initial_position: np.ndarray,
                                pedestrians_positions: np.ndarray,
                                robot_vision_range: float = ROBOT_VISION_RANGE,
                                borders_x: List[float] = BORDERS_X,
                                borders_y: List[float] = BORDERS_Y,
                                inflation_radius: float = INFLATION_RADIUS) -> np.ndarray:
    """Sample robot final position in vision radius

    Args:
        robot_initial_position (np.ndarray): robot initial position
        pedestrians_positions: pedestrians positions

    Returns:
        np.ndarray: robot final position
    """
    cnt = 0
    limit = 10000
    while cnt <= limit:
        cnt += 1
        # Polar
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.85 * robot_vision_range, 2*robot_vision_range)
        # Cartesian
        delta_x = r * np.cos(phi)
        delta_y = r * np.sin(phi)
        robot_goal_position = np.array([robot_initial_position[0] + delta_x, 
                                        robot_initial_position[1] + delta_y, 
                                        0])
        # Check if position is within the borders
        if borders_x[0] <= robot_goal_position[0] <= borders_x[1] and \
           borders_y[0] <= robot_goal_position[1] <= borders_y[1]:
            pass
        else:
            continue
        # Check if position i is not intersect with pedestrians
        for position in pedestrians_positions:
            resample_flag = is_point_belongs_circle(robot_goal_position[:2],
                                                    position[:2],
                                                    inflation_radius*2.5)
            if resample_flag:
                break
        if not resample_flag:
            break
    else:
        print("Generation limit was reached!")
        return False
    return robot_goal_position

def generate_circular_scenarios(folder_name,
                                pedestrian_range: List[int],
                                total_scenarios: int,
                                robot_vision_range: float,
                                inflation_radius: float,
                                borders_x: List[int],
                                borders_y: List[int],
                                visualization: bool = False,
                                save_config: bool = True) -> None:
    """ Circular crossing scenario
    """
    with open(f"evaluation/studies/{folder_name}/configs/scenes/scene_config.yaml") as f:
        config = yaml.safe_load(f)

    pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/circular_crossing").mkdir(parents=True, exist_ok=True)

    min_circle_rad = 3.8
    max_circle_rad = 4.2

    def generate_position(pedestrians_initial_positions: np.ndarray,
                          pedestrians_goal_positions: np.ndarray,
                          robot: bool = False) -> np.ndarray:
        while True:
            # Polar
            phi = np.random.uniform(0, 2 * np.pi)
            r = np.random.uniform(min_circle_rad, max_circle_rad)
            # Cartesian
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            if robot:
                x = np.random.uniform(*borders_x)
                y = np.random.uniform(*borders_y)
                phi = np.random.uniform(-np.pi, np.pi)
            position = np.array([x, y, phi + np.pi])
            # Check if position i is not intersect with pedestrians
            saved_positions = np.vstack([pedestrians_initial_positions, 
                                         pedestrians_goal_positions])
            for pos in saved_positions:
                resample_flag = is_point_belongs_circle(position[:2],
                                                        pos[:2],
                                                        1.0) #inflation_radius)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in pedestrian_range:
        pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/circular_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(total_scenarios):
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            pedestrians_goal_positions = np.zeros([total_peds, 3])
            # Sample pedestrians
            for i in range(total_peds):
                position = generate_position(pedestrians_initial_positions,
                                             pedestrians_goal_positions)
                # Add pedestrian position
                pedestrians_initial_positions[i] = position
                pedestrians_goal_positions[i] = position * (-1)
            # Sample robot
            robot_goal_position = False
            while type(robot_goal_position) == bool:
                position = generate_position(pedestrians_initial_positions,
                                            pedestrians_goal_positions,
                                            robot=True)
                robot_initial_position = position
                robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                                    np.vstack([pedestrians_initial_positions,
                                                                                pedestrians_goal_positions]),
                                                                    robot_vision_range=robot_vision_range,
                                                                    borders_x=borders_x,
                                                                    borders_y=borders_y,
                                                                    inflation_radius=inflation_radius)
            if visualization:
                # Plot points
                visualize_scenario(pedestrians_initial_positions,
                                pedestrians_goal_positions,
                                robot_initial_position,
                                robot_goal_position)
            
            if save_config:
                config["init_state"] = robot_initial_position.tolist()
                config["goal"] = robot_goal_position.tolist()
                config["total_peds"] = total_peds
                config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
                config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

                with open(f'evaluation/studies/{folder_name}/configs/scenes/circular_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)
    
def generate_random_scenarios(folder_name,
                                pedestrian_range: List[int],
                                total_scenarios: int,
                                robot_vision_range: float,
                                inflation_radius: float,
                                borders_x: List[int],
                                borders_y: List[int],
                                visualization: bool = False,
                                save_config: bool = True) -> None:
    """ Random crossing scenario
    """
    with open(f"evaluation/studies/{folder_name}/configs/scenes/scene_config.yaml") as f:
        config = yaml.safe_load(f)

    pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/random_crossing").mkdir(parents=True, exist_ok=True)

    def generate_position(pedestrians_initial_positions: np.ndarray,
                          pedestrians_goal_positions: np.ndarray,
                          robot: bool = False) -> np.ndarray:
        inf_rad = inflation_radius*2 if robot else inflation_radius   
        while True:
            # Cartesian
            if robot:
                pedestrian_center_mass_initial_positions = np.mean(pedestrians_initial_positions, axis=1)
                x0 = pedestrian_center_mass_initial_positions[0]
                y0 = pedestrian_center_mass_initial_positions[1]
                L = 3 # Length of sampling
                bord_x = [x0 - L, x0 + L]
                bord_y = [y0 - L, y0 + L]
                x = np.random.uniform(*bord_x)
                y = np.random.uniform(*bord_y)
            else:
                x = np.random.uniform(*borders_x)
                y = np.random.uniform(*borders_y)
            phi = np.random.uniform(-np.pi, np.pi)
            position = np.array([x, y, phi])
            # Check if position i is not intersect with pedestrians
            saved_positions = np.vstack([pedestrians_initial_positions, 
                                         pedestrians_goal_positions])
            for pos in saved_positions: 
                resample_flag = is_point_belongs_circle(position[:2],
                                                        pos[:2],
                                                        inf_rad)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in pedestrian_range:
        pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/random_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(total_scenarios):
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            pedestrians_goal_positions = np.zeros([total_peds, 3])
            # Sample pedestrians
            for i in range(total_peds):
                init_position = generate_position(pedestrians_initial_positions,
                                                  pedestrians_goal_positions)
                # Add pedestrian position
                pedestrians_initial_positions[i] = init_position
                goal_position = generate_position(pedestrians_initial_positions,
                                                  pedestrians_goal_positions)
                pedestrians_goal_positions[i] = goal_position
            # Sample robot
            robot_goal_position = False
            while type(robot_goal_position) == bool:
                position = generate_position(pedestrians_initial_positions,
                                            pedestrians_goal_positions,
                                            robot=True)
                robot_initial_position = position
                robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                                np.vstack([pedestrians_initial_positions,
                                                                            pedestrians_goal_positions]),
                                                                robot_vision_range=robot_vision_range,
                                                                borders_x=borders_x,
                                                                borders_y=borders_y,
                                                                inflation_radius=inflation_radius)
            if visualization:
                # Plot points
                visualize_scenario(pedestrians_initial_positions,
                                pedestrians_goal_positions,
                                robot_initial_position,
                                robot_goal_position)
                
            if save_config:
                config["init_state"] = robot_initial_position.tolist()
                config["goal"] = robot_goal_position.tolist()
                config["total_peds"] = total_peds
                config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
                config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

                with open(fr'evaluation/studies/{folder_name}/configs/scenes/random_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)
                
def generate_parallel_scenarios(folder_name,
                                pedestrian_range: List[int],
                                total_scenarios: int,
                                robot_vision_range: float,
                                inflation_radius: float,
                                borders_x: List[int],
                                borders_y: List[int],
                                visualization: bool = False,
                                save_config: bool = True)-> None:
    """ Parallel crossing scenario
    """

    with open(f"evaluation/studies/{folder_name}/configs/scenes/scene_config.yaml") as f:
        config = yaml.safe_load(f)
        
    pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/parallel_crossing").mkdir(parents=True, exist_ok=True)

    range_x = [-4, -3]
    range_y = [-4, 4]

    def generate_position(pedestrians_initial_positions: np.ndarray,
                          pedestrians_goal_positions: np.ndarray,
                          robot: bool = False) -> np.ndarray:
        while True:
            # Cartesian
            sign = np.random.choice([1, -1])
            x = sign * np.random.uniform(*range_x)
            y = np.random.uniform(*range_y)
            if robot:
                x = np.random.uniform(*[-0.5, 0.5])
            phi = np.random.uniform(-np.pi, np.pi)
            position = np.array([x, y, phi])
            # Check if position i is not intersect with pedestrians
            saved_positions = np.vstack([pedestrians_initial_positions, 
                                         pedestrians_goal_positions])
            for pos in saved_positions:
                resample_flag = is_point_belongs_circle(position[:2],
                                                        pos[:2],
                                                        inflation_radius*0.95)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in pedestrian_range:
        pathlib.Path(f"evaluation/studies/{folder_name}/configs/scenes/parallel_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(total_scenarios):
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            pedestrians_goal_positions = np.zeros([total_peds, 3])
            # Sample pedestrians
            for i in range(total_peds):
                init_position = generate_position(pedestrians_initial_positions,
                                                  pedestrians_goal_positions)
                # Add pedestrian position
                pedestrians_initial_positions[i] = init_position
                goal_position = np.array([init_position[0] * (-1), init_position[1], init_position[2]])
                pedestrians_goal_positions[i] = goal_position
            # Sample robot
            robot_goal_position = False
            while type(robot_goal_position) == bool:
                position = generate_position(pedestrians_initial_positions,
                                            pedestrians_goal_positions,
                                            True)
                robot_initial_position = position
                robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                                np.vstack([pedestrians_initial_positions,
                                                                            pedestrians_goal_positions]),
                                                                robot_vision_range=robot_vision_range,
                                                                borders_x=borders_x,
                                                                borders_y=borders_y,
                                                                inflation_radius=inflation_radius)
            if visualization:
                # Plot points
                visualize_scenario(pedestrians_initial_positions,
                                pedestrians_goal_positions,
                                robot_initial_position,
                                robot_goal_position)
            
            if save_config:
                config["init_state"] = robot_initial_position.tolist()
                config["goal"] = robot_goal_position.tolist()
                config["total_peds"] = total_peds
                config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
                config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

                with open(fr'evaluation/studies/{folder_name}/configs/scenes/parallel_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)



def generate_scenes(folder_name: str,
                    scenes_list: List[str],
                    pedestrian_range: List[int] = PEDESTRIAN_RANGE,
                    total_scenarios: int = TOTAL_SCENARIOS,
                    robot_vision_range: float = ROBOT_VISION_RANGE,
                    inflation_radius: float = INFLATION_RADIUS,
                    borders_x: List[int] = BORDERS_X,
                    borders_y: List[int] = BORDERS_Y,
                    seed: int = 42) -> None:

    # Set random seed
    np.random.seed(seed)
    random.seed(seed)

    if "circular_crossing" in scenes_list:
        generate_circular_scenarios(folder_name,
                                    pedestrian_range,
                                    total_scenarios,
                                    robot_vision_range,
                                    inflation_radius,
                                    borders_x,
                                    borders_y,
                                    visualization=False,
                                    save_config=True)
    
    if "random_crossing" in scenes_list:
        generate_random_scenarios(folder_name,
                                    pedestrian_range,
                                    total_scenarios,
                                    robot_vision_range,
                                    inflation_radius,
                                    borders_x,
                                    borders_y,
                                    visualization=False,
                                    save_config=True)
    
    if "parallel_crossing" in scenes_list:
        generate_parallel_scenarios(folder_name,
                                    pedestrian_range,
                                    total_scenarios,
                                    robot_vision_range,
                                    inflation_radius,
                                    borders_x,
                                    borders_y,
                                    visualization=False,
                                    save_config=True)



if __name__ == "__main__":
    generate_scenes()

