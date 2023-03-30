import yaml
import numpy as np
import random
import pathlib
import matplotlib.pyplot as plt

PEDESTRIAN_RANGE = [3, 4, 5, 6]
TOTAL_SCENARIOS = 100
ROBOT_VISION_RANGE = 5
INFLATION_RADIUS = 1.0 # robot_radius + pedestrian_radius + safe_distances
BORDERS_X = [-5, 5]
BORDERS_Y = [-5, 5]

SEED = 42
np.random.seed(42)
random.seed(42)

DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

with open(r"evaluation/scenes/template_scene_config.yaml") as f:
    config = yaml.safe_load(f)

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
    #print(pedestrians_initial_positions)
    #print(pedestrians_goal_positions)
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
                                pedestrians_positions: np.ndarray) -> np.ndarray:
    """Sample robot final position in vision radius

    Args:
        robot_initial_position (np.ndarray): robot initial position
        pedestrians_positions: pedestrians positions

    Returns:
        np.ndarray: robot final position
    """
    cnt = 0
    limit = 200000
    while cnt <= limit:
        cnt += 1
        # Polar
        phi = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0.5 * ROBOT_VISION_RANGE, ROBOT_VISION_RANGE)
        # Cartesian
        delta_x = r * np.cos(phi)
        delta_y = r * np.sin(phi)
        robot_goal_position = np.array([robot_initial_position[0] + delta_x, 
                                        robot_initial_position[1] + delta_y, 
                                        0])
        # Check if position is within the borders
        if BORDERS_X[0] <= robot_goal_position[0] <= BORDERS_X[1] and \
           BORDERS_Y[0] <= robot_goal_position[1] <= BORDERS_Y[1]:
            pass
        else:
            continue
        # Check if position i is not intersect with pedestrians
        for position in pedestrians_positions:
            resample_flag = is_point_belongs_circle(robot_goal_position[:2],
                                                    position[:2],
                                                    INFLATION_RADIUS*2.5)
            if resample_flag:
                break
        if not resample_flag:
            break
    else:
        print("Generation limit was reached!")
    return robot_goal_position

def generate_circular_scenarios(vizualization: bool = False,
                                save_config: bool = True) -> None:
    """ Circular crossing scenario
    """
    pathlib.Path("evaluation/scenes/circular_crossing").mkdir(parents=True, exist_ok=True)

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
                x = np.random.uniform(*BORDERS_X)
                y = np.random.uniform(*BORDERS_Y)
                phi = np.random.uniform(-np.pi, np.pi)
            position = np.array([x, y, phi + np.pi])
            # Check if position i is not intersect with pedestrians
            saved_positions = np.vstack([pedestrians_initial_positions, 
                                         pedestrians_goal_positions])
            for pos in saved_positions:
                resample_flag = is_point_belongs_circle(position[:2],
                                                        pos[:2],
                                                        INFLATION_RADIUS)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in PEDESTRIAN_RANGE:
        pathlib.Path(f"evaluation/scenes/circular_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(TOTAL_SCENARIOS):
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
            position = generate_position(pedestrians_initial_positions,
                                         pedestrians_goal_positions,
                                         robot=True)
            robot_initial_position = position
            robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                              np.vstack([pedestrians_initial_positions,
                                                                         pedestrians_goal_positions]))
            if vizualization:
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

                with open(fr'evaluation/scenes/circular_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)
    
def generate_random_scenarios(vizualization: bool = False,
                                save_config: bool = True) -> None:
    """ Random crossing scenario
    """
    pathlib.Path("evaluation/scenes/random_crossing").mkdir(parents=True, exist_ok=True)

    def generate_position(pedestrians_initial_positions: np.ndarray,
                          pedestrians_goal_positions: np.ndarray,
                          robot: bool = False) -> np.ndarray:
        inflation_radius = INFLATION_RADIUS*2 if robot else INFLATION_RADIUS
        while True:
            # Cartesian
            x = np.random.uniform(*BORDERS_X)
            y = np.random.uniform(*BORDERS_Y)
            phi = np.random.uniform(-np.pi, np.pi)
            position = np.array([x, y, phi])
            # Check if position i is not intersect with pedestrians
            saved_positions = np.vstack([pedestrians_initial_positions, 
                                         pedestrians_goal_positions])
            for pos in saved_positions: 
                resample_flag = is_point_belongs_circle(position[:2],
                                                        pos[:2],
                                                        inflation_radius)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in PEDESTRIAN_RANGE:
        pathlib.Path(f"evaluation/scenes/random_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(TOTAL_SCENARIOS):
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
            position = generate_position(pedestrians_initial_positions,
                                         pedestrians_goal_positions,
                                         robot=True)
            robot_initial_position = position
            robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                              np.vstack([pedestrians_initial_positions,
                                                                         pedestrians_goal_positions]))
            if vizualization:
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

                with open(fr'evaluation/scenes/random_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)
                
def generate_parallel_scenarios(vizualization: bool = False,
                                  save_config: bool = True)-> None:
    """ Parallel crossing scenario
    """
    pathlib.Path("evaluation/scenes/parallel_crossing").mkdir(parents=True, exist_ok=True)

    range_x = [-4, -3]
    range_y = [-4, 4]

    def generate_position(pedestrians_initial_positions: np.ndarray,
                          pedestrians_goal_positions: np.ndarray,
                          robot: bool = False) -> np.ndarray:
        while True:
            # Cartesian
            x = np.random.uniform(*range_x)
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
                                                        INFLATION_RADIUS)
                if resample_flag:
                    break
            if not resample_flag:
                break
        return position

    for total_peds in PEDESTRIAN_RANGE:
        pathlib.Path(f"evaluation/scenes/parallel_crossing/{total_peds}").mkdir(parents=True, exist_ok=True)
        for id in range(TOTAL_SCENARIOS):
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
            position = generate_position(pedestrians_initial_positions,
                                         pedestrians_goal_positions,
                                         True)
            robot_initial_position = position
            robot_goal_position = sample_robot_final_position(robot_initial_position,
                                                              np.vstack([pedestrians_initial_positions,
                                                                         pedestrians_goal_positions]))
            if vizualization:
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

                with open(fr'evaluation/scenes/parallel_crossing/{total_peds}/{id}.yaml', 'w') as outfile:
                    yaml.dump(config, outfile)

def main():
    generate_circular_scenarios(False)
    generate_random_scenarios(False)
    generate_parallel_scenarios(False)

if __name__ == "__main__":
    main()

