import yaml
import numpy as np
import random
import matplotlib.pyplot as plt
from pyminisim.pedestrians import RandomWaypointTracker

SEED = 42
np.random.seed(42)
random.seed(42)

with open(r"evaluation/scenes/template_scene_config.yaml") as f:
    config = yaml.safe_load(f)

def generate_circular_scenarios():
    # Circular crossing scenario
    # Circle properites
    min_circle_rad = 3.8
    max_circle_rad = 4.2

    for total_peds in [1, 2, 4, 5, 6, 7]:
        for scenario_id in range(30):
            circle_sectors = np.linspace(0, 2 * np.pi, total_peds + 2) # add sector for the robot
            phis = np.array([np.random.uniform(circle_sectors[i], circle_sectors[i+1]) for i in range(len(circle_sectors) - 1)])
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            for ped in range(total_peds):
                circle_rad = np.random.uniform(min_circle_rad, max_circle_rad)
                pedestrians_initial_positions[ped, 0] = circle_rad * np.cos(phis[ped])  # set x coordinate
                pedestrians_initial_positions[ped, 1] = circle_rad * np.sin(phis[ped])  # set y coordinate
            pedestrians_goal_positions = pedestrians_initial_positions * (-1)

            # add robot sampled position
            circle_rad = np.random.uniform(min_circle_rad, max_circle_rad)
            robot_initial_position = np.array([circle_rad * np.cos(phis[-1]), circle_rad * np.sin(phis[-1]), phis[-1] + 3.14])
            robot_goal_position = robot_initial_position * (-1)
            
            config["init_state"] = robot_initial_position.tolist()
            config["goal"] = robot_goal_position.tolist()
            config["total_peds"] = total_peds
            config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
            config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

            with open(fr'evaluation/scenes/circular_crossing/{total_peds}/{scenario_id}.yaml', 'w') as outfile:
                yaml.dump(config, outfile)

def generate_random_scenarios():
    # Random traffic scenario
    for total_peds in [1, 2, 4, 5, 6, 7]:
        for scenario_id in range(30):
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            pedestrians_initial_positions[:, :2] = RandomWaypointTracker(world_size=(10.0, 15.0)).sample_independent_points(total_peds)
            pedestrians_goal_positions = RandomWaypointTracker(world_size=(10.0, 15.0)).sample_independent_points(total_peds)

            
            config["init_state"] = [-4, 0, 0]
            config["goal"] = [4, 0, 0]
            config["total_peds"] = total_peds
            config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
            config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

            with open(fr'evaluation/scenes/random/{total_peds}/{scenario_id}.yaml', 'w') as outfile:
                yaml.dump(config, outfile)

def generate_parallel_scenarios():
    # Parallel traffic scenario
    for total_peds in [1, 2, 4, 5, 6, 7]:
        for scenario_id in range(30):
            sectors_X_axis = np.linspace(-4, 4, total_peds + 2) # add sector for the robot
            X_coordinates = np.array([np.random.uniform(sectors_X_axis[i], sectors_X_axis[i+1]) for i in range(len(sectors_X_axis) - 1)])
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            for ped in range(total_peds):
                y_coord = np.random.uniform(1, 4)
                side = np.random.choice(["bottom","top"])
                if side == "bottom":
                    y_coord = y_coord * (-1)
                pedestrians_initial_positions[ped, 0] = X_coordinates[ped]
                pedestrians_initial_positions[ped, 1] = y_coord
            
            # add robot position
            robot_initial_position = np.array([X_coordinates[-1], np.random.uniform(-4, -1), 1.57])
            
            # switch robot init position with a random pedestrian
            if total_peds > 1:
                ped_id_to_switch = np.random.randint(1, total_peds)
                temp = robot_initial_position.copy()
                robot_initial_position = pedestrians_initial_positions[ped_id_to_switch].copy()
                pedestrians_initial_positions[ped_id_to_switch] = temp.copy()

            # assign goal states
            robot_goal_position = robot_initial_position.copy()
            robot_goal_position[1] = (robot_initial_position * (-1))[1].copy()
            pedestrians_goal_positions = pedestrians_initial_positions.copy()
            pedestrians_goal_positions[:, 1] = (pedestrians_initial_positions * (-1))[:, 1].copy()

            
            config["init_state"] = robot_initial_position.tolist()
            config["goal"] = robot_goal_position.tolist()
            config["total_peds"] = total_peds
            config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
            config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

            with open(fr'evaluation/scenes/parallel_traffic/{total_peds}/{scenario_id}.yaml', 'w') as outfile:
                yaml.dump(config, outfile)

def generate_perpendicular_scenarios():
    # Parallel traffic scenario
    for total_peds in [1, 2, 4, 5, 6, 7]:
        for scenario_id in range(30):
            sectors_X_axis = np.linspace(-4, 4, total_peds + 1) # add sector for the robot
            X_coordinates = np.array([np.random.uniform(sectors_X_axis[i], sectors_X_axis[i+1]) for i in range(len(sectors_X_axis) - 1)])
            pedestrians_initial_positions = np.zeros([total_peds, 3])
            for ped in range(total_peds):
                y_coord = np.random.uniform(1, 4)
                side = np.random.choice(["bottom","top"])
                if side == "bottom":
                    y_coord = y_coord * (-1)
                pedestrians_initial_positions[ped, 0] = X_coordinates[ped]
                pedestrians_initial_positions[ped, 1] = y_coord
            
            # add robot position
            robot_initial_position = np.array([-4, 0, 0])

            # assign goal states
            robot_goal_position = np.array([4, 0, 0])
            pedestrians_goal_positions = pedestrians_initial_positions.copy()
            pedestrians_goal_positions[:, 1] = (pedestrians_initial_positions * (-1))[:, 1].copy()

            
            config["init_state"] = robot_initial_position.tolist()
            config["goal"] = robot_goal_position.tolist()
            config["total_peds"] = total_peds
            config["pedestrians_init_states"] = pedestrians_initial_positions.tolist()
            config["pedestrians_goals"] = pedestrians_goal_positions[:, :2].reshape(total_peds, 1, 2).tolist()

            with open(fr'evaluation/scenes/perpendicular_traffic/{total_peds}/{scenario_id}.yaml', 'w') as outfile:
                yaml.dump(config, outfile)

generate_circular_scenarios()
generate_random_scenarios()
generate_parallel_scenarios()
generate_perpendicular_scenarios()

"""
print(pedestrians_initial_positions)
print(pedestrians_goal_positions)

plt.scatter(pedestrians_initial_positions[:,0], pedestrians_initial_positions[:, 1])
plt.scatter(pedestrians_goal_positions[:,0], pedestrians_goal_positions[:, 1])
for i in range(total_peds):
    plt.plot([pedestrians_initial_positions[i,0], pedestrians_goal_positions[i,0]], [pedestrians_initial_positions[i,1], pedestrians_goal_positions[i,1]])
plt.plot([robot_initial_position[0], robot_goal_position[0]], [robot_initial_position[1], robot_goal_position[1]], '*')
plt.show()
"""
