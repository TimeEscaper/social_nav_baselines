from core.controllers import DoMPCController
from core.predictors import ConstantVelocityPredictor
from core.utils import create_sim
from core.visualizer import Visualizer
import numpy as np
import fire
import yaml
import pygame
import pathlib

pathlib.Path(r"results").mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG_PATH = r"configs/mpc_config.yaml"
DEFAULT_RESULT_PATH = r"results/mpc.gif"

def main(config_path: str = DEFAULT_CONFIG_PATH) -> None:

    # Initialization
    with open(config_path) as f:
        config = yaml.safe_load(f)

    simulator, renderer = create_sim(np.array(config["init_state"]),
                                     config["model_type"],
                                     config["total_peds"],
                                     config["is_robot_visible"],
                                     config["ped_dect_range"],
                                     np.deg2rad(config["ped_dect_fov"]),
                                     config["misdetection_prob"],
                                     config["dt"] / 10,
                                     config["waypoint_tracker"],
                                     config["pedestrians_init_states"],
                                     config["pedestrians_goals"])
    renderer.initialize()
    if config["ped_predictor"] == "constant_velocity":
        if config["total_peds"] > 0:
            predictor = ConstantVelocityPredictor(config["dt"],
                                                  config["total_peds"],
                                                  config["horizon"])
        elif config["total_peds"] == 0:
            predictor = ConstantVelocityPredictor(config["dt"],
                                                  1,
                                                  config["horizon"])
    elif config["ped_predictor"] == "neural":
        pass
    controller = DoMPCController(np.array(config["init_state"]),
                                 np.array(config["goal"]),
                                 config["horizon"],
                                 config["dt"],
                                 config["model_type"],
                                 config["total_peds"],
                                 np.array(config["Q"]),
                                 np.array(config["R"]),
                                 config["lb"],
                                 config["ub"],
                                 config["r_rob"],
                                 config["r_ped"],
                                 config["min_safe_dist"],
                                 predictor,
                                 config["is_store_robot_predicted_trajectory"],
                                 config["max_ghost_tracking_time"],
                                 config["state_dummy_ped"],
                                 config["solver"])
    visualizer = Visualizer(config["total_peds"],
                            renderer)
    visualizer.visualize_goal(config["goal"])
    
    # Loop
    simulator.step()
    hold_time = simulator.sim_dt
    state = config["init_state"]
    control = np.array([0, 0]) 

    while True:
        renderer.render()
        if hold_time >= controller.dt:
            error = np.linalg.norm(controller.goal[:2] - state[:2])
            if error >= config["tollerence_error"]:
                state = simulator.current_state.world.robot.state
                visualizer.append_ground_truth_robot_state(state)
                if config["total_peds"] > 0:
                    detected_pedestrian_indices = simulator.current_state.sensors['pedestrian_detector'].reading.pedestrians.keys()
                    undetected_pedestrian_indices = list(set(range(config["total_peds"])) - set(detected_pedestrian_indices))
                    ground_truth_pedestrians_state = np.concatenate([simulator.current_state.world.pedestrians.poses[:, :2],
                                                                     simulator.current_state.world.pedestrians.velocities[:, :2]], axis=1)
                    ground_truth_pedestrians_state[undetected_pedestrian_indices] = config["state_dummy_ped"]
                    controller.previously_detected_pedestrians |= set(detected_pedestrian_indices)
                    if undetected_pedestrian_indices:
                        pedestrians_ghosts_states = controller.get_pedestrains_ghosts_states(ground_truth_pedestrians_state,
                                                                                             undetected_pedestrian_indices)
                    
                    visualizer.append_ground_truth_pedestrians_pose(simulator.current_state.world.pedestrians.poses[:, :2])
                elif config["total_peds"] == 0:
                    pedestrians_ghosts_states = np.array([config["state_dummy_ped"]])
                control, predicted_pedestrians_trajectories = controller.make_step(state,
                                                                                   pedestrians_ghosts_states)
                visualizer.append_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])
                visualizer.visualize_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])
                predicted_robot_trajectory = controller.get_predicted_robot_trajectory()      
                visualizer.append_predicted_robot_trajectory(predicted_robot_trajectory)
                visualizer.visualize_predicted_robot_trajectory(predicted_robot_trajectory)                      
                hold_time = 0.
            else:
                simulator.step(np.array([0, 0]))
                print("The goal", *["{0:0.2f}".format(i) for i in controller.goal], "was reached!")
                input_value = input("Enter new goal in format '0 0 0' or type 'exit' to exit: ")
                if input_value == "exit":
                    break
                new_goal = np.array(list(map(int, input_value.split())))
                visualizer.visualize_goal(new_goal)
                controller.set_new_goal(state,
                                        new_goal)
        simulator.step(control)
        hold_time += simulator.sim_dt
    pygame.quit()

    visualizer.make_animation("Model Predictive Control",
                              DEFAULT_RESULT_PATH,
                              config)

if __name__ == "__main__":
    fire.Fire(main)