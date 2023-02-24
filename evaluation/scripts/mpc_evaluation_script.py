from core.controllers import DoMPCController
from core.predictors import ConstantVelocityPredictor, NeuralPredictor
from core.utils import create_sim
from core.visualizer import Visualizer
from core.statistics import Statistics
import numpy as np
import fire
import yaml
import pygame

def main(scene_config_path: str,
         controller_config_path: str,
         result_path: str = "") -> Statistics:

    # Initialization
    with open(scene_config_path) as f:
        scene_config = yaml.safe_load(f)

    with open(controller_config_path) as f:
        controller_config = yaml.safe_load(f)

    config = {}
    config.update(scene_config)
    config.update(controller_config)

    simulator, renderer = create_sim(np.array(config["init_state"]),
                                     config["model_type"],
                                     config["total_peds"],
                                     config["is_robot_visible"],
                                     config["ped_detc_range"],
                                     np.deg2rad(config["ped_detc_fov"]),
                                     config["misdetection_prob"],
                                     config["dt"] / 10,
                                     config["waypoint_tracker"],
                                     config["pedestrians_init_states"],
                                     config["pedestrians_goals"],
                                     config["ped_model"])
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
        if config["total_peds"] > 0:
            predictor = NeuralPredictor(config["dt"],
                                        config["total_peds"],
                                        config["horizon"])
        elif config["total_peds"] == 0:
            predictor = NeuralPredictor(config["dt"],
                                        1,
                                        config["horizon"])
    controller = DoMPCController(np.array(config["init_state"]),
                                 np.array(config["goal"]),
                                 config["horizon"],
                                 config["dt"],
                                 config["model_type"],
                                 config["total_peds"],
                                 np.array(config["Q"]),
                                 np.array(config["R"]),
                                 config["W"],
                                 config["lb"],
                                 config["ub"],
                                 config["r_rob"],
                                 config["r_ped"],
                                 config["constraint_value"],
                                 predictor,
                                 config["is_store_robot_predicted_trajectory"],
                                 config["max_ghost_tracking_time"],
                                 config["state_dummy_ped"],
                                 config["solver"])
    visualizer = Visualizer(config["total_peds"],
                            renderer)
    visualizer.visualize_goal(config["goal"])
    
    statistics = Statistics(simulator,
                            scene_config_path,
                            controller_config_path)

    # Loop
    simulator.step()
    hold_time = simulator.sim_dt
    state = config["init_state"]
    control = np.array([0, 0]) 

    while statistics.simulation_ticks < 3000:

        # if simulation time exceeded
        if statistics.simulation_ticks >= 3000:
            statistics.set_failure_flag()
            break

        renderer.render()
        if hold_time >= controller.dt:
            error = np.linalg.norm(controller.goal[:2] - state[:2])
            if error >= config["tolerance_error"]:
                state = simulator.current_state.world.robot.state
                visualizer.append_ground_truth_robot_state(state)
                if config["total_peds"] > 0:
                    detected_pedestrian_indices = simulator.current_state.sensors['pedestrian_detector'].reading.pedestrians.keys()
                    undetected_pedestrian_indices = list(set(range(config["total_peds"])) - set(detected_pedestrian_indices))
                    ground_truth_pedestrians_state = np.concatenate([simulator.current_state.world.pedestrians.poses[:, :2],
                                                                     simulator.current_state.world.pedestrians.velocities[:, :2]], axis=1)
                    ground_truth_pedestrians_state[undetected_pedestrian_indices] = config["state_dummy_ped"]
                    controller.previously_detected_pedestrians |= set(detected_pedestrian_indices)
                    pedestrians_ghosts_states = ground_truth_pedestrians_state
                    if undetected_pedestrian_indices:
                        pedestrians_ghosts_states = controller.get_pedestrains_ghosts_states(ground_truth_pedestrians_state,
                                                                                             undetected_pedestrian_indices)
                    
                    visualizer.append_ground_truth_pedestrians_pose(simulator.current_state.world.pedestrians.poses[:, :2])
                elif config["total_peds"] == 0:
                    pedestrians_ghosts_states = np.array([config["state_dummy_ped"]])
                control, predicted_pedestrians_trajectories, predicted_pedestrians_covariances = controller.make_step(state,
                                                                                                                      pedestrians_ghosts_states)        
                hold_time = 0.
            else:
                break
        simulator.step(control)
        statistics.track_collisions()
        statistics.track_simulation_ticks()
        hold_time += simulator.sim_dt
        # Terminate the simulation anytime by pressing ESC
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
    pygame.quit()
    return statistics

if __name__ == "__main__":
    fire.Fire(main)