from core.controllers import DoMPCController
from core.predictors import ConstantVelocityPredictor, NeuralPredictor, PedestrianTracker
from core.planners import RandomPlanner
from core.utils import create_sim
from core.visualizer import Visualizer
from core.statistics import Statistics
import numpy as np
import fire
import yaml
import pygame
import pathlib

def main(scene_config_path: str = r"evaluation/scenes/random/7/0.yaml",
         controller_config_path: str = r"evaluation/controllers/MD-MPC-EDC.yaml",
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
                                     config["ped_model"],
                                     False)
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
            predictor = NeuralPredictor(total_peds=config["total_peds"],
                                        horizon=config["horizon"])
        elif config["total_peds"] == 0:
            predictor = NeuralPredictor(total_peds=1,
                                        horizon=config["horizon"])

    ped_tracker = PedestrianTracker(predictor, config["max_ghost_tracking_time"], False)

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
                                 ped_tracker,
                                 config["is_store_robot_predicted_trajectory"],
                                 config["max_ghost_tracking_time"],
                                 config["state_dummy_ped"],
                                 config["solver"],
                                 config["cost_function"],
                                 config["constraint_type"],
                                 config["constraint_value"])

    statistics = Statistics(simulator,
                            scene_config_path,
                            controller_config_path)

    planner = RandomPlanner(global_goal=np.array(config["goal"]),
                            controller=controller,
                            subgoal_reach_threshold=config["tolerance_error"],
                            subgoal_to_goal_threshold=2.,
                            pedestrian_tracker=ped_tracker,
                            statistics_module=statistics)

    visualizer = Visualizer(config["total_peds"])
    visualizer.visualize_goal(config["goal"])

    # Loop
    simulator.step()
    hold_time = simulator.sim_dt
    state = config["init_state"]
    control = np.array([0, 0]) 

    while statistics.simulation_ticks <= 2000:

        # if simulation time exceeded
        if statistics.simulation_ticks == 2000:
            statistics.set_failure_flag()
            break
        
        #renderer.render()
        if hold_time >= controller.dt:
            error = np.linalg.norm(planner.global_goal[:2] - state[:2])
            if error >= config["tolerance_error"]:
                state = simulator.current_state.world.robot.state

                detected_peds_keys = simulator.current_state.sensors["pedestrian_detector"].reading.pedestrians.keys()
                ground_truth_pedestrians_state = np.concatenate([simulator.current_state.world.pedestrians.poses[:, :2],
                                                                 simulator.current_state.world.pedestrians.velocities[:, :2]], axis=1)
                observation = {k: ground_truth_pedestrians_state[k, :] for k in detected_peds_keys}

                visualizer.append_ground_truth_robot_state(state)
                visualizer.append_ground_truth_pedestrians_pose(simulator.current_state.world.pedestrians.poses[:, :2])

                control, predicted_pedestrians_trajectories, predicted_pedestrians_covariances = planner.make_step(state,
                                                                                                                      observation)
                statistics.append_current_subgoal(planner.current_subgoal)
                visualizer.append_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])

                predicted_robot_trajectory = controller.get_predicted_robot_trajectory()      
                visualizer.append_predicted_robot_trajectory(predicted_robot_trajectory)         
                visualizer.append_predicted_pedestrians_covariances(predicted_pedestrians_covariances)          
                hold_time = 0.
            else:
                break

        simulator.step(control)
        statistics.track_collisions()
        statistics.track_simulation_ticks()
        hold_time += simulator.sim_dt
    pygame.quit()

    statistics._ground_truth_pedestrian_trajectories = visualizer._ground_truth_pedestrian_trajectories
    statistics._ground_truth_robot_trajectory = visualizer._ground_truth_robot_trajectory
    statistics._predicted_pedestrians_trajectories = visualizer._predicted_pedestrians_trajectories
    statistics._predicted_robot_trajectory = visualizer._predicted_robot_trajectory
    statistics._predicted_pedestrians_covariances = visualizer._predicted_pedestrians_covariances
    
    return statistics 

if __name__ == "__main__":
    fire.Fire(main)
