from core.controllers import ControllerFactory
from core.predictors import PredictorFactory, PedestrianTracker
from core.planners import RandomPlanner
from core.utils import create_sim
from core.visualizer import Visualizer
from core.statistics import Statistics
import numpy as np
import fire
import yaml
import pygame
import pathlib
import random
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

pathlib.Path(r"results").mkdir(parents=True, exist_ok=True)

DEFAULT_SCENE_CONFIG_PATH = r"evaluation/scenes/circular_crossing/4/0.yaml"
DEFAULT_CONTROLLER_CONFIG_PATH = r"evaluation/controllers/MD-MPC.yaml"
DEFAULT_RESULT_PATH = r"results/mpc.gif"

def main(scene_config_path: str = DEFAULT_SCENE_CONFIG_PATH,
         controller_config_path: str = DEFAULT_CONTROLLER_CONFIG_PATH,
         result_path: str = DEFAULT_RESULT_PATH) -> Statistics:

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
                                     create_renderer = True)
    
    predictor = PredictorFactory.create_predictor(config)

    ped_tracker = PedestrianTracker(predictor, 
                                    config["max_ghost_tracking_time"], 
                                    False)

    controller = ControllerFactory.create_controller(config,
                                                     predictor,
                                                     ped_tracker)

    statistics = Statistics(simulator,
                            scene_config_path,
                            controller_config_path)

    planner = RandomPlanner(global_goal=np.array(config["goal"]),
                            controller=controller,
                            subgoal_reach_threshold=config["tolerance_error"],
                            subgoal_to_goal_threshold=2.,
                            pedestrian_tracker=ped_tracker,
                            statistics_module=statistics)

    visualizer = Visualizer(config["total_peds"],
                            renderer)
    visualizer.visualize_goal(config["goal"])

    # Loop
    simulator.step()
    hold_time = simulator.sim_dt
    state = config["init_state"]
    control = np.array([0, 0]) 

    while True:
        # if simulation time exceeded
        if statistics.simulation_ticks >= 2000:
            statistics.set_failure_flag()
            break
        if renderer:
            renderer.render()
        if hold_time >= controller.dt:
            error = np.linalg.norm(planner.global_goal[:2] - state[:2])
            if error >= config["tolerance_error"]:
                state = simulator.current_state.world.robot.state
                detected_peds_keys = simulator.current_state.sensors["pedestrian_detector"].reading.pedestrians.keys()
                ground_truth_pedestrians_state = np.concatenate([simulator.current_state.world.pedestrians.poses[:, :2],
                                                                 simulator.current_state.world.pedestrians.velocities[:, :2]], axis=1)
                observation = {k: ground_truth_pedestrians_state[k, :] for k in detected_peds_keys}
                control, predicted_pedestrians_trajectories, predicted_pedestrians_covariances = planner.make_step(state,
                                                                                                                      observation)
                predicted_robot_trajectory = controller.get_predicted_robot_trajectory()    
                statistics.append_current_subgoal(planner.current_subgoal)
                visualizer.append_ground_truth_robot_state(state)
                visualizer.append_ground_truth_pedestrians_pose(simulator.current_state.world.pedestrians.poses[:, :2])
                visualizer.append_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])
                  
                visualizer.append_predicted_robot_trajectory(predicted_robot_trajectory)
                visualizer.append_predicted_pedestrians_covariances(predicted_pedestrians_covariances)    
                if renderer:
                    visualizer.visualize_goal(config["goal"])
                    visualizer.visualize_predicted_robot_trajectory(predicted_robot_trajectory)        
                    visualizer.visualize_subgoal(planner.current_subgoal)
                    visualizer.visualize_predicted_pedestrians_trajectory_with_covariances(predicted_pedestrians_trajectories[:, :, :2], predicted_pedestrians_covariances)      
                hold_time = 0.
            else:
                if config["experiment_run"]:
                    break
                else:
                    simulator.step(np.array([0, 0]))
                    print("The goal", *["{0:0.2f}".format(i) for i in controller.goal], "was reached!")
                    input_value = input("Enter new goal in format '0 0 0' or type 'exit' to exit: ")
                    if input_value == "exit":
                        break
                    new_goal = np.array(list(map(int, input_value.split())))
                    visualizer.visualize_goal(new_goal)
                    planner.set_new_global_goal(new_goal)

        simulator.step(control)
        statistics.track_collisions()
        statistics.track_simulation_ticks()
        hold_time += simulator.sim_dt

        if config["experiment_run"] == False:
            # Terminate the simulation anytime by pressing ESC
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
    pygame.quit()

    statistics._ground_truth_pedestrian_trajectories = visualizer._ground_truth_pedestrian_trajectories
    statistics._ground_truth_robot_trajectory = visualizer._ground_truth_robot_trajectory
    statistics._predicted_pedestrians_trajectories = visualizer._predicted_pedestrians_trajectories
    statistics._predicted_robot_trajectory = visualizer._predicted_robot_trajectory
    statistics._predicted_pedestrians_covariances = visualizer._predicted_pedestrians_covariances

    if config["experiment_run"] == False:
        visualizer.make_animation(f"Model Predictive Control", 
                                result_path, 
                                config)
    
    return statistics
    
if __name__ == "__main__":
    fire.Fire(main)
