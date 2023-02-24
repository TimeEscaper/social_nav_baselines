import time

from core.controllers import DDPController
from core.predictors import ConstantVelocityPredictor, NeuralPredictor
from core.utils import create_sim
from core.visualizer import Visualizer
import numpy as np
import fire
import yaml
import pygame
import pathlib

from pyminisim.visual import Covariance2dDrawing

pathlib.Path(r"results").mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG_PATH = r"configs/ddp_config.yaml"
DEFAULT_RESULT_PATH = r"results/ddp.gif"


def main(config_path: str = DEFAULT_CONFIG_PATH) -> None:
    # Initialization
    with open(config_path) as f:
        config = yaml.safe_load(f)

    simulator, renderer = create_sim(np.array(config["init_state"]),
                                     config["model_type"],
                                     config["total_peds"],
                                     config["is_robot_visible"],
                                     config["ped_detc_range"],
                                     np.deg2rad(config["ped_detc_fov"]),
                                     config["misdetection_prob"],
                                     0.01,
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
    controller = DDPController(init_state=np.array(config["init_state"]),
                               goal=np.array(config["goal"]),
                               horizon=config["horizon"],
                               dt=config["dt"],
                               model_type=config["model_type"],
                               total_peds=config["total_peds"],
                               Q=np.array(config["Q"]),
                               R=np.array(config["R"]),
                               q_b=config["q_b"],
                               q_m=config["q_m"],
                               r_rob=config["r_rob"],
                               r_ped=config["r_ped"],
                               n_iterations=config["n_iterations"],
                               constraint_value=config["constraint_value"],
                               predictor=predictor,
                               is_store_robot_predicted_trajectory=config["is_store_robot_predicted_trajectory"],
                               max_ghost_tracking_time=config["max_ghost_tracking_time"],
                               state_dummy_ped=config["state_dummy_ped"])
    visualizer = Visualizer(config["total_peds"],
                            renderer)
    visualizer.visualize_goal(config["goal"])

    # Loop
    simulator.step()
    hold_time = simulator.sim_dt
    state = config["init_state"]
    control = np.array([0, 0])

    time.sleep(5)

    while True:
        renderer.render()
        if hold_time >= controller.dt:
            error = np.linalg.norm(controller.goal[:2] - state[:2])
            if error >= config["tollerance_error"]:
                state = simulator.current_state.world.robot.state
                visualizer.append_ground_truth_robot_state(state)
                if config["total_peds"] > 0:
                    detected_pedestrian_indices = simulator.current_state.sensors[
                        'pedestrian_detector'].reading.pedestrians.keys()
                    undetected_pedestrian_indices = list(
                        set(range(config["total_peds"])) - set(detected_pedestrian_indices))
                    ground_truth_pedestrians_state = np.concatenate(
                        [simulator.current_state.world.pedestrians.poses[:, :2],
                         simulator.current_state.world.pedestrians.velocities[:, :2]], axis=1)
                    ground_truth_pedestrians_state[undetected_pedestrian_indices] = config["state_dummy_ped"]
                    controller.previously_detected_pedestrians |= set(detected_pedestrian_indices)
                    pedestrians_ghosts_states = ground_truth_pedestrians_state
                    if undetected_pedestrian_indices:
                        pedestrians_ghosts_states = controller.get_pedestrains_ghosts_states(
                            ground_truth_pedestrians_state,
                            undetected_pedestrian_indices)

                    visualizer.append_ground_truth_pedestrians_pose(
                        simulator.current_state.world.pedestrians.poses[:, :2])
                elif config["total_peds"] == 0:
                    pedestrians_ghosts_states = np.array([config["state_dummy_ped"]])
                control, predicted_pedestrians_trajectories, predicted_pedestrians_covariances = controller.make_step(
                    state,
                    pedestrians_ghosts_states)
                for i in range(predicted_pedestrians_covariances.shape[0]):
                    for j in range(predicted_pedestrians_covariances.shape[1]):
                        cov = predicted_pedestrians_covariances[i, j]
                        renderer.draw(f"cov_{i}_{j}", Covariance2dDrawing(predicted_pedestrians_trajectories[i, j, :2],
                                                                          cov,
                                                                          (255, 0, 0),
                                                                          0.1))

                visualizer.append_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])
                visualizer.visualize_predicted_pedestrians_trajectories(predicted_pedestrians_trajectories[:, :, :2])
                # visualizer.visualize_predicted_pedestrians_trajectory_with_covariances(predicted_pedestrians_trajectories[:, :, :2], predicted_pedestrians_covariances)
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
        # Terminate the simulation anytime by pressing ESC
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
    pygame.quit()

    visualizer.make_animation("DDP",
                              DEFAULT_RESULT_PATH,
                              config)


if __name__ == "__main__":
    fire.Fire(main)
