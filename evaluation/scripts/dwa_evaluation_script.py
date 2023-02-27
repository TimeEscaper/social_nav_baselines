from core.controllers import DWAController
from core.predictors import ConstantVelocityPredictor, NeuralPredictor
from core.utils import create_sim
from core.visualizer import Visualizer
from core.statistics import Statistics
import numpy as np
import fire
import yaml
import pathlib

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

    simulator, _ = create_sim(np.array(config["init_state"]),
                                     "unicycle",
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
                                     create_renderer=False)
    #renderer.initialize()
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
    controller = DWAController(np.array(config["init_state"]),
                               np.array(config["goal"]),
                               config["dt"],
                               config["horizon"],
                               predictor,
                               config["cost_function"],
                               config["v_res"],
                               config["w_res"],
                               config["lb"],
                               config["ub"],
                               config["weights"],
                               config["total_peds"],
                               config["state_dummy_ped"],
                               config["max_ghost_tracking_time"])

    statistics = Statistics(simulator,
                            scene_config_path,
                            controller_config_path)

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
        
        if hold_time >= controller.dt:
            error = np.linalg.norm(controller.goal[:2] - state[:2])
            if error >= config["tolerance_error"]:
                state = simulator.current_state.world.robot.state
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
                elif config["total_peds"] == 0:
                    pedestrians_ghosts_states = np.array([config["state_dummy_ped"]])
                control, _ = controller.make_step(state,
                                                  pedestrians_ghosts_states)             
                hold_time = 0.
            else:
                break
        simulator.step(control)
        statistics.track_collisions()
        statistics.track_simulation_ticks()
        hold_time += simulator.sim_dt
    return statistics

if __name__ == "__main__":
    fire.Fire(main)