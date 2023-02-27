import mpc_evaluation_script
import dwa_evaluation_script
import json
import time
import traceback
import fire

def main(controller: str):
    start_time = time.time()
    exp = 0

    total_peds_list = [7] # [1, 2, 4, 7]
    scenes_list = ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "random"]
    controllers = [controller] #["ED-DWA", "MD-MPC", "ED-MPC", "MPC-MDC", "MPC-EDC", "MD-MPC-EDC", "MPPI"] # Тут можно выбрать какие контроллеры запускать
    total_scenarios_for_scene = 1
    statistics = {scene:dict() for scene in scenes_list}
    trajectory_dataset = {scene:dict() for scene in scenes_list}
    log = ""
    total_experiments = len(total_peds_list) * len(scenes_list) * total_scenarios_for_scene * len(controllers)

    for controller in controllers:
        controller_config_path = fr"evaluation/controllers/{controller}.yaml"
        for scene in scenes_list:
            statistics[scene][controller] = {}
            trajectory_dataset[scene][controller] = {}
            for total_peds in total_peds_list:
                statistics[scene][controller][total_peds] = {}
                trajectory_dataset[scene][controller][total_peds] = {}
                for scenario_id in range(total_scenarios_for_scene):
                    try:
                        scene_config_path = fr'evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml'
                        if controller == "ED-DWA":
                            scenario_statistics = dwa_evaluation_script.main(scene_config_path, controller_config_path)
                        else:
                            scenario_statistics = mpc_evaluation_script.main(scene_config_path, controller_config_path)
                        statistics[scene][controller][total_peds][scenario_id] = {"failure_status": scenario_statistics.failure, 
                                                                                "simulation_ticks": scenario_statistics.simulation_ticks, 
                                                                                "total_collisions": scenario_statistics.total_collisions,
                                                                                "scene_config": scenario_statistics._scene_config_path,
                                                                                "controller_config": scenario_statistics._controller_config_path,
                                                                                "collisions_per_subgoal": scenario_statistics.collisions_per_subgoall_array}
                        
                        trajectory_dataset[scene][controller][total_peds][scenario_id] = {"ground_truth_pedestrian_trajectories": scenario_statistics.ground_truth_pedestrian_trajectories.tolist(),
                                                                                          "ground_truth_robot_trajectory": scenario_statistics.ground_truth_robot_trajectory.tolist(),
                                                                                          "predicted_pedestrians_trajectories": scenario_statistics.predicted_pedestrians_trajectories.tolist(),
                                                                                          "predicted_robot_trajectory": scenario_statistics.predicted_robot_trajectory.tolist(),
                                                                                          "predicted_pedestrians_covariances": scenario_statistics.predicted_pedestrians_covariances.tolist(),
                                                                                          "subgoals_trajectory": scenario_statistics.subgoals_trajectory.tolist()}
                        exp += 1
                        print(f"Experiment: {exp}/{total_experiments}")
                    except Exception as e:
                        error_msg = f"""
Error in scene config: evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml
with controller {controller_config_path}
Exception: {e}
                        """
                        print(error_msg)
                        log += error_msg

    with open(fr'evaluation/statistics/stats_{"_".join(controllers)}.json', 'w') as outfile:
            json.dump(statistics, outfile, indent=4)
    
    with open(fr'evaluation/datasets/trajectory_dataset_{"_".join(controllers)}.json', 'w') as outfile:
            json.dump(trajectory_dataset, outfile)


    end_time = time.time()

    conclusion_msg = f"""
    Total time: {round(end_time - start_time, 3)}s
    Experiments conducted: {exp}/{total_experiments}
    Mean time for one experiment: {round((end_time - start_time) / exp, 3)}s
    """
    print(conclusion_msg)
    log += conclusion_msg

    with open(fr"evaluation/logs/log_file_{round(time.time())}.txt", "w") as outlogfile:
        outlogfile.write(log)

if __name__ == "__main__":
    fire.Fire(main)
