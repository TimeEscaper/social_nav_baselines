from run_experiment import run_experiment
import json
import time
import fire
import pathlib
from typing import List

def conduct_experiments(folder_name: str,       
                        pedestrian_range: List[int],
                        scenes_list: List[int],
                        total_scenarios: int,
                        controller: str = None):
    start_time = time.time()
    exp = 0

    controllers = [controller]

    statistics = {scene:dict() for scene in scenes_list}
    trajectory_dataset = {scene:dict() for scene in scenes_list}
    log = ""
    total_experiments = len(pedestrian_range) * len(scenes_list) * total_scenarios * len(controllers)

    for controller in controllers:
        controller_config_path = fr"evaluation/studies/{folder_name}/configs/controllers/{controller}.yaml"
        for scene in scenes_list:
            statistics[scene][controller] = {}
            trajectory_dataset[scene][controller] = {}
            for total_peds in pedestrian_range:
                statistics[scene][controller][total_peds] = {}
                trajectory_dataset[scene][controller][total_peds] = {}
                for scenario_id in range(total_scenarios):
                    try:
                        exp += 1
                        scene_config_path = fr'evaluation/studies/{folder_name}/configs/scenes/{scene}/{total_peds}/{scenario_id}.yaml'
                        scenario_statistics = run_experiment(scene_config_path, controller_config_path)
                        statistics[scene][controller][total_peds][scenario_id] = {"failure_status": scenario_statistics.failure, 
                                                                                "simulation_ticks": scenario_statistics.simulation_ticks, 
                                                                                "total_collisions": scenario_statistics.total_collisions,
                                                                                "scene_config": scenario_statistics._scene_config_path,
                                                                                "controller_config": scenario_statistics._controller_config_path,
                                                                                "collisions_per_subgoal": scenario_statistics.collisions_per_subgoal_array}
                        print(f"Experiment: {exp}/{total_experiments}")
                    except Exception as e:
                        error_msg = f"""
Error in scene config: evaluation/studies/{folder_name}/configs/scenes/{scene}/{total_peds}/{scenario_id}.yaml
with controller {controller_config_path}
Exception: {e}
                        """
                        print(error_msg)
                        log += error_msg
    pathlib.Path(f"evaluation/studies/{folder_name}/results/datasets").mkdir(parents=True, exist_ok=True)
    with open(fr'evaluation/studies/{folder_name}/results/datasets/stats_{"_".join(controllers)}.json', 'w') as outfile:
            json.dump(statistics, outfile, indent=4)

    end_time = time.time()

    conclusion_msg = f"""
    Total time: {round(end_time - start_time, 3)}s
    Experiments conducted: {exp}/{total_experiments}
    Mean time for one experiment: {round((end_time - start_time) / exp, 3)}s
    """
    print(conclusion_msg)
    log += conclusion_msg
   
    pathlib.Path(f"evaluation/studies/{folder_name}/logs").mkdir(parents=True, exist_ok=True)
    with open(f"evaluation/studies/{folder_name}/logs/log_file_{round(time.time())}.txt", "w") as outlogfile:
        outlogfile.write(log)

if __name__ == "__main__":
    fire.Fire(conduct_experiments)
