import mpc_evaluation_script
import dwa_evaluation_script
import json
import time

start_time = time.time()
exp = 0

total_peds_list = [1, 2, 4, 7]
scenes_list = ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "random"]
controllers = ["ED-DWA", "MD-MPC", "ED-MPC", "MPC-MDC", "MPC-EDC"] # Тут можно выбрать какие контроллеры запускать
total_scenarios_for_scene = 30
statistics = {}
log = ""
total_experiments = len(total_peds_list) * len(scenes_list) * total_scenarios_for_scene

for controller in controllers:
    controller_config_path = fr"evaluation/controllers/{controller}.yaml"
    for scene in scenes_list:
        statistics[scene] = {}
        statistics[scene][controller] = {}
        for total_peds in total_peds_list:
            statistics[scene][controller][total_peds] = {}
            for scenario_id in range(total_scenarios_for_scene):
                try:
                    scene_config_path = fr'evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml'
                    scenario_statistics = dwa_evaluation_script.main(scene_config_path, controller_config_path)

                    statistics[scene][controller][total_peds][scenario_id] = {"failure_status": scenario_statistics.failure, 
                                                                              "simulation_ticks": scenario_statistics.simulation_ticks, 
                                                                              "total_collisions": scenario_statistics.total_collisions,
                                                                              "scene_config": scenario_statistics._scene_config_path,
                                                                              "controller_config": scenario_statistics._controller_config_path}
                    exp += 1
                    print(f"Experiment: {exp}/{total_experiments}")
                except:
                    error_msg = f"""
Error in scene config: evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml
with controller {controller_config_path}
                    """
                    print(error_msg)
                    log += error_msg

with open(fr'evaluation/statistics/stats_{"_".join(controllers)}.json', 'w') as outfile:
        json.dump(statistics, outfile, indent=4)


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