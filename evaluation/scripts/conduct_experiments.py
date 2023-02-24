import mpc_evaluation_script
import dwa_evaluation_script
import json
import time

start_time = time.time()
exp = 0

total_peds_list = [1, 2, 4, 7]
scenes_list = ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "random"]
total_scenarios_for_scene = 30
statistics = {
    "circular_crossing": {
        "dwa_0": {
            
            }
        },
    "parallel_traffic": {
        "dwa_0": {
            
            }
        },
    "perpendicular_traffic": {
        "dwa_0": {
            
            }
        },
    "random": {
        "dwa_0": {
            
            }
        }
}

# dwa controller config
controller_config_path = r"evaluation/controllers/dwa/0.yaml"
for scene in scenes_list:
    for total_peds in total_peds_list:
        statistics[scene]["dwa_0"][total_peds] = {}
        for scenario_id in range(total_scenarios_for_scene):
            try:
                scene_config_path = fr'evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml'
                scenario_statistics = dwa_evaluation_script.main(scene_config_path, controller_config_path)

                statistics[scene]["dwa_0"][total_peds][scenario_id] = {"failure_status": scenario_statistics.failure, 
                                                                       "simulation_ticks": scenario_statistics.simulation_ticks, 
                                                                       "total_collisions": scenario_statistics.total_collisions,
                                                                       "scene_config": scenario_statistics._scene_config_path,
                                                                       "controller_config": scenario_statistics._controller_config_path}
                exp += 1
            except:
                print(f"Error in scene config: evaluation/scenes/{scene}/{total_peds}/{scenario_id}.yaml")
                print(f"with controller evaluation/controllers/dwa/0.yaml")

with open(fr'evaluation/statistics/stats.json', 'w') as outfile:
        json.dump(statistics, outfile, indent=4)


end_time = time.time()

print("Время выполнения эксперимента: ", end_time - start_time)
print("Всего проведено эксперементов: ", exp)
print("Среднее время выполнения одного эксперемента: ", (end_time - start_time) / exp)