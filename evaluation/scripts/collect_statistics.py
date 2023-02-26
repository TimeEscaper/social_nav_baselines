# Таблички формуруем по сценариям, то есть всего 4 таблички. В каждой из них описываются методы
# Метрики: mean, std 
import json
import numpy as np
import pandas as pd

scenes_list = ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "random"]
#controllers = ["dwa_0"]#["dwa_0", "mpc_0", "mpc_1", "mpc_2", "mpc_3", "mpc_4"]
controllers = ["ED-DWA", "MD-MPC", "ED-MPC", "MPC-MDC", "MPC-EDC"]
total_peds_list = [1, 2, 4, 7]
total_scenarios_for_scene = 30
metrics = ["failure_status", "simulation_ticks", "total_collisions"]

tables_data = np.empty([len(scenes_list), len(controllers), len(metrics)*len(total_peds_list)], dtype='<U10')

statistics = {}

for controller in controllers:
    with open(fr'evaluation/statistics/stats_{controller}.json') as file:
            sub_statistics = json.load(file)
    for scene in scenes_list:
        statistics[scene] = statistics.setdefault(scene, dict())
        statistics[scene].update(sub_statistics[scene])

for scene_id, scene in enumerate(scenes_list):
    for controller_id, controller in enumerate(controllers):
        for total_peds_id, total_peds in enumerate(total_peds_list):
            list_sim_ticks_for_scenario_batch = []
            list_collisions_for_scenario_batch = []
            list_timeouts_for_scenario_batch = []
            for scenario_id in range(total_scenarios_for_scene):
                try:
                    list_sim_ticks_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["simulation_ticks"])
                    list_collisions_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["total_collisions"])
                    list_timeouts_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["failure_status"])
                except:
                     pass
            # Calculate mean
            mean_sim_ticks_for_scenario_batch = round(np.mean(np.array(list_sim_ticks_for_scenario_batch)), 1)
            mean_collisions_for_scenario_batch = round(np.mean(np.array(list_collisions_for_scenario_batch)), 1)
            mean_timeouts_for_scenario_batch = round(np.mean(np.array(list_timeouts_for_scenario_batch)), 1)
            # Calculate std
            std_sim_ticks_for_scenario_batch = round(np.std(np.array(list_sim_ticks_for_scenario_batch)), 1)
            std_collisions_for_scenario_batch = round(np.std(np.array(list_collisions_for_scenario_batch)), 1)
            std_timeouts_for_scenario_batch = round(np.std(np.array(list_timeouts_for_scenario_batch)), 1)
            # Add calculated data into the table
            tables_data[scene_id, controller_id, total_peds_id] = f"{mean_sim_ticks_for_scenario_batch}±{std_sim_ticks_for_scenario_batch}"  # simulation_ticks
            tables_data[scene_id, controller_id, total_peds_id+4] = f"{mean_collisions_for_scenario_batch}±{std_collisions_for_scenario_batch}" #collisions
            tables_data[scene_id, controller_id, total_peds_id+8] = f"{mean_timeouts_for_scenario_batch}±{std_timeouts_for_scenario_batch}" # timeouts           

# convert array into dataframe
circular_crossing_table = pd.DataFrame(tables_data[0])
parallel_traffic_table = pd.DataFrame(tables_data[1])
perpendicular_traffic_table = pd.DataFrame(tables_data[2])
random_table = pd.DataFrame(tables_data[3])
 
# save the dataframe as a csv file
circular_crossing_table.to_csv("evaluation/results/circular_crossing_table.csv", header=False, index=False)
parallel_traffic_table.to_csv("evaluation/results/parallel_traffic_table.csv", header=False, index=False)
perpendicular_traffic_table.to_csv("evaluation/results/perpendicular_traffic_table.csv", header=False, index=False)
random_table.to_csv("evaluation/results/random_table.csv", header=False, index=False)