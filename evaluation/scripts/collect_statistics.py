# Таблички формуруем по сценариям, то есть всего 4 таблички. В каждой из них описываются методы
# Метрики: mean, std 
import json
import numpy as np
from tabulate import tabulate
from texttable import Texttable
import latextable

scenes_list = ["circular_crossing", "parallel_traffic", "perpendicular_traffic", "random"]
controllers = ["dwa_0", "mpc_0", "mpc_1", "mpc_2", "mpc_3"]
total_peds_list = [1, 2, 4, 7]
total_scenarios_for_scene = 30
metrics = ["failure_status", "simulation_ticks", "total_collisions"]

with open(fr'evaluation/statistics/stats_dwa.json') as file:
        statistics = json.load(file)

for scene in scenes_list:
    for controller in controllers:
        for total_peds in total_peds_list:
            list_sim_ticks_for_scenario_batch = []
            list_collisions_for_scenario_batch = []
            list_timeouts_for_scenario_batch = []
            for scenario_id in range(total_scenarios_for_scene):
                list_sim_ticks_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["simulation_ticks"])
                list_collisions_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["total_collisions"])
                list_timeouts_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["failure_status"])
            # Calculate mean
            mean_sim_ticks_for_scenario_batch = np.mean(np.array(list_sim_ticks_for_scenario_batch))
            mean_collisions_for_scenario_batch = np.mean(np.array(list_collisions_for_scenario_batch))
            mean_timeouts_for_scenario_batch = np.mean(np.array(list_timeouts_for_scenario_batch))
            # Calculate std
            std_sim_ticks_for_scenario_batch = np.std(np.array(list_sim_ticks_for_scenario_batch))
            std_collisions_for_scenario_batch = np.std(np.array(list_collisions_for_scenario_batch))
            std_timeouts_for_scenario_batch = np.std(np.array(list_timeouts_for_scenario_batch))            
    # Вот здесь формируем табличку по этой сцене
    