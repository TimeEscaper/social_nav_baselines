import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

scenes_list = ["circular_crossing", "parallel_crossing", "random_crossing"]
controllers = ["ED-MPC", "ED-MPC-EDC", "ED-MPC-MDC", "MD-MPC-MDC", "MD-MPC-EDC"]
total_peds_list = [3, 4, 5, 6]
total_scenarios_for_scene = 100
metrics = ["failure_status", "simulation_ticks", "total_collisions"]

tables_data = np.empty([len(scenes_list), len(controllers), len(metrics)*len(total_peds_list)], dtype='<U10')

graph_tables_data = np.zeros([len(scenes_list), len(controllers), len(metrics)*len(total_peds_list)])

statistics = {}

for controller in controllers:
    with open(fr'evaluation/statistics/stats_{controller}.json') as file:
            sub_statistics = json.load(file)
    for scene in scenes_list:
        statistics[scene] = statistics.setdefault(scene, dict())
        statistics[scene].update(sub_statistics[scene])

for scene_id, scene in enumerate(scenes_list):

    # Create distribution plot for a scene
    fig_hist, axs_hist = plt.subplots(len(controllers), len(metrics)*len(total_peds_list), 
                            figsize=[32, 16], constrained_layout=True, facecolor='white')
    fig_hist.suptitle(scenes_list[scene_id], fontsize=28)

    for controller_id, controller in enumerate(controllers):
        for total_peds_id, total_peds in enumerate(total_peds_list):
            list_sim_ticks_for_scenario_batch = []
            list_collisions_for_scenario_batch = []
            list_timeouts_for_scenario_batch = []
            for scenario_id in range(total_scenarios_for_scene):
                try:
                    simulation_ticks = statistics[scene][controller][str(total_peds)][str(scenario_id)]["simulation_ticks"]
                    if simulation_ticks < 2000: list_sim_ticks_for_scenario_batch.append(simulation_ticks)
                    list_collisions_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["total_collisions"])
                    list_timeouts_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["failure_status"])
                except:
                     pass

            # Plot Histograms 
            # Simulation time distribution  
            n, bins, patches = axs_hist[controller_id, total_peds_id].hist(list_sim_ticks_for_scenario_batch, 10, density=False, alpha=0.75, rwidth=0.85, color="#1f77b4")
            axs_hist[controller_id, total_peds_id].set_xlabel(r'Range of Simulation Time, [s]')
            axs_hist[controller_id, total_peds_id].set_ylabel(r'Amount of Experiments')
            axs_hist[controller_id, total_peds_id].set_title(f'Metric: Simulation Time \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
            axs_hist[controller_id, total_peds_id].grid(True)
            # Collision distribution
            n, bins, patches = axs_hist[controller_id, total_peds_id+len(total_peds_list)].hist(list_collisions_for_scenario_batch, 10, density=False, alpha=0.75, rwidth=0.85, color="#ff7f0e")
            axs_hist[controller_id, total_peds_id+len(total_peds_list)].set_xlabel(r'Range of Collisions')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)].set_ylabel(r'Amount of Experiments')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)].set_title(f'Metric: Collision \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)].grid(True)
            # Timeout distribution
            n, bins, patches = axs_hist[controller_id, total_peds_id+len(total_peds_list)*2].hist(np.array(list_timeouts_for_scenario_batch).astype(int), 10, density=False, alpha=0.75, rwidth=0.85, color="#2ca02c")
            axs_hist[controller_id, total_peds_id+len(total_peds_list)*2].set_xlabel(r'Range of Timeouts')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)*2].set_ylabel(r'Amount of Experiments')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)*2].set_title(f'Metric: Timeout \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
            axs_hist[controller_id, total_peds_id+len(total_peds_list)*2].grid(True)

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
            tables_data[scene_id, controller_id, total_peds_id+len(total_peds_list)] = f"{mean_collisions_for_scenario_batch}±{std_collisions_for_scenario_batch}" #collisions
            tables_data[scene_id, controller_id, total_peds_id+len(total_peds_list)*2] = f"{mean_timeouts_for_scenario_batch}±{std_timeouts_for_scenario_batch}" # timeouts           

            # Add mean to the graph table
            graph_tables_data[scene_id, controller_id, total_peds_id] = mean_sim_ticks_for_scenario_batch
            graph_tables_data[scene_id, controller_id, total_peds_id+len(total_peds_list)] = mean_collisions_for_scenario_batch
            graph_tables_data[scene_id, controller_id, total_peds_id+len(total_peds_list)*2] = mean_timeouts_for_scenario_batch          

    # Save histogram figure
    fig_hist.savefig(fr"evaluation/results/{scene}_distribution.png")

size = 25
# Create graph plot
for scene_id, scene in enumerate(scenes_list):
    
    fig_graph, axs_graph = plt.subplots(1, 3, 
                            figsize=[32, 16], constrained_layout=True, facecolor='white')
    fig_graph.suptitle(scenes_list[scene_id], fontsize=28)

    for controller_id, controller in enumerate(controllers):
        # Create graph
        # Simulation step to goal
        axs_graph[0].plot(total_peds_list, graph_tables_data[scene_id, controller_id, :len(total_peds_list)], linewidth=5, label=f"{controller}", color=DEFAULT_COLOR_HEX_PALETTE[controller_id], marker="o", markersize=size)
        axs_graph[0].grid(True)
        axs_graph[0].set_xlabel("Number of Pedestrians, [#]", fontsize=size)
        axs_graph[0].set_ylabel("Simulation Steps to Goal, [#]", fontsize=size)
        axs_graph[0].set_title("Simulation Steps to Goal, [#]", fontsize=size)
        axs_graph[0].legend(fontsize=size)
        # Collisions
        axs_graph[1].plot(total_peds_list, graph_tables_data[scene_id, controller_id, len(total_peds_list):len(total_peds_list)*2], linewidth=5, label=f"{controller}", color=DEFAULT_COLOR_HEX_PALETTE[controller_id], marker="o", markersize=size)
        axs_graph[1].grid(True)
        axs_graph[1].set_xlabel("Number of Pedestrians, [#]", fontsize=size)
        axs_graph[1].set_ylabel("Number of Collisions, [#]", fontsize=size)
        axs_graph[1].set_title("Number of Collisions, [#]", fontsize=size)
        axs_graph[1].legend(fontsize=size)
        # Timeouts
        axs_graph[2].plot(total_peds_list, graph_tables_data[scene_id, controller_id, len(total_peds_list)*2:], linewidth=5, label=f"{controller}", color=DEFAULT_COLOR_HEX_PALETTE[controller_id], marker="o", markersize=size)
        axs_graph[2].grid(True)
        axs_graph[2].set_xlabel("Number of Pedestrians, [#]", fontsize=size)
        axs_graph[2].set_ylabel("Number of Timeouts, [#]", fontsize=size)
        axs_graph[2].set_title("Number of Timeouts, [#]", fontsize=size)
        axs_graph[2].legend(fontsize=size)

    fig_graph.savefig(fr"evaluation/results/{scene}_graph.png")


# convert array into dataframe
circular_crossing_table = pd.DataFrame(tables_data[0])
parallel_traffic_table = pd.DataFrame(tables_data[1])
#perpendicular_traffic_table = pd.DataFrame(tables_data[2])
random_table = pd.DataFrame(tables_data[2])
 
# save the dataframe as a csv file
circular_crossing_table.to_csv("evaluation/results/circular_crossing_table.csv", header=False, index=False)
parallel_traffic_table.to_csv("evaluation/results/parallel_traffic_table.csv", header=False, index=False)
#perpendicular_traffic_table.to_csv("evaluation/results/perpendicular_traffic_table.csv", header=False, index=False)
random_table.to_csv("evaluation/results/random_table.csv", header=False, index=False)