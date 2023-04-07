import json
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import fire
import pathlib
from typing import List

# TODO: Refactor code on statistics collection

DEFAULT_COLOR_HEX_PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#724e56', '#999140', '#4f6d7a', '#f8c9a3', '#000000', '#4d4f53']
SIZE = 20

def add_box(ax, center, q25, q75, color, width=0.1, alpha=0.3):
    x0 = center - width / 2
    y0 = q25
    height = q75 - q25
    ax.add_patch(Rectangle((x0, y0), width, height, facecolor=color, alpha=alpha))

def collect_statistics(folder_name: str,
                       scenes_list: List[str],
                       controller_list: List[str],
                       pedestrian_range: List[int],
                       total_scenarios: int,
                       metrics: List[str]) -> None:
    
    # Folder creation
    pathlib.Path(f"evaluation/studies/{folder_name}/results").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"evaluation/studies/{folder_name}/results/datasets").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"evaluation/studies/{folder_name}/results/plots").mkdir(parents=True, exist_ok=True)
    pathlib.Path(f"evaluation/studies/{folder_name}/results/tables").mkdir(parents=True, exist_ok=True)

    tables_data = np.empty([len(scenes_list), len(controller_list), len(metrics)*len(pedestrian_range)], dtype='<U20')
    mean_tables_data = np.zeros([len(scenes_list), len(controller_list), len(metrics)*len(pedestrian_range)])
    median_tables_data = np.copy(mean_tables_data)
    q25_tables_data = np.copy(mean_tables_data)
    q75_tables_data = np.copy(mean_tables_data)
    statistics = {}

    for controller in controller_list:
        with open(fr'evaluation/studies/{folder_name}/results/datasets/stats_{controller}.json') as file:
                sub_statistics = json.load(file)
        for scene in scenes_list:
            statistics[scene] = statistics.setdefault(scene, dict())
            statistics[scene].update(sub_statistics[scene])
    
    for scene_id, scene in enumerate(scenes_list):
        # Create distribution plot for a scene
        fig_hist, axs_hist = plt.subplots(len(controller_list), len(metrics)*len(pedestrian_range), 
                                figsize=[12*4, len(controller_list)*4], constrained_layout=True, facecolor='white')
        fig_hist.suptitle(scenes_list[scene_id], fontsize=28)

        # Create mean plot
        fig_graph, axs_graph = plt.subplot_mosaic([[0, 1]], 
                                figsize=[21, 8.5], facecolor='white', layout='constrained')
        fig_graph.suptitle(scenes_list[scene_id], fontsize=28)

        displacement = np.linspace(-0.25, 0.25, len(controller_list))
        for controller_id, controller in enumerate(controller_list):
            for total_peds_id, total_peds in enumerate(pedestrian_range):
                list_sim_ticks_for_scenario_batch = []
                list_collisions_for_scenario_batch = []
                list_timeouts_for_scenario_batch = []
                for scenario_id in range(total_scenarios):
                    try:
                        simulation_ticks = statistics[scene][controller][str(total_peds)][str(scenario_id)]["simulation_ticks"]
                        if simulation_ticks < 2000: list_sim_ticks_for_scenario_batch.append(simulation_ticks)
                        list_collisions_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["total_collisions"])
                        list_timeouts_for_scenario_batch.append(statistics[scene][controller][str(total_peds)][str(scenario_id)]["failure_status"])
                    except:
                        pass

                # Plot Histograms 
                # Simulation time distribution  
                n, bins, patches = axs_hist[controller_id, total_peds_id].hist(list_sim_ticks_for_scenario_batch, 20, density=False, alpha=0.75, rwidth=0.85, color="#1f77b4")
                axs_hist[controller_id, total_peds_id].set_xlabel(r'Range of Simulation Time, [s]')
                axs_hist[controller_id, total_peds_id].set_ylabel(r'Amount of Experiments')
                axs_hist[controller_id, total_peds_id].set_title(f'Metric: Simulation Time \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
                axs_hist[controller_id, total_peds_id].grid(True)
                # Collision distribution
                n, bins, patches = axs_hist[controller_id, total_peds_id+len(pedestrian_range)].hist(list_collisions_for_scenario_batch, 20, density=False, alpha=0.75, rwidth=0.85, color="#ff7f0e")
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)].set_xlabel(r'Range of Collisions')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)].set_ylabel(r'Amount of Experiments')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)].set_title(f'Metric: Collision \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)].grid(True)
                # Timeout distribution
                n, bins, patches = axs_hist[controller_id, total_peds_id+len(pedestrian_range)*2].hist(np.array(list_timeouts_for_scenario_batch).astype(int), 20, density=False, alpha=0.75, rwidth=0.85, color="#2ca02c")
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)*2].set_xlabel(r'Range of Timeouts')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)*2].set_ylabel(r'Amount of Experiments')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)*2].set_title(f'Metric: Timeout \n Controller: {controller} \n Number of Pedestrians: {total_peds}')
                axs_hist[controller_id, total_peds_id+len(pedestrian_range)*2].grid(True)

                # Calculate mean
                mean_sim_ticks_for_scenario_batch = round(np.mean(np.array(list_sim_ticks_for_scenario_batch)), 3)
                mean_collisions_for_scenario_batch = round(np.mean(np.array(list_collisions_for_scenario_batch)), 3)
                mean_timeouts_for_scenario_batch = round(np.mean(np.array(list_timeouts_for_scenario_batch)), 3)
                # Calculate median
                median_sim_ticks_for_scenario_batch = round(np.median(np.array(list_sim_ticks_for_scenario_batch)), 3)
                median_collisions_for_scenario_batch = round(np.median(np.array(list_collisions_for_scenario_batch)), 3)
                median_timeouts_for_scenario_batch = round(np.median(np.array(list_timeouts_for_scenario_batch)), 3)
                # Calculate std
                std_sim_ticks_for_scenario_batch = round(np.std(np.array(list_sim_ticks_for_scenario_batch)), 3)
                std_collisions_for_scenario_batch = round(np.std(np.array(list_collisions_for_scenario_batch)), 3)
                std_timeouts_for_scenario_batch = round(np.std(np.array(list_timeouts_for_scenario_batch)), 3)
                # Calculate quantile 25
                q25_sim_ticks_for_scenario_batch = round(np.quantile(np.array(list_sim_ticks_for_scenario_batch), 0.25), 3)
                q25_collisions_for_scenario_batch = round(np.quantile(np.array(list_collisions_for_scenario_batch), 0.25), 3)
                #q25_timeouts_for_scenario_batch = round(np.quantile(np.array(list_timeouts_for_scenario_batch), 0.25), 3)
                # Calculate quantile 75
                q75_sim_ticks_for_scenario_batch = round(np.quantile(np.array(list_sim_ticks_for_scenario_batch), 0.75), 3)
                q75_collisions_for_scenario_batch = round(np.quantile(np.array(list_collisions_for_scenario_batch), 0.75), 3)
                #q75_timeouts_for_scenario_batch = round(np.quantile(np.array(list_timeouts_for_scenario_batch), 0.75), 3)
                
                # Add calculated data into the table
                tables_data[scene_id, controller_id, total_peds_id] = f"{mean_sim_ticks_for_scenario_batch}±{std_sim_ticks_for_scenario_batch}"  # simulation_ticks
                tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)] = f"{mean_collisions_for_scenario_batch}±{std_collisions_for_scenario_batch}" #collisions
                tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)*2] = f"{mean_timeouts_for_scenario_batch}±{std_timeouts_for_scenario_batch}" # timeouts           

                # Add mean to the graph table
                mean_tables_data[scene_id, controller_id, total_peds_id] = mean_sim_ticks_for_scenario_batch
                mean_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)] = mean_collisions_for_scenario_batch
                mean_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)*2] = mean_timeouts_for_scenario_batch

                # Add median to the graph table
                median_tables_data[scene_id, controller_id, total_peds_id] = median_sim_ticks_for_scenario_batch
                median_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)] = median_collisions_for_scenario_batch
                median_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)*2] = median_timeouts_for_scenario_batch          
                
                # Add q25 to the graph table
                q25_tables_data[scene_id, controller_id, total_peds_id] = q25_sim_ticks_for_scenario_batch
                q25_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)] = q25_collisions_for_scenario_batch
                #q25_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)*2] = q25_timeouts_for_scenario_batch

                # Add q75 to the graph table
                q75_tables_data[scene_id, controller_id, total_peds_id] = q75_sim_ticks_for_scenario_batch
                q75_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)] = q75_collisions_for_scenario_batch
                #q75_tables_data[scene_id, controller_id, total_peds_id+len(pedestrian_range)*2] = q75_timeouts_for_scenario_batch

            # Create graph
            # Simulation step to goal
            axs_graph[0].scatter(pedestrian_range + displacement[controller_id], mean_tables_data[scene_id, controller_id, :len(pedestrian_range)], linewidth=1, label=f"{controller}", marker="o", color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            axs_graph[0].plot(pedestrian_range + displacement[controller_id], median_tables_data[scene_id, controller_id, :len(pedestrian_range)], linewidth=2, linestyle="--", label=f"{controller}", marker="^", markersize=SIZE//2, color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            for i, ped in enumerate(pedestrian_range):
                center_range = np.linspace(-0.25, 0.25, len(controller_list)) + ped
                q25 = q25_tables_data[scene_id, controller_id, i]
                q75 = q75_tables_data[scene_id, controller_id, i]
                add_box(axs_graph[0], center_range[controller_id], q25, q75, color=DEFAULT_COLOR_HEX_PALETTE[controller_id], width=0.01, alpha=0.5)
            axs_graph[0].grid(True)
            axs_graph[0].set_xlabel("Number of Pedestrians, [#]", fontsize=SIZE)
            axs_graph[0].set_ylabel("Simulation Steps to Goal, [#]", fontsize=SIZE)
            axs_graph[0].set_title("Simulation Steps to Goal, [#]", fontsize=SIZE*1.2)
            axs_graph[0].tick_params(labelsize=12)
            #axs_graph[0].legend(fontsize=size)
            legend_elements = [Line2D([0], [0], color='black', marker="o", lw=0, label='Mean'),
                               Line2D([0], [0], marker='^', color='black', label='Median', linestyle='--',
                          markerfacecolor='black', markersize=10),
                               Patch(facecolor='black', alpha=0.5, 
                         label='IQR')]

            axs_graph[0].legend(handles=legend_elements, bbox_to_anchor=(-.2, 1), loc='upper right', borderaxespad=0., fontsize=20)
            # Collisions
            axs_graph[1].scatter(pedestrian_range + displacement[controller_id], mean_tables_data[scene_id, controller_id, len(pedestrian_range):len(pedestrian_range)*2], linewidth=1, label=f"{controller}",  marker="o", color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            axs_graph[1].plot(pedestrian_range + displacement[controller_id], median_tables_data[scene_id, controller_id, len(pedestrian_range):len(pedestrian_range)*2], linewidth=2, linestyle="--",  marker="^", markersize=SIZE//2, color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            for i, ped in enumerate(pedestrian_range):
                center_range = np.linspace(-0.25, 0.25, len(controller_list)) + ped
                q25 = q25_tables_data[scene_id, controller_id, i+len(pedestrian_range)]
                q75 = q75_tables_data[scene_id, controller_id, i+len(pedestrian_range)]
                add_box(axs_graph[1], center_range[controller_id], q25, q75, color=DEFAULT_COLOR_HEX_PALETTE[controller_id], width=0.01, alpha=0.5)
            axs_graph[1].grid(True)
            axs_graph[1].set_xlabel("Number of Pedestrians, [#]", fontsize=SIZE)
            axs_graph[1].set_ylabel("Number of Collisions, [#]", fontsize=SIZE)
            axs_graph[1].set_title("Number of Collisions, [#]", fontsize=SIZE*1.2)
            axs_graph[1].tick_params(labelsize=12)
            #axs_graph[1].legend(fontsize=size)
            # Timeouts
            #axs_graph[2].scatter(pedestrian_range, mean_tables_data[scene_id, controller_id, len(pedestrian_range)*2:], linewidth=1, label=f"{controller}", marker="o", color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            #axs_graph[2].plot(pedestrian_range, median_tables_data[scene_id, controller_id, len(pedestrian_range)*2:], linewidth=2, linestyle="--", marker="^", markersize=SIZE//2, color=DEFAULT_COLOR_HEX_PALETTE[controller_id])
            #axs_graph[2].grid(True)
            #axs_graph[2].set_xlabel("Number of Pedestrians, [#]", fontsize=SIZE)
            #axs_graph[2].set_ylabel("Number of Timeouts, [#]", fontsize=SIZE)
            #axs_graph[2].set_title("Number of Timeouts, [#]", fontsize=SIZE*1.2)
            #axs_graph[2].tick_params(labelsize=12)
            # Legend
            axs_graph[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=20)


        # Save histogram figure
        fig_hist.savefig(fr"evaluation/studies/{folder_name}/results/plots/{scene}_distribution.png")
        
        # Save mean plot
        fig_graph.savefig(fr"evaluation/studies/{folder_name}/results/plots/{scene}_graph.png")

    # convert array into dataframe
    if "circular_crossing" in scenes_list:
        circular_crossing_table = pd.DataFrame(tables_data[0])
        circular_crossing_table.to_csv(f"evaluation/studies/{folder_name}/results/tables/circular_crossing.csv", header=False, index=False)
    if "parallel_crossing" in scenes_list:
        parallel_traffic_table = pd.DataFrame(tables_data[1])
        parallel_traffic_table.to_csv(f"evaluation/studies/{folder_name}/results/tables/parallel_crossing.csv", header=False, index=False)
    if "random_crossing" in scenes_list:
        random_table = pd.DataFrame(tables_data[2])
        random_table.to_csv(f"evaluation/studies/{folder_name}/results/tables/random_crossing.csv", header=False, index=False)
    
if __name__=="__main__":
    fire.Fire(collect_statistics)
