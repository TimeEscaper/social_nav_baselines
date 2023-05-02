import fire 
import yaml
import numpy as np
import torch
import random
import multiprocessing
from generate_scenes import generate_scenes
from conduct_experiments import conduct_experiments
from collect_statistics_2 import collect_statistics

DEFAULT_EXPERIMENT_CONFIG_PATH = r"evaluation/studies/study_10/configs/experiments/experiment_config.yaml"

def run_experiment(experiment_config_path: str = DEFAULT_EXPERIMENT_CONFIG_PATH) -> None:
    
    # Load experiment configuration file
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)

    # Setup random seed
    random.seed(experiment_config["seed"])
    np.random.seed(experiment_config["seed"])
    torch.manual_seed(experiment_config["seed"])
    torch.cuda.manual_seed_all(experiment_config["seed"])

    # Scene generation 
    if experiment_config["generate_scenes"]:
        print("---------------1. Starting to generate scenes!---------------")
        generate_scenes(experiment_config["folder_name"],
                        experiment_config["scenes_list"],
                        experiment_config["pedestrian_range"],
                        experiment_config["total_scenarios"],
                        experiment_config["robot_vision_range"],
                        experiment_config["inflation_radius"],
                        experiment_config["borders_x"],
                        experiment_config["borders_y"],
                        experiment_config["seed"])
        print("--------------------Scenes are generated!--------------------")

    if False:
        
        # Conduct experiments
        print("-------------2. Starting to conduct experiments!-------------")
        input_data_list = []
        input_data_template = [experiment_config["folder_name"],
                            experiment_config["pedestrian_range"],
                            experiment_config["scenes_list"],
                            experiment_config["total_scenarios"]]
        for controller in experiment_config["controller_list"]:
            input_data_i = input_data_template.copy()
            input_data_i.append(controller)
            input_data_list.append(tuple(input_data_i))
        num_processes = len(experiment_config["controller_list"])
        pool = multiprocessing.Pool(num_processes)
        _ = pool.starmap(conduct_experiments, input_data_list)
        print("------------------Experiments are conducted!-----------------")
    
    if True:
        # Form statistics
        print("----------3. Starting to form statistics and plots!----------")
        collect_statistics(experiment_config["folder_name"],
                           experiment_config["scenes_list"],
                           experiment_config["controller_list"],
                           experiment_config["pedestrian_range"],
                           experiment_config["total_scenarios"],
                           experiment_config["metrics"])
        print("----------------Statistics and plots are formed!-------------")

if __name__ == "__main__":
    fire.Fire(run_experiment)