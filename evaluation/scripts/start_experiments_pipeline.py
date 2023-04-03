import fire 
import yaml
import multiprocessing
from generate_scenes import generate_scenes
from conduct_experiments import conduct_experiments

DEFAULT_EXPERIMENT_CONFIG_PATH = r"evaluation/studies/study_0/configs/experiments/experiment_config.yaml"

def run_experiment(experiment_config_path: str = DEFAULT_EXPERIMENT_CONFIG_PATH) -> None:
    
    # Load experiment configuration file
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)

    # TODO: Check scene generation on the server
    # Scene generation 
    if experiment_config["generate_scenes"]:
        print("---------------Starting to generate scenes!---------------")
        generate_scenes(experiment_config["folder_name"],
                        experiment_config["scenes_list"],
                        experiment_config["pedestrian_range"],
                        experiment_config["total_scenarios"],
                        experiment_config["robot_vision_range"],
                        experiment_config["inflation_radius"],
                        experiment_config["borders_x"],
                        experiment_config["borders_y"],
                        experiment_config["seed"])
        print("------------------Scenes are generated!-------------------")

    # Conduct experiments
    input_data_list = []
    input_data_template = [experiment_config["folder_name"],
                           experiment_config["pedestrian_range"],
                           experiment_config["scenes_list"],
                           experiment_config["total_scenarios"]]
    for i, controller in enumerate(experiment_config["controller_list"]):
        input_data_i = input_data_template.copy()
        input_data_i.append(controller)
        input_data_list.append(tuple(input_data_i))
    num_processes = len(experiment_config["controller_list"])
    pool = multiprocessing.Pool(num_processes)
    _ = pool.map(conduct_experiments, input_data_list)
    

if __name__ == "__main__":
    fire.Fire(run_experiment)