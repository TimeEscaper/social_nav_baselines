import fire 
import yaml
from generate_scenes import generate_scenes

DEFAULT_EXPERIMENT_CONFIG_PATH = r"evaluation/configs/experiments/experiment_config.yaml"

def main(experiment_config_path: str = DEFAULT_EXPERIMENT_CONFIG_PATH) -> None:
    
    # Load experiment configuration file
    with open(experiment_config_path) as f:
        experiment_config = yaml.safe_load(f)

    # Scene generation
    if experiment_config["generate_scenes"]:
        print("---------------Starting to generate scenes!---------------")
        generate_scenes(experiment_config["pedestrian_range"],
                        experiment_config["total_scenarios"],
                        experiment_config["robot_vision_range"],
                        experiment_config["inflation_radius"],
                        experiment_config["borders_x"],
                        experiment_config["borders_y"],
                        experiment_config["seed"])
        print("------------------Scenes are generated!-------------------")

    # Conduct experiments
    

if __name__ == "__main__":
    fire.Fire(main)