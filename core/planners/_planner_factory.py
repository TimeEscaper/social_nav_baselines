from core.planners import DummyPlanner, RandomPlanner, RLPlanner, AbstractPlanner
from core.predictors import AbstractPredictor, PedestrianTracker
from core.controllers import AbstractController
from core.statistics import Statistics
from typing import Union
import numpy as np

class PlannerFactory():
    @staticmethod
    def create_planner(config: dict,
                       controller: AbstractController,
                       pedestrian_tracker: PedestrianTracker,
                       statistics_module: Statistics) -> AbstractPlanner:
        if ("planner" not in config) or (config["planner"] == "dummy"):
            planner = DummyPlanner(global_goal=np.array(config["goal"]),
                                    controller=controller,
                                    subgoal_reach_threshold=config["tolerance_error"],
                                    subgoal_to_goal_threshold=2.,
                                    pedestrian_tracker=pedestrian_tracker,
                                    statistics_module=statistics_module)
        elif config["planner"] == "RL":
            planner = RLPlanner(version=2,
                                global_goal=np.array(config["goal"]),
                                controller=controller,
                                subgoal_reach_threshold=config["tolerance_error"],
                                subgoal_to_goal_threshold=1.,
                                pedestrian_tracker=pedestrian_tracker,
                                statistics_module=statistics_module,
                                max_subgoal_steps=25)
        elif config["planner"] == "random":
            planner = RandomPlanner(global_goal=np.array(config["goal"]),
                                    controller=controller,
                                    subgoal_reach_threshold=config["tolerance_error"],
                                    subgoal_to_goal_threshold=2.,
                                    pedestrian_tracker=pedestrian_tracker,
                                    statistics_module=statistics_module)
        return planner