from core.controllers import AbstractController, DDPController, DWAController, MPPIController, DoMPCController
from core.predictors import AbstractPredictor, PedestrianTracker
from typing import Union
import numpy as np

class ControllerFactory():
    @staticmethod
    def create_controller(config: dict,
                          predictor: AbstractPredictor,
                          pedestrian_tracker: PedestrianTracker) -> AbstractController:
        if config["controller_type"] == "DWA":
            controller = DWAController(np.array(config["init_state"]),
                                       np.array(config["goal"]),
                                       config["dt"],
                                       config["horizon"],
                                       predictor,
                                       config["cost_function"],
                                       config["v_res"],
                                       config["w_res"],
                                       config["lb"],
                                       config["ub"],
                                       config["weights"],
                                       config["total_peds"],
                                       config["state_dummy_ped"],
                                       config["max_ghost_tracking_time"])
        elif config["controller_type"] == "MPC":
            controller = DoMPCController(np.array(config["init_state"]),
                                         np.array(config["goal"]),
                                         config["horizon"],
                                         config["dt"],
                                         config["model_type"],
                                         config["total_peds"],
                                         np.array(config["Q"]),
                                         np.array(config["R"]),
                                         config["W"],
                                         config["lb"],
                                         config["ub"],
                                         config["r_rob"],
                                         config["r_ped"],
                                         pedestrian_tracker,
                                         config["is_store_robot_predicted_trajectory"],
                                         config["max_ghost_tracking_time"],
                                         config["state_dummy_ped"],
                                         config["solver"],
                                         config["cost_function"],
                                         config["constraint_type"],
                                         config["constraint_value"])
        elif config["controller_type"] == "MPPI":
            controller = MPPIController(init_state=np.array(config["init_state"]),
                                        goal=np.array(config["goal"]),
                                        horizon=config["horizon"],
                                        dt=config["dt"],
                                        model_type=config["model_type"],
                                        total_peds=config["total_peds"],
                                        Q=config["Q"],
                                        R=np.array(config["R"]),
                                        W=config["W"],
                                        noise_sigma=np.array(config["noise_sigma"]),
                                        lb=config["lb"],
                                        ub=config["ub"],
                                        r_rob=config["r_rob"],
                                        r_ped=config["r_ped"],
                                        pedestrian_tracker=pedestrian_tracker,
                                        max_ghost_tracking_time=config["max_ghost_tracking_time"],
                                        cost_function=config["cost_function"],
                                        num_samples=config["num_samples"],
                                        lambda_=config["lambda_"],
                                        device=config["device"],
                                        cost_n_samples=config["cost_n_samples"])
        elif config["controller_type"] == "DDP":
            controller = DDPController(init_state=np.array(config["init_state"]),
                                       goal=np.array(config["goal"]),
                                       horizon=config["horizon"],
                                       dt=config["dt"],
                                       model_type=config["model_type"],
                                       total_peds=config["total_peds"],
                                       Q=np.array(config["Q"]),
                                       R=np.array(config["R"]),
                                       q_b=config["q_b"],
                                       q_m=config["q_m"],
                                       r_rob=config["r_rob"],
                                       r_ped=config["r_ped"],
                                       n_iterations=config["n_iterations"],
                                       constraint_value=config["constraint_value"],
                                       predictor=predictor,
                                       is_store_robot_predicted_trajectory=config["is_store_robot_predicted_trajectory"],
                                       max_ghost_tracking_time=config["max_ghost_tracking_time"],
                                       state_dummy_ped=config["state_dummy_ped"])

        return controller