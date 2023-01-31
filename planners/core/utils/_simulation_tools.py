import numpy as np
from typing import Tuple
from pyminisim.world_map import EmptyWorld
from pyminisim.visual import Renderer
from pyminisim.sensors import (PedestrianDetector, PedestrianDetectorConfig,
                               PedestrianDetectorNoise)
from pyminisim.robot import UnicycleDoubleIntegratorRobotModel, UnicycleRobotModel
from pyminisim.pedestrians import (HeadedSocialForceModelPolicy,
                                   RandomWaypointTracker)
from pyminisim.core import Simulation

def create_sim(init_state: np.ndarray,
               model_type: str,
               total_peds: int,
               is_robot_visible: bool,
               ped_dect_range: float,
               ped_dect_fov: int,
               misdetection_prob: float,
               dt: float) -> Tuple[Simulation, Renderer]:
    """Function sets up simulation parameters

    Args:
        init_state (np.ndarray): Initial state of the system
        model_type (str): Type of the model, ["unicycle", "unicycle double integrator"]
        total_peds (int): Amount of pedestrians in the system, >= 0
        is_robot_visible (bool): Robot visibity flag for pedestrians
        ped_dect_range (float): Range of the pedestrian detector, [m]
        ped_dect_fov (int): Field of view of the pedestrian detector, [deg]
        misdetection_prob (float): Misdetection probability, range(0, 1)
        dt (float): Time delta, [s]

    Returns:
        Tuple[Simulation, Renderer]: Simulator and Renderer instances
    """
    if model_type == "unicycle":
        robot_model = UnicycleRobotModel(initial_pose=init_state,
                                     initial_control=np.array([0., np.deg2rad(0.)]))
    elif model_type == "unicycle double integrator":
        robot_model = UnicycleDoubleIntegratorRobotModel(initial_state=init_state,
                                                        initial_control=np.array([0., np.deg2rad(0.)]))
    pedestrians_model = None
    sensors = []
    if total_peds > 0:
        tracker = RandomWaypointTracker(world_size=(10.0, 15.0))
        pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=total_peds,
                                                        waypoint_tracker=tracker,
                                                        robot_visible=is_robot_visible) # initial_poses=np.array([[1000., 1000, 0]])                    
        pedestrian_detector_config = PedestrianDetectorConfig(ped_dect_range,
                                                              ped_dect_fov)
        sensors = [PedestrianDetector(config=pedestrian_detector_config,
                                    noise=PedestrianDetectorNoise(0, 0, 0, 0, misdetection_prob))]
    sim = Simulation(sim_dt=dt,
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     sensors=sensors)
    renderer = Renderer(simulation=sim,
                        resolution=80.0,
                        screen_size=(700, 700))
    return sim, renderer
