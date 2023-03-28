import numpy as np
from typing import Tuple
from pyminisim.world_map import EmptyWorld
from pyminisim.visual import Renderer
from pyminisim.sensors import (PedestrianDetector, PedestrianDetectorConfig,
                               PedestrianDetectorNoise)
from pyminisim.robot import UnicycleDoubleIntegratorRobotModel, UnicycleRobotModel
from pyminisim.pedestrians import (HeadedSocialForceModelPolicy,
                                   RandomWaypointTracker, FixedWaypointTracker, OptimalReciprocalCollisionAvoidance)
from pyminisim.core import Simulation
from typing import List

def create_sim(init_state: np.ndarray,
               model_type: str,
               total_peds: int,
               is_robot_visible: bool,
               ped_detc_range: float,
               ped_detc_fov: int,
               misdetection_prob: float,
               dt: float,
               waypoint_tracker: str,
               pedestrians_init_states: List[List[float]],
               pedestrians_goals: List[List[List[float]]],
               pedestrian_model: str,
               create_renderer: bool = True) -> Tuple[Simulation, Renderer]:
    """Function sets up simulation parameters

    Args:
        init_state (np.ndarray): Initial state of the system
        model_type (str): Type of the model, ["unicycle", "unicycle_double_integrator"]
        total_peds (int): Amount of pedestrians in the system, >= 0
        is_robot_visible (bool): Robot visibity flag for pedestrians
        ped_detc_range (float): Range of the pedestrian detector, [m]
        ped_detc_fov (int): Field of view of the pedestrian detector, [deg]
        misdetection_prob (float): Misdetection probability, range(0, 1)
        dt (float): Time delta, [s]
        waypoint_tracker (str): Waypoint tracker for pedestrians: ["random", "fixed"]
        pedestrians_init_states (List[List[float]]): Initial states of the pedestrians according to the amount specified in total_peds
        pedestrians_goals (List[List[List[float]]]): Goals for the pedestrians according to the amount specified in total_peds

    Returns:
        Tuple[Simulation, Renderer]: Simulator and Renderer instances
    """
    if model_type == "unicycle":
        robot_model = UnicycleRobotModel(initial_pose=init_state,
                                         initial_control=np.array([0., np.deg2rad(0.)]))
    elif model_type == "unicycle_double_integrator":
        robot_model = UnicycleDoubleIntegratorRobotModel(initial_state=init_state,
                                                         initial_control=np.array([0., np.deg2rad(0.)]))
    pedestrians_model = None
    sensors = []
    if total_peds > 0:

        if waypoint_tracker == "random":
            tracker = RandomWaypointTracker(world_size=(10.0, 15.0))
            initial_poses = None
        elif waypoint_tracker == "fixed":
            waypoints = np.array(pedestrians_goals)
            tracker = FixedWaypointTracker(waypoints=waypoints)
            initial_poses = np.array(pedestrians_init_states)
            
        if pedestrian_model == "ORCA":
            pedestrians_model = OptimalReciprocalCollisionAvoidance(dt=dt,
                                                                    waypoint_tracker=tracker,
                                                                    n_pedestrians=total_peds,
                                                                    initial_poses=initial_poses,
                                                                    robot_visible=is_robot_visible)
        elif pedestrian_model == "HSFM":
            pedestrians_model = HeadedSocialForceModelPolicy(n_pedestrians=total_peds,
                                                             waypoint_tracker=tracker,
                                                             robot_visible=is_robot_visible,
                                                             initial_poses=initial_poses)
                                                             
        pedestrian_detector_config = PedestrianDetectorConfig(ped_detc_range,
                                                              ped_detc_fov)
        sensors = [PedestrianDetector(config=pedestrian_detector_config,
                                    noise=PedestrianDetectorNoise(0, 0, 0, 0, misdetection_prob))]
    sim = Simulation(sim_dt=dt,
                     world_map=EmptyWorld(),
                     robot_model=robot_model,
                     pedestrians_model=pedestrians_model,
                     sensors=sensors,
                     rt_factor=1)
    if create_renderer:
        renderer = Renderer(simulation=sim,
                            resolution=80.0,
                            screen_size=(700, 700))
        renderer.initialize()
        return sim, renderer
    else:
        return sim, None