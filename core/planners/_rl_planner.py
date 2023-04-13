import configparser
import pkg_resources

import numpy as np
import torch

from abc import ABC, abstractmethod
from typing import Any, Tuple, Optional, List

from pyminisim.core import PEDESTRIAN_RADIUS, ROBOT_RADIUS

from core.planners import AbstractPlanner
from core.controllers import AbstractController
from core.planners.rl.crowd_nav.utils.action import ActionPoint
from core.planners.rl.crowd_nav.utils.info import Collision, ReachGoal, Danger, Nothing
from core.planners.rl.crowd_nav.utils.robot import Robot
from core.planners.rl.crowd_nav.utils.state import ObservableState
from core.planners.rl.crowd_nav.utils.utils import transition_to_subgoal_polar, subgoal_to_global
from core.predictors import PedestrianTracker
from core.statistics import Statistics
from core.planners.rl.crowd_nav.policy.policy_factory import policy_factory


class _EnvStub:

    _PEDS_PADDING = 8

    def __init__(self, tracker: PedestrianTracker, global_goal: np.ndarray):
        self._tracker = tracker

        self.robot = None
        self.global_time = None
        self.time_step = 0.1
        self.time_limit = None

        self.success_reward: Optional[float] = None
        self.collision_penalty: Optional[float] = None
        self.discomfort_dist: Optional[float] = None
        self.discomfort_penalty_factor: Optional[float] = None

        self._global_goal = global_goal[:2].copy()

    @property
    def goal(self) -> np.ndarray:
        return self._global_goal.copy()

    def configure(self, config):
        self.success_reward = config.getfloat('reward', 'success_reward')
        self.collision_penalty = config.getfloat('reward', 'collision_penalty')
        self.discomfort_dist = config.getfloat('reward', 'discomfort_dist')
        self.discomfort_penalty_factor = config.getfloat('reward', 'discomfort_penalty_factor')

    def set_robot(self, robot):
        self.robot = robot

    def reset(self, robot_pose: np.ndarray, robot_vel: np.ndarray):
        self.robot.set(robot_pose[0],
                       robot_pose[1],
                       self._global_goal[0],
                       self._global_goal[1],
                       robot_vel[0],
                       robot_vel[1],
                       robot_pose[2],
                       ROBOT_RADIUS)
        self.robot.time_step = 0.1  # TODO: Should we use this value for subgoals?
        self.robot.policy.time_step = 0.1
        self.global_time = 0.
        max_steps = 20
        self.time_limit = (max_steps + 1) * self.robot.time_step

    def step(self, old_pose: np.ndarray, new_pose: np.ndarray, new_vel: np.ndarray):
        # TODO: Check for problems
        subgoal_action = transition_to_subgoal_polar(old_pose, new_pose)
        action = ActionPoint(s_lin=subgoal_action[0],
                             s_ang=subgoal_action[1],
                             px=new_pose[0],
                             py=new_pose[1],
                             theta=new_pose[2],
                             vx=new_vel[0],
                             vy=new_vel[1],
                             omega=new_vel[2])
        self.robot.step(action)
        self.global_time += self.time_step

    def get_observation(self):
        obs = []
        current_poses = self._tracker.get_current_poses(return_velocities=True)
        for v in current_poses.values():
            obs.append(ObservableState(v[0], v[1], v[2], v[3], PEDESTRIAN_RADIUS))
        if len(obs) < _EnvStub._PEDS_PADDING:
            for _ in range(_EnvStub._PEDS_PADDING - len(obs)):
                obs.append(ObservableState(-10., -10., 0., 0., 0.01))
        return obs

    def onestep_lookahead(self, action: ActionPoint):
        assert not action.is_empty, "Empty actions are not allowed here"
        robot_position = np.array([action.px, action.py])

        max_lin_vel = 2.
        distance = action.s_lin
        dt = 0.1
        approx_timesteps_index = int((distance / max_lin_vel) // dt)

        predictions = self._tracker.get_predictions()

        collision = False
        min_ped_distance = np.inf

        obs = []
        for v in predictions.values():
            vel_estimation = np.stack((np.ediff1d(v[0][:, 0]), np.ediff1d(v[0][:, 1])), axis=1)
            vel_estimation = np.concatenate((vel_estimation, vel_estimation[-1, np.newaxis]), axis=0)
            vel_estimation = vel_estimation / dt
            vel = vel_estimation[approx_timesteps_index]

            pose = v[0][approx_timesteps_index]
            dist = np.linalg.norm(pose - robot_position) - PEDESTRIAN_RADIUS - ROBOT_RADIUS
            if dist <= 0:
                collision = True
            if dist < min_ped_distance:
                min_ped_distance = dist

            obs.append(ObservableState(pose[0], pose[1], vel[0], vel[1], PEDESTRIAN_RADIUS))

        if len(obs) < _EnvStub._PEDS_PADDING:
            for _ in range(_EnvStub._PEDS_PADDING - len(obs)):
                obs.append(ObservableState(-10., -10., 0., 0., 0.01))

        goal_reached = np.linalg.norm(robot_position - self._global_goal) < ROBOT_RADIUS

        if collision:
            reward = self.collision_penalty
            done = True
            info = Collision()
        elif goal_reached:
            reward = self.success_reward
            done = True
            info = ReachGoal()
        elif min_ped_distance < self.discomfort_dist:
            reward = (min_ped_distance - self.discomfort_dist) * self.discomfort_penalty_factor * self.time_step
            done = False
            info = Danger(min_ped_distance)
        else:
            reward = 0.
            done = False
            info = Nothing()

        return obs, reward, done, info


class RLPlanner(AbstractPlanner):

    def __init__(self,
                 version: int,
                 global_goal: np.ndarray,
                 controller: AbstractController,
                 subgoal_reach_threshold: float,
                 subgoal_to_goal_threshold: float,
                 pedestrian_tracker: PedestrianTracker,
                 max_subgoal_steps: int = 25,
                 statistics_module: Optional[Statistics] = None):
        assert version > 0, f"Version must be an integer > 0"
        super(RLPlanner, self).__init__(global_goal=global_goal,
                                            controller=controller,
                                            subgoal_reach_threshold=subgoal_reach_threshold,
                                            subgoal_to_goal_threshold=subgoal_to_goal_threshold,
                                            max_steps_to_subgoal=max_subgoal_steps,
                                            statistics_module=statistics_module)
        self._pedestrian_tracker = pedestrian_tracker

        self._policy = policy_factory["sarl"]()
        policy_config = configparser.RawConfigParser( )
        policy_config.read(pkg_resources.resource_filename("core.planners.rl",
                                                           f"config/v{version}/policy_subgoal.config"))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._policy.configure(policy_config)
        self._policy.get_model().load_state_dict(torch.load(pkg_resources.resource_filename("core.planners.rl",
                                                           f"weights/v{version}/rl_model.pth"), map_location=device))

        env_config = configparser.RawConfigParser()
        env_config.read(pkg_resources.resource_filename("core.planners.rl",
                                                        f"config/v{version}/env.config"))
        self._env = _EnvStub(pedestrian_tracker, global_goal)
        self._env.configure(env_config)

        self._robot = Robot(env_config, 'robot')
        self._robot.set_policy(self._policy)
        self._env.set_robot(self._robot)

        self._policy.set_env(self._env)

        self._policy.set_phase("test")
        self._policy.set_device(device)
        self._policy.set_env(self._env)
        self._robot.print_info()

        self._env_inited = False
        self._previous_pose = None


    def generate_subgoal(self, state: np.ndarray, observation: Any,
                         robot_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        if not self._env_inited:
            self._env.reset(state, robot_velocity)
            self._env_inited = True
        else:
            self._env.step(self._previous_pose, state, robot_velocity)
        self._previous_pose = state.copy()

        obs = self._env.get_observation()
        action = self._robot.act(obs)

        subgoal = subgoal_to_global(np.array([action.s_lin, action.s_ang]), state)

        return subgoal

    def make_step(self, state: np.ndarray, observation: Any, robot_velocity: Optional[np.ndarray] = None) -> Tuple[np.ndarray, Any]:
        self._pedestrian_tracker.update(observation)
        new_subgoal, global_goal_reached = self.update_subgoal(state, observation, robot_velocity)
        if new_subgoal:
            self._controller.set_new_goal(state, self._current_subgoal)
        return self._controller.make_step(state, observation)
