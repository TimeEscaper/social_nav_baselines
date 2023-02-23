from pyminisim.core import Simulation

class Statistics():
    def __init__(self,
                 simulator: Simulation,
                 config_path: str) -> None:
        self._simulator = simulator
        self._simulation_ticks = 0
        self._total_collisions = 0
        self._previous_step_collision_indices = simulator._get_world_state().robot_to_pedestrians_collisions
        self._current_step_collision_indices = simulator._get_world_state().robot_to_pedestrians_collisions
        self._failure = False
        self._config_path = config_path

    def set_failure_flag(self):
        self._failure = True

    def track_collisions(self):
        self._current_step_collision_indices = self._simulator._get_world_state().robot_to_pedestrians_collisions
        delta = len(set(self._previous_step_collision_indices) & (set(self._previous_step_collision_indices) - set(self._current_step_collision_indices)))
        if delta > 0:
            self._total_collisions += delta
        self._previous_step_collision_indices = self._current_step_collision_indices

    def track_simulation_ticks(self):
        self._simulation_ticks += 1
    
    @property
    def failure(self) -> int:
        return self._failure

    @property
    def simulation_ticks(self) -> int:
        return self._simulation_ticks
    
    @property
    def total_collisions(self) -> int:
        return self._total_collisions

    