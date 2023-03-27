from ._controller import AbstractController
from core.predictors import AbstractPredictor, NeuralPredictor, ConstantVelocityPredictor
from core.predictors import PedestrianTracker
import numpy as np
import casadi
from typing import List, Dict
import shutil
import sys
import forcespro

class FORCESPROMPCController(AbstractController):

    def __init__(self,
                 init_state: np.ndarray,
                 goal: np.ndarray,
                 horizon: int,
                 dt: float,
                 model_type: str,
                 total_peds: int,
                 Q: np.ndarray,
                 R: np.ndarray,
                 W: float,
                 lb: List[float],
                 ub: List[float],
                 r_rob: float,
                 r_ped: float,
                 pedestrian_tracker: PedestrianTracker,
                 is_store_robot_predicted_trajectory: bool,
                 max_ghost_tracking_time: int,
                 state_dummy_ped: List[float],
                 solver_name: str,
                 cost_function: str,
                 constraint_type: str,
                 constraint_value: float) -> None:
        
        """Initiazition of the class instance

        Args:
            init_state (np.ndarray): Initial state of the system
            goal (np.ndarray): Goal of the system
            horizon (int): Prediction horizon of the controller, [steps]
            dt (float): Time delta, [s]
            model_type (str): Type of the model, ["unicycle", "unicycle_double_integrator"]
            total_peds (int): Amount of pedestrians in the system, >= 0
            Q (np.ndarray): State weighted matrix, [state_dim * state_dim]
            R (np.ndarray): Control weighted matrix, [control_dim * control_dim]
            lb (List[float]): Lower boundaries for system states or/and controls
            ub (List[float]): Upper boundaries for system states or/and controls
            r_rob (float): Robot radius, [m]
            r_ped (float): Pedestrian radius, [m]
            constraint_value (float): Minimal safe distance between robot and pedestrian, [m]
            predictor (AbstractPredictor): Predictor, [Constant Velocity Predictor, Neural Predictor]
            # TODO

        Returns:
            _type_: None
        """
        super().__init__(init_state,
                         self.get_ref_direction(init_state,
                                                goal),
                         horizon,
                         dt,
                         state_dummy_ped,
                         max_ghost_tracking_time)
        self._ped_tracker = pedestrian_tracker
        self._is_store_robot_predicted_trajectory = is_store_robot_predicted_trajectory
        self._ghost_tracking_times : List[int]= [0] * total_peds

        self.R = R
        self.Q = Q

        # Problem dimensions
        self.model = forcespro.nlp.SymbolicModel()
        self.model.N = horizon # horizon length
        self.model.nvar = 5  # number of variables
        self.model.neq = 3  # number of equality constraints
        self.model.npar = 6 # number of runtime params [x_goal y_goal p_rob_0_x p_rob_0_y]

        # Objective function
        self.model.objective =  self._stage_cost
        self.model.objectiveN = self._terminal_cost
        
        integrator_stepsize = dt
        self.model.eq = lambda z: forcespro.nlp.integrate(self._unicycle_continuous_dynamics,
                                                     z[2:5], z[0:2],
                                                     integrator=forcespro.nlp.integrators.RK4,
                                                     stepsize=integrator_stepsize)
    
        # Indices on LHS of dynamical constraint - for efficiency reasons, make
        # sure the matrix E has structure [0 I] where I is the identity matrix.
        self.model.E = np.concatenate([np.zeros((3,2)), np.eye(3)], axis=1)

        # Inequality constraints
        #  upper/lower variable bounds lb <= z <= ub
        #                     inputs   |  states
        #                     v     w  |   x       y     theta         
        self.model.lb = np.array([ 0,  -2,   -np.inf,  -np.inf, -np.inf])
        self.model.ub = np.array([+2,  +2,   +np.inf,  +np.inf, +np.inf])

        # Initial condition on vehicle states x
        self.model.xinitidx = range(2,5) # use this to specify on which variables initial 
        # conditions are imposed

        # Solver generation
        # -----------------
        # Set solver options
        codeoptions = forcespro.CodeOptions('FORCESNLPsolver')
        codeoptions.maxit = 1000     # Maximum number of iterations
        codeoptions.printlevel = 2  # Use printlevel = 2 to print progress (but 
        #                             not for timings)
        codeoptions.optlevel = 0    # 0 no optimization, 1 optimize for size, 
        #                             2 optimize for speed, 3 optimize for size & speed

        codeoptions.cleanup = False
        codeoptions.timing = 1
        codeoptions.solvemethod = solver_name # choose the solver method Sequential 

        codeoptions.nohash = 0
        codeoptions.overwrite = 1

        self.solver = self.model.generate_solver(options=codeoptions)

        # Simulation parameters
        # Simulation

        # Set initial guess to start solver from
        self.x0i = np.zeros((self.model.nvar, 1))
        self.x0 = np.transpose(np.tile(self.x0i, (1, self.model.N)))
        # Set initial condition
        self.xinit = np.transpose(np.array(init_state))

        # cost-func values
        self.cost_val = []

        self.problem = {"x0": self.x0,
                        "xinit": self.xinit}

        # predicted positions of the robot
        #self.x_pred_stacked = np.zeros((sim_length, model.neq, model.N))
        
        
    def _unicycle_continuous_dynamics(self, x: casadi.SX, u: casadi.SX) -> casadi.SX:
        """Defines continuos kinematics of the robot

        Unicycle model 

        Args:
            state x = [xPos, yPos, theta] 
            input u = [v, w]

        Returns:
            state xdot = [xPos_dot, yPos_dot, theta_dot]
        """

        theta = x[2]
        v = u[0]
        w = u[1]
        return casadi.vertcat(v * casadi.cos(theta),
                              v * casadi.sin(theta),
                              w)
    
    def _stage_cost(self, z: casadi.SX) -> casadi.SX:
        """Control input weighted stage cost function 
        
        Args:
            variables z = [v, w, x, y, theta]

        Returns:
            casadi.SX: J_i - terminal cost
        """
        u = z[0:2]
        #v = z[0]
        #w = z[1]
    
        # Cost function at the i-step
        #J_i = v**2 + w**2
        
        J_i = u.T @ self.R @ u

        return J_i
    
    @staticmethod
    def _square_distance(vector: casadi.SX) -> casadi.SX:
        return vector[0] ** 2 + vector[1] ** 2

    def _terminal_cost(self, z: casadi.SX, p: casadi.SX) -> casadi.SX:
        """Square distance terminal cost

        Args:
            z (casadi.SX): variables z = [v, w, x, y, theta]
            p (casadi.SX): run-time parameters p = [x_goal y_goal theta_goal p_rob_0_x p_rob_0_y p_rob_0_theta]

        Returns:
            casadi.SX: J_N - terminal cost
        """
        u = z[0:2]
        p_rob_N = z[2:5]
        p_rob_0 = p[3:6]
        current_goal = p[0:3]

        J_state = self.Q \
                  * FORCESPROMPCController._square_distance(p_rob_N - current_goal) \
                  / FORCESPROMPCController._square_distance(p_rob_0 - current_goal)

        J_control = u.T @ self.R @ u

        return J_state #+ J_control
    
    def get_predicted_robot_trajectory(self) -> List[float]:
        return self.pred_x[:2].T

    def make_step(self,
                  state: np.ndarray,
                  observation: Dict[int, np.ndarray]) -> np.ndarray:
        
        # Set initial condition
        self.problem["xinit"] = state.T

        # Initial position on the prediction step
        p_rob_0 = state

        # Formulate full vector of transmitted parameters
        parameter_vector = np.hstack([self._goal, p_rob_0])

        # Set goal position
        self.problem["all_parameters"] = np.tile(parameter_vector, (self.model.N,))

        # Solve
        self.output, exitflag, info = self.solver.solve(self.problem)

        # Make sure the solver has exited properly.
        assert exitflag == 1, f"bad exitflag {exitflag}"
        sys.stderr.write("FORCESPRO took {} iterations and {} seconds to solve the problem.\n"
                         .format(info.it, info.solvetime))
        
        # Extract output
        temp = np.zeros((np.max(self.model.nvar), self.model.N))
        for i in range(0, self.model.N):
            temp[:, i] = self.output['x{0:02d}'.format(i+1)]
        pred_u = temp[0:2, :]
        self.pred_x = temp[2:5, :]
        
        return pred_u[:, 0]

    @property
    def predictor(self) -> AbstractPredictor:
        return self._predictor
    
    @property
    def init_state(self) -> np.ndarray:
        return self._init_state
    
    @property
    def ghost_tracking_times(self) -> List[int]:
        return self._ghost_tracking_times
    
    @property
    def max_ghost_tracking_time(self) -> int:
        return self._max_ghost_tracking_time

    @property
    def predicted_pedestrians_trajectories(self) -> np.ndarray:
        return self._predicted_pedestrians_trajectories