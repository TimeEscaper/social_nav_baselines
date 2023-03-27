%% Clean upgenerate_pathplanner;
clear; clc; close all; clearvars;
rng('shuffle');

%% Delete previous Solver
% Forces does not always code changes and might reuse the previous solution
try
FORCEScleanup('FORCESNLPsolver','all');
catch
end

try
    rmdir('@FORCESproWS','s')
catch
end
try
    rmdir('FORCESNLPsolver','s')
catch
end

%%
model = generate_pathplanner();

function [xdot] = continuous_dynamics(x, u)
    % state x = [xPos, yPos, theta]
    % input u = [v, w] 
    theta = x(3);
    v = u(1);
    w = u(2);
    xdot = [v * cos(theta);
            v * sin(theta);
            w];
end

function [J_i] = obj(z, p)
    % Reference (goal)robot position
    p_ref_0 = p(1:2);

    % Initial position on the prediction step
    p_0 = p(3:4);

    % Position of the robot on the i-step
    p_N = z(3:4);

    % Cost function at the i-step
    %J_i =  norm(p_N - p_ref_0) / norm(p_0 - p_ref_0);
    J_i = norm(p_0 - p_ref_0);

end

function [model] = generate_pathplanner()
    % Problem dimensions
    model.N = 20;  % horizon length
    model.nvar = 5;  % number of variables
    model.neq = 3;  % number of equality constraints
    model.npar = 4; % number of runtime parameters [x_goal y_goal p_rob_0_x p_rob_0_y] 

    model.objective = @obj; % Здесь можно будет использовать LSobjective для апроксимации методом гаусс-ньютона

    % We use an explicit RK4 integrator here to discretize continuous dynamics
    integrator_stepsize = 0.1;
    model.eq = @(z) RK4(z(3:5), z(1:2), @continuous_dynamics, integrator_stepsize);

    % Selection matrix
    model.E = [zeros(3, 2), eye(3)];

    % Variable bounds
    %            inputs   |  states
    %            v     w  |   x    y    theta  
    model.lb = [ 0,  -2,   -inf,  -inf, -inf];
    model.ub = [+2,  +2,   +inf,  +inf, +inf];

    % Initial condition on vehicle states x
    model.xinitidx = 3:5; % use this to specify on which variables initial conditions
    % are imposed

    %% Set solver options
    codeoptions = getOptions('FORCESNLPsolver');
    codeoptions.maxit = 400; % Maximum number of iterations
    codeoptions.printlevel = 2; % Use printlevel = 2 to print progress (but(not for timings)
    codeoptions.optlevel = 0; % 0: no optimization, 1: optimize for size, 2: optimize for speed, 3: optimize for size & speed
    codeoptions.cleanup = false;
    codeoptions.timing = 1;
    codeoptions.printlevel = 0;
    %codeoptions.nlp.hessian_approximation = 'bfgs'; % set initialization of the hessian approximation
    %codeoptions.solvemethod = 'SQP_NLP'; % choose the solver method Sequential Quadratic Programming
    %codeoptions.sqp_nlp.maxqps = 5; % maximum number of quadratic problems to be solved during one solver call
    %codeoptions.sqp_nlp.reg_hessian = 5e-9; % increase this parameter if exitflag=-8
    FORCES_NLP(model, codeoptions);
end

function simulate(model)
    
    sim_length = 80; % simulate 8sec

    % Variables for storing simulation data
    x = zeros(3, sim_length+1) % states
    u = zeros(2, sim_length) % inputs

end