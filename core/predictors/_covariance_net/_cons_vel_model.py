import numpy as np


def constant_velocity_model(positions, num_future_positions=12):
    dt = 0.1  # time step, can be adjusted as needed

    # estimate linear velocity
    velocities = (positions[1:, :2] - positions[:-1, :2]) / dt
    cur_vel = 0.6 * velocities[ 6, :] + 0.3 * velocities[5, :] + 0.1 * velocities[4, :]
    # generate future positions
    future_positions = np.zeros((num_future_positions, 2))
    for i in range(num_future_positions):
        future_positions[i, 0] = positions[-1, 0] + cur_vel[0] * dt * (i+1)
        future_positions[i, 1] = positions[-1, 1] + cur_vel[1] * dt * (i+1)

    return future_positions


def batched_constant_velocity_model(positions, num_future_positions=12, dt=0.1):
    # dt = 0.1  # time step, can be adjusted as needed


    # estimate linear velocity
    velocities = (positions[:, 1:, :2] - positions[:, :-1, :2]) / dt
    cur_vel = 0.6 * velocities[:, 6, :] + 0.3 * velocities[:, 5, :] + 0.1 * velocities[:, 4, :]

    # generate future positions
    future_positions = np.zeros((positions.shape[0], num_future_positions, 2))
    for i in range(num_future_positions):
        future_positions[:, i, 0] = positions[:, -1, 0] + cur_vel[:, 0] * dt * (i+1)
        future_positions[:, i, 1] = positions[:, -1, 1] + cur_vel[:, 1] * dt * (i+1)

    return future_positions
