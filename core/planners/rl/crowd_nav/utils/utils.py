import numpy as np

from pyminisim.util import wrap_angle


def point_to_segment_dist(x1, y1, x2, y2, x3, y3):
    """
    Calculate the closest distance between point(x3, y3) and a line segment with two endpoints (x1, y1), (x2, y2)

    """
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        return np.linalg.norm((x3-x1, y3-y1))

    u = ((x3 - x1) * px + (y3 - y1) * py) / (px * px + py * py)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    # (x, y) is the closest point to (x3, y3) on the line segment
    x = x1 + u * px
    y = y1 + u * py

    return np.linalg.norm((x - x3, y-y3))


def subgoal_to_global(subgoal_polar: np.ndarray, robot_pose: np.ndarray) -> np.ndarray:
    x_rel_rot = subgoal_polar[0] * np.cos(subgoal_polar[1])
    y_rel_rot = subgoal_polar[0] * np.sin(subgoal_polar[1])
    theta = robot_pose[2]
    x_rel = x_rel_rot * np.cos(theta) - y_rel_rot * np.sin(theta)
    y_rel = x_rel_rot * np.sin(theta) + y_rel_rot * np.cos(theta)
    x_abs = x_rel + robot_pose[0]
    y_abs = y_rel + robot_pose[1]
    return np.array([x_abs, y_abs])


def transition_to_subgoal_polar(pose_initial: np.ndarray, pose_new: np.ndarray) -> np.ndarray:
    linear = np.linalg.norm(pose_initial[:2] - pose_new[:2])
    angular = np.arctan2(pose_new[1] - pose_initial[1], pose_new[0] - pose_initial[0])
    angular = wrap_angle(angular - pose_initial[2])
    return np.array([linear, angular])
