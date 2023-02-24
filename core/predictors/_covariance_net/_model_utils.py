import torch
import numpy as np


def get_rotation_matrix(vel_x, vel_y):
    vel = np.array([vel_x, vel_y])
    if np.linalg.norm(vel) == 0:
        return np.array([[1, 0], [0, 1]])
    vel = vel / np.linalg.norm(vel)

    return np.array([[vel[0], vel[1]], [-vel[1], vel[0]]])


def get_translation_matrix(pos_x, pos_y):
    return np.array([-pos_x, -pos_y])


def get_rotation_translation_matrix(pos_x, pos_y, vel_x, vel_y):
    # return 3x3 matrix
    return np.block(
        [[get_rotation_matrix(vel_x, vel_y), get_translation_matrix(pos_x, pos_y).reshape(-1, 1)], [0, 0, 1]])


def get_cov_mat(a, b, c, coordinates):
    """
    reference http://www.inference.org.uk/mackay/covariance.pdf
    :param a: first parameter of the covariance matrix
    :param b: second parameter of the covariance matrix
    :param c: third parameter of the covariance matrix
    :param coordinates: coordinates of the pedestrians used to get shapes
    :return:  the covariance matrices
    """

    covariance_matrices = (
            torch.cat(
                [
                    torch.exp(a) * torch.cosh(b),
                    torch.sinh(b),
                    torch.sinh(b),
                    torch.exp(-a) * torch.cosh(b),
                ],
                dim=-1,
            )
            * torch.exp(c)
    )

    covariance_matrices = covariance_matrices.reshape(
        coordinates.shape[0], coordinates.shape[1], 2, 2
    )
    return covariance_matrices


def rollout_gru(gru, scene, poses, pred_steps=12):
    """
        :param gru: the gru cell
        :param scene: the scene representation # bs, datalen
        :param poses: the poses to be predicted # bs, times, datalen
        :param pred_steps: the number of steps to be predicted
    """

    outputs = []
    for step in range(pred_steps):
        scene = gru(poses[:, step, :], scene)
        outputs.append(scene)

    return torch.stack(outputs, dim=1)


def nll_loss(pred, cov, gt):
    """
        :param pred: the predicted positions # bs, times, 2
        :param cov: the predicted covariance matrices # bs, times, 2, 2
        :param gt: the ground truth positions # bs, times, 2
        output: the negative log likelihood loss # bs, times
    """
    bs = pred.shape[0]
    times = pred.shape[1]
    pred = pred.reshape(bs * times, 2)
    cov = cov.reshape(bs * times, 2, 2)
    gt = gt.reshape(bs * times, 2)
    loss = -torch.distributions.MultivariateNormal(pred, cov).log_prob(gt)
    return loss.view(bs, times)
