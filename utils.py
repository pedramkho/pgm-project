import numpy as np


def eight_gaussians(mean_scale: float = 5., cov_scale: float = 0.2, size: int = 256):
    num_gaussians = 8
    # Initiate centers of 8 points on the circle with radius 1
    centers = [
        (0, 1),
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (1, 0),
        (1. / np.sqrt(2), -1. / np.sqrt(2)),
        (0, -1),
        (-1. / np.sqrt(2), -1. / np.sqrt(2)),
        (-1, 0),
        (-1. / np.sqrt(2), 1. / np.sqrt(2))
    ]
    # Scale centers
    centers = [(mean_scale * x, mean_scale * y) for x, y in centers]

    # Define covariance matrices for 8 Gaussians
    diagonal = np.array([[1, 0], [0, 1]]) * cov_scale
    cov_matrices = [diagonal for _ in range(num_gaussians)]

    # Generate points
    dataset = []
    for i in range(num_gaussians):
        points = np.random.multivariate_normal((centers[i]), cov_matrices[i], size)
        dataset.extend(points)
    dataset = np.array(dataset)
    return dataset


def two_gaussians(mean_scale: float = 5., cov_scale: float = 0.2, sizes: list = [256, 256]):
    num_gaussians = 2
    # Initiate centers of 2 points on the circle with radius 1

    centers = [
        (1. / np.sqrt(2), 1. / np.sqrt(2)),
        (-1. / np.sqrt(2), -1. / np.sqrt(2))
    ]
    # Scale centers
    centers = [(mean_scale * x, mean_scale * y) for x, y in centers]

    # Define covariance matrices for 8 Gaussians
    diagonal = np.array([[1, 0], [0, 1]]) * cov_scale
    cov_matrices = [diagonal for i in range(num_gaussians)]

    # Generate points
    dataset = []
    for i in range(num_gaussians):
        points = np.random.multivariate_normal((centers[i]), cov_matrices[i], sizes[i])
        dataset.extend(points)
    dataset = np.array(dataset)
    return dataset

