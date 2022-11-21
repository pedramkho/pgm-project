import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def save_scatter_plot(data, x, y, address):
    sns.set_theme()

    s = [0.5 for _ in range(x.shape[0])]
    plt.scatter(x, y, marker='o', alpha=0.5)
    plt.scatter(data[:, 0], data[:, 1], c='r', marker='o', alpha=0.5)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    plt.savefig(address)
    plt.clf()

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

