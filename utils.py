import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import seaborn
import os


def save_numpy_files(images, directory, file_name):
    if not os.path.exists(directory):
      os.makedirs(directory)

    with open(os.path.join(directory, file_name), 'wb') as f:
      np.save(f, images)  


def one_gaussian():
    center = (0,0)
    cov_matrix = [[1, 0.8],
                  [0.8, 1]]

    size = 8 * 256
    return np.random.multivariate_normal(center, cov_matrix, size)


def grid_gaussians(mean_scale: float = 2., cov_scale: float = 0.02, size=3 * 265):    
    centers = []
    grid_dim = 3
    move = (grid_dim - 1) / 2
    for i in range(grid_dim):
        for j in range(grid_dim):
            # centers.append((i, j))
            centers.append((i - move, j - move))


    # Scale centers
    centers = [(mean_scale * x, mean_scale * y) for x, y in centers]

    # Define covariance matrices for 8 Gaussians
    diagonal = np.array([[1, 0], [0, 1]]) * cov_scale
    cov_matrices = [diagonal for _ in range(grid_dim * grid_dim)]

    # Generate points
    dataset = []
    for i in range(grid_dim):
        for j in range(grid_dim):
            index = i*grid_dim + j
            points = np.random.multivariate_normal((centers[index]), cov_matrices[index], size)
            dataset.extend(points)
    dataset = np.array(dataset)
    return dataset


def plot_samples(samples, title, color='Greens'):
    xmax = 3
    cols = len(samples)
    bg_color  = seaborn.color_palette(color, n_colors=256)[0]
    plt.figure(figsize=(2*cols, 2))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i+1, sharex=ax, sharey=ax)
        ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap=color, n_levels=15, clip=[[-xmax,xmax]]*2)
        plt.xticks([])
        plt.yticks([])
        
    
    # ax.set_ylabel('epochs')
    plt.gcf().tight_layout()
    plt.savefig(title+'.png', dpi=300)
    plt.show()
    plt.close()

def plot_samples2(samples, title, color='Greens', scatter_color_name='green', sub_titles=None):
    y_fig_size = 2
    if sub_titles:
      y_fig_size = 2.23
    xmax = 5
    cols = len(samples)
    bg_color  = seaborn.color_palette(color, n_colors=256)[0]
    plt.figure(figsize=(2*cols, y_fig_size))
    for i, samps in enumerate(samples):
        if i == 0:
            ax = plt.subplot(1, cols, 1)
        else:
            plt.subplot(1, cols, i+1, sharex=ax, sharey=ax)
        # ax2 = seaborn.kdeplot(samps[:, 0], samps[:, 1], shaded=True, cmap=color, n_levels=15, clip=[[-xmax,xmax]]*2, fill=False)
        plt.scatter(samps[:, 0], samps[:, 1], alpha=0.5, color=scatter_color_name, edgecolor='white', s=40)
        plt.xticks([])
        plt.yticks([])
        if sub_titles:
          plt.title(sub_titles[i])
        
    
    # ax.set_ylabel('epochs')
    plt.gcf().tight_layout()
    plt.savefig(title+'.png', dpi=300)
    plt.show()
    plt.close()

def save_scatter_plot(data, x, y, address):
    # sns.set_theme()

    s = [0.5 for _ in range(x.shape[0])]
    plt.scatter(data[:, 0], data[:, 1], c='r', marker='o', alpha=0.5)
    plt.scatter(x, y, marker='o', alpha=0.5)
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    
    plt.savefig(address, dpi=300)
    plt.clf()

def eight_gaussians(mean_scale: float = 2.0, cov_scale: float = 0.02, size: int = 264):
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


def two_gaussians(mean_scale: float = 2.0, cov_scale: float = 0.02, sizes: list = [4 * 64, 64 * 10]):
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