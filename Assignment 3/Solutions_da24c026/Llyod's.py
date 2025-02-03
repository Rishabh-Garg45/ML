import numpy as np
import pandas as pd
from scipy.linalg import eigh
import matplotlib.pyplot as plt

data = pd.read_csv('cm_dataset_2.csv', header=None).to_numpy()

def Lloyds_scratch(data, k, max_iter=200):
    features = data.shape[1]
    centers = np.random.rand(k, features)  # Random initialization
    error_history = []  # To store error at each iteration

    for _ in range(max_iter):
        labels = np.argmin(np.linalg.norm(data[:, np.newaxis] - centers, axis=2), axis=1)
        new_centers = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centers[i] 
                                for i in range(k)])
        
        # Calculate the error as the sum of distances from points to their cluster centers
        error = np.sum([np.linalg.norm(data[labels == i] - new_centers[i], axis=1).sum() for i in range(k)])
        error_history.append(error)  # Append current error to history
        
        # Break if centers converge
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers, labels, error_history

# Q2 Part a - For K = 2, Try 5 different random initializations
fig, axes = plt.subplots(2, 5, figsize=(15, 6)) 

for i in range(5):
    centers, labels, error_history = Lloyds_scratch(data, 2)
    
    # Plotting clusters in the first row
    ax_clusters = axes[0, i]
    ax_clusters.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', alpha=0.5)
    ax_clusters.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=150)
    ax_clusters.set_title(f'Initialization {i+1}')

    # Plotting the error function in the second row against iterations
    ax_error = axes[1, i]
    ax_error.plot(error_history)
    ax_error.set_title(f'Error vs Iterations - Init {i+1}')
    ax_error.set_xlabel("Iterations")
    ax_error.set_ylabel("Error")

plt.tight_layout()
plt.show()

# Q2 Part b - Running Lloyds for K = 2,3,4,5 with fixed Initialization and plotting Voronoi regions

def plot_voronoi(data, centers, ax, k):
    xmin, xmax = data[:, 0].min(), data[:, 0].max()
    ymin, ymax = data[:, 1].min(), data[:, 1].max()
    x = np.linspace(xmin, xmax, 100)
    y = np.linspace(ymin, ymax, 100)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]  
    dist = np.linalg.norm(grid_points[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
    closest_cluster = np.argmin(dist, axis=1)
    ax.imshow(closest_cluster.reshape(xx.shape), extent=(xmin, xmax, ymin, ymax), origin='lower', cmap='coolwarm', alpha=0.5)
    ax.scatter(data[:, 0], data[:, 1], c='black', alpha=0.25)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='x', s=150)
    ax.set_title(f'Voronoi for k = {k}')

fig, axes = plt.subplots(1, 4, figsize=(18, 5)) 
for idx, k in enumerate([2, 3, 4, 5]):
    np.random.seed(42)
    centers, _, error = Lloyds_scratch(data, k=k)
    plot_voronoi(data, centers, axes[idx], k)

plt.tight_layout()
plt.show()

# Q2 Part c - Kernelized K-Means for Non-linear Clusters

# It can be seen that following dataset is clustered in non linear patterns and normal Lloyds won't give us non linear clusters
# Hence, we will use kernelised K Means.

def kernelised_clustering(data, n_clusters, sigma=1.0):
    n_samples = data.shape[0]
    similarity_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            dist = np.linalg.norm(data[i] - data[j])
            similarity_matrix[i, j] = np.exp(-dist**2 / (2 * sigma**2))
    degree_matrix = np.diag(similarity_matrix.sum(axis=1))
    laplacian = degree_matrix - similarity_matrix
    d_inv_sqrt = np.diag(1.0 / np.sqrt(degree_matrix.diagonal()))
    laplacian_norm = d_inv_sqrt @ laplacian @ d_inv_sqrt
    eigvals, eigvecs = eigh(laplacian_norm)
    eigvecs = eigvecs[:, :n_clusters]
    eigvecs = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
    centers, labels, error = Lloyds_scratch(eigvecs, k=n_clusters)
    return labels

# Plotting the results of kernelized K-means clustering
labels = kernelised_clustering(data, n_clusters=2, sigma=2)
plt.figure(figsize=(8, 6))
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='coolwarm', alpha=0.8)
plt.title("Kernelized K-Means with 2 clusters")
plt.show()
