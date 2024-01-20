#Part 1a) without normalization of the kmtest dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define your custom KMeans function
def custom_kmeans(X, n_clusters, max_iters=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Randomly initialize cluster centroids
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the nearest cluster centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update cluster centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Load the dataset with custom delimiter (multiple spaces)
data = pd.read_csv("C:\KSU\Grad School\Fall 2023\Machine Learning\Project 1\kmtest.csv", sep=r'\s+')

# Convert the DataFrame to a NumPy array
X = data.values

# Specify the values of K you want to try
k_values = [2, 3, 4, 5]

# Create subplots to visualize results for each K value
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, k in enumerate(k_values):
    # Use the custom KMeans function
    labels, centroids = custom_kmeans(X, n_clusters=k, random_state=0)

    # Scatter plot the data points with different colors for each cluster
    axes[i].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
    axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    axes[i].set_title(f'K = {k}')
    axes[i].legend()

plt.tight_layout()
plt.show()

#Part 1 b) with normalization of the kmtest dataset using z-score method
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define your custom KMeans function
def custom_kmeans(X, n_clusters, max_iters=100, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)
    
    # Randomly initialize cluster centroids
    centroids = X[np.random.choice(len(X), n_clusters, replace=False)]
    
    for _ in range(max_iters):
        # Assign each data point to the nearest cluster centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        
        # Update cluster centroids
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        
        centroids = new_centroids
    
    return labels, centroids

# Load the dataset with custom delimiter (multiple spaces)
data = pd.read_csv("C:\KSU\Grad School\Fall 2023\Machine Learning\Project 1\kmtest.csv", sep=r'\s+')

# Extract the relevant columns for clustering
X = data.values

# Perform z-score normalization
mean = np.mean(X, axis=0)
std = np.std(X, axis=0)
X_normalized = (X - mean) / std

# Specify the values of K you want to try
k_values = [2, 3, 4, 5]

# Create subplots to visualize results for each K value
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for i, k in enumerate(k_values):
    # Use the custom KMeans function on normalized data
    labels, centroids = custom_kmeans(X_normalized, n_clusters=k, random_state=0)

    # Scatter plot the data points with different colors for each cluster
    axes[i].scatter(X_normalized[:, 0], X_normalized[:, 1], c=labels, cmap='viridis')
    axes[i].scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Cluster Centers')
    axes[i].set_title(f'K = {k}')
    axes[i].legend()

plt.tight_layout()
plt.show()

