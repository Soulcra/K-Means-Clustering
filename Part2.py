#Part 2 Clustering the dataset iris.csv, all necessary code including plotting the best, worst and original clustering results.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

# Load the Iris dataset
iris_df = pd.read_csv("C:\KSU\Grad School\Fall 2023\Machine Learning\Project 1\iris.csv")

# Select columns 1, 2, 3, and 4
X = iris_df.iloc[:, [0, 1, 2, 3]].values

# Define your custom K-means function
def custom_kmeans(X, n_clusters, max_iters=100, num_runs=5):
    best_labels = None
    best_centers = None
    best_inertia = float('inf')
    
    for run in range(num_runs):
        # Randomly initialize cluster centroids
        np.random.seed(run)
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
        
        # Calculate the inertia (within-cluster sum of squares)
        inertia = np.sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(n_clusters)])
        
        # Update best result if necessary
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels
            best_centers = centroids
    
    return best_labels, best_centers

# Perform K-means clustering with K=3 (Best Result)
best_labels, best_centers = custom_kmeans(X, n_clusters=3)

# Run the custom K-means algorithm at least five times to find the worst result (Random Initialization)
worst_labels = None
worst_centers = None
worst_inertia = float('inf')  # Initialize with a large value

for run in range(5):
    # Custom K-means implementation
    centroids = X[np.random.choice(len(X), 3, replace=False)]
    for _ in range(100):  # Max iterations
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(3)])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    inertia = np.sum([np.sum((X[labels == i] - centroids[i]) ** 2) for i in range(3)])
    
    if inertia < worst_inertia:
        worst_inertia = inertia
        worst_labels = labels
        worst_centers = centroids

# Plot all three results using attributes 3 and 4
plt.figure(figsize=(18, 5))

# Plot the best clustering result
plt.subplot(131)
plt.scatter(X[:, 2], X[:, 3], c=best_labels, cmap='viridis')
plt.scatter(best_centers[:, 2], best_centers[:, 3], c='red', marker='x', s=200, label='Cluster Centers (Best)')
plt.title('Best Clustering Result (Attributes 3 and 4)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()

# Plot the worst clustering result
plt.subplot(132)
plt.scatter(X[:, 2], X[:, 3], c=worst_labels, cmap='viridis')
plt.scatter(worst_centers[:, 2], worst_centers[:, 3], c='blue', marker='x', s=200, label='Cluster Centers (Worst)')
plt.title('Worst Clustering Result (Attributes 3 and 4)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.legend()

# Plot the original data points
plt.subplot(133)
plt.scatter(X[:, 2], X[:, 3], c='gray', cmap='viridis')
plt.title('Original Data (Attributes 3 and 4)')
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')

plt.tight_layout()
plt.show()

# Calculate the distance between the centers of the best result and original centers
original_cluster_centers = X[np.random.choice(len(X), 3, replace=False)]  # Use random points as placeholder for original centers
distance = euclidean(best_centers.flatten(), original_cluster_centers.flatten())

print(f"Distance between best result centers and original centers: {distance}")
