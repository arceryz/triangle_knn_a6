import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from dataclasses import dataclass

@dataclass
class ExperimentResult:
	S: np.ndarray
	T: np.ndarray
	s_labels: np.ndarray
	knn_labels: np.ndarray
	num_misses: int

def plot_points(points: np.ndarray, labels: np.ndarray, bg_points: np.ndarray=None, bg_labels: np.ndarray=None):
	plt.figure()

	if bg_points is not None and bg_labels is not None:
		bg_colors = np.where(bg_labels == 0, "blue", "red")
		plt.scatter(bg_points[:, 0], bg_points[:, 1], c=bg_colors, s=8, alpha=0.1)
	
	colors = np.where(labels == 0, "blue", "red")
	plt.scatter(points[:, 0], points[:, 1], c=colors, s=8)
	plt.plot([3, 7, 7, 3], [3, 3, 7, 3], color="black")

	plt.xlim(0, 10)
	plt.ylim(0, 10)
	plt.gca().set_aspect("equal")
	plt.xlabel("x")
	plt.ylabel("y")

# Returns 1 if point is in triangle, 0 otherwise. 
# Vertices (3,3), (7,3), (7,7).
def get_point_labels(pts: np.ndarray) -> np.ndarray:
	x, y = pts[:, 0], pts[:, 1]
	inside = (y >= 3) & (x <= 7) & (y <= x)
	return np.where(inside, 0, 1).astype(np.int8)

# Define function to run single experiment of triangle sampling.
# 1. Generates set S with number of points n
# 2. Creates outliers with probability fo, inliers with fi.
# 3. Clusters with knn with k neighbors.
# 4. Computes number of misclassified points and returns this.
def run_experiment_sample(n, n_test, k, fi, fo) -> ExperimentResult:
	# Generate sample of points in S.
	rng = np.random.default_rng()
	S = rng.uniform(0, 10, size=(n, 2))
	s_labels = get_point_labels(S)

	# Flip the labels in S based on inlier/outlier probabilities.
	flip_prob = np.where(s_labels == 0, fi, fo)
	flips = rng.random(n) < flip_prob
	s_labels = np.where(flips, 1 - s_labels, s_labels).astype(np.int8)

	# Sample from K nearest neighbors.
	tree = KDTree(S)
	T = rng.uniform(0, 10, size=(n_test, 2))
	_, nearest_indices = tree.query(T, k=k)

	# Set label as majority of neighbors.
	knn_labels_avg = s_labels[nearest_indices].mean(axis=1)
	knn_labels = (knn_labels_avg > 0.5).astype(np.int8)

	# Compute number of mismatched labels from exact triangle classifier.
	misses = int(np.sum(knn_labels != get_point_labels(T)))
	result = ExperimentResult(S=S, T=T, s_labels=s_labels, knn_labels=knn_labels, num_misses=misses)
	return result

def run_experiment_batch(n, n_test, k, fi, fo) -> int:
	count = 0
	for i in range(30):
		count += run_experiment_sample(n, n_test, k, fi, fo).num_misses
	return int(round(float(count) / 30))