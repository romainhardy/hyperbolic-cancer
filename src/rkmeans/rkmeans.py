import geomstats.backend as gs
import logging
import numpy as np
import random
import time

from geomstats.geometry.manifold import Manifold
from geomstats.learning._template import TransformerMixin
from geomstats.learning.frechet_mean import FrechetMean
from random import randint
from scipy.stats import rv_discrete
from sklearn.base import BaseEstimator, ClusterMixin


class RiemannianKMeans(TransformerMixin, ClusterMixin, BaseEstimator):
    """Class for k-means clustering on manifolds.

    K-means algorithm using Riemannian manifolds. Optionally supports mini-batch
    updates for scalability on large datasets.

    Parameters
    ----------
    space : Manifold
        Equipped manifold.
    n_clusters : int
        Number of clusters (k value for k-means).
        Optional, default: 8.
    init : str or callable or array-like, shape=[n_clusters, n_features]
        Method for initializing cluster centers.
        The choice 'random' selects initial centers uniformly at random from the training points.
        The choice 'kmeans++' selects centers using the k-means++ strategy for faster convergence.
        If an array of shape (n_clusters, n_features) is provided, it is used as initial centers.
        If a callable is provided, it should take (X, n_clusters) and return an array of shape (n_clusters, n_features).
        Optional, default: 'random'.
    tol : float
        Convergence threshold. Convergence is achieved when the average displacement
        (mean distance) of cluster centers between iterations is less than tol.
        Optional, default: 1e-2.
    max_iter : int
        Maximum number of iterations.
        Optional, default: 100.
    verbose : int
        Verbosity level. If > 0, information is logged at each iteration.
        Optional, default: 0.
    batch_size : int or None
        Size of the mini-batches to use for each iteration. If None or greater than the number of samples, the full dataset is used (full batch k-means).
        Using mini-batches can speed up convergence at the cost of some accuracy.
        Optional, default: None.
    random_state : int or None
        Random seed for centroid initialization and mini-batch sampling. If set, results are reproducible.
        Optional, default: None.

    Notes
    -----
    * Required metric methods: `dist`.
    * If batch_size is specified, each iteration of the fitting process uses a random subset of the data (without replacement within the batch) to update cluster centers via their Fréchet mean.
    """

    def __init__(
        self,
        space,
        n_clusters=8,
        init="random",
        tol=1e-2,
        max_iter=100,
        verbose=0,
        batch_size=None,
        random_state=None,
    ):
        self.space = space
        self.n_clusters = n_clusters
        self.init = init
        self.tol = tol
        self.max_iter = max_iter
        self.verbose = verbose
        self.batch_size = batch_size
        self.random_state = random_state

        self.init_cluster_centers_ = None

        self.mean_estimator = FrechetMean(space=space)
        if isinstance(self.mean_estimator, FrechetMean):
            # Set default FrechetMean parameters for faster convergence
            self.mean_estimator.set(max_iter=100, init_step_size=1.0)

        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def _pick_init_cluster_centers(self, X):
        """Helper to choose initial cluster centers based on `self.init`."""
        n_samples = X.shape[0]

        if isinstance(self.init, str):
            if self.init == "kmeans++":
                # K-means++ initialization
                cluster_centers = [X[randint(0, n_samples - 1)]]
                for i in range(self.n_clusters - 1):
                    dists = gs.array([
                        self.space.metric.dist(cluster_centers[j], X)
                        for j in range(i + 1)
                    ])
                    # Distances to the closest existing center for each point
                    dists_to_closest = gs.amin(dists, axis=0)
                    indices = gs.arange(n_samples)
                    weights = dists_to_closest / gs.sum(dists_to_closest)
                    # Choose a new center index with probability proportional to distance
                    index = rv_discrete(values=(gs.to_numpy(indices), gs.to_numpy(weights))).rvs()
                    cluster_centers.append(X[index])
            elif self.init == "random":
                # Randomly choose n_clusters points as initial centers
                cluster_centers = [
                    X[randint(0, n_samples - 1)] for _ in range(self.n_clusters)
                ]
            else:
                raise ValueError(f"Unknown initialization method '{self.init}'.")
            cluster_centers = gs.stack(cluster_centers, axis=0)
        else:
            # If init is a callable or an array of initial centers
            if callable(self.init):
                cluster_centers = self.init(X, self.n_clusters)
            else:
                cluster_centers = self.init
            if cluster_centers.shape[0] != self.n_clusters:
                raise ValueError("Number of initial centers must equal n_clusters.")
            if cluster_centers.shape[1] != X.shape[1]:
                raise ValueError("Dimensions of initial cluster centers do not match data.")

        return cluster_centers

    def fit(self, X):
        """Compute cluster centers and assign labels to data points.

        Run the k-means clustering algorithm (optionally in mini-batch mode) on the manifold,
        alternating between assigning points to nearest cluster centers and updating the centers.
        
        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Training data.
        
        Returns
        -------
        self : RiemannianKMeans
            The fitted k-means estimator.
        """
        n_samples = X.shape[0]
        if self.verbose > 0:
            logging.info("Initializing...")

        # Set random seed for reproducibility if provided
        if self.random_state is not None:
            random.seed(self.random_state)
            try:
                import numpy as np
                np.random.seed(self.random_state)
            except ImportError:
                pass

        # Initialize cluster centers
        cluster_centers = self._pick_init_cluster_centers(X)
        self.init_cluster_centers_ = gs.copy(cluster_centers)

        # Determine effective batch size
        batch_size = n_samples if self.batch_size is None or self.batch_size >= n_samples else self.batch_size

        # Prepare index list for mini-batches (shuffled for sampling without replacement)
        indices = list(range(n_samples))
        if batch_size < n_samples:
            random.shuffle(indices)
        pointer = 0

        for iteration in range(self.max_iter):
            if self.verbose > 0:
                logging.info(f"Iteration {iteration}...")

            # Sample a mini-batch of points without replacement
            if pointer >= n_samples:
                # All points have been used, reshuffle for a new epoch
                if batch_size < n_samples:
                    random.shuffle(indices)
                pointer = 0
            if pointer + batch_size <= n_samples:
                batch_indices = indices[pointer : pointer + batch_size]
                pointer += batch_size
            else:
                batch_indices = indices[pointer : n_samples]
                pointer = n_samples  # Reached end, will reshuffle next iteration
            X_batch = X[batch_indices]

            # Assign mini-batch points to nearest cluster centers
            dists = [
                gs.to_ndarray(self.space.metric.dist(cluster_centers[i], X_batch), 2, 1)
                for i in range(self.n_clusters)
            ]
            dists = gs.hstack(dists)
            batch_labels = gs.argmin(dists, axis=1)

            # Update cluster centers using the mini-batch assignments
            old_cluster_centers = gs.copy(cluster_centers)
            for i in range(self.n_clusters):
                points_i = X_batch[batch_labels == i]
                if len(points_i) > 0:
                    # Compute Frechet mean of points in cluster i
                    self.mean_estimator.fit(points_i)
                    cluster_centers[i] = self.mean_estimator.estimate_
                else:
                    # No points in mini-batch for cluster i
                    if batch_size == n_samples:
                        # If using full batch (global assignment) and cluster is empty, reinitialize it
                        cluster_centers[i] = X[randint(0, n_samples - 1)]

            # Compute shift of cluster centers and check convergence
            center_shift = self.space.metric.dist(old_cluster_centers, cluster_centers)
            mean_shift = gs.mean(center_shift)
            if self.verbose > 0:
                logging.info(f"Convergence criterion at the end of iteration {iteration} is {mean_shift}.")
            if mean_shift < self.tol:
                if self.verbose > 0:
                    logging.info(f"Convergence reached after {iteration} iterations.")
                break
        else:
            # No convergence within max_iter
            logging.warning(
                f"K-means maximum number of iterations {self.max_iter} reached. The mean may be inaccurate."
            )

        # Final assignment of all points to closest centers and inertia calculation
        dists_full = [
            gs.to_ndarray(self.space.metric.dist(cluster_centers[i], X), 2, 1)
            for i in range(self.n_clusters)
        ]
        dists_full = gs.hstack(dists_full)
        self.labels_ = gs.argmin(dists_full, axis=1)
        # Sum of squared distances to closest cluster center
        min_dists = gs.amin(dists_full, axis=1)
        self.inertia_ = gs.sum(min_dists ** 2)

        self.cluster_centers_ = cluster_centers
        return self

    def predict(self, X):
        """Predict the nearest cluster each sample in X belongs to.

        Parameters
        ----------
        X : array-like, shape=[n_samples, n_features]
            Data points to predict cluster membership for.

        Returns
        -------
        labels : array-like, shape=[n_samples,]
            Index of the cluster each sample is assigned to.
        """
        if self.cluster_centers_ is None:
            raise RuntimeError("fit needs to be called first.")
        dists = gs.stack(
            [self.space.metric.dist(center, X) for center in self.cluster_centers_],
            axis=1
        )
        dists = gs.squeeze(dists)
        labels = gs.argmin(dists, axis=-1)
        return labels


class UnionFind:
    """Optimized Union-Find data structure for merging clusters."""
    def __init__(self, n):
        self.parent = np.arange(n)
        self.num_sets = n

    def find(self, i):
        # Path compression heuristic
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Simple union (no ranking heuristic)
            self.parent[root_j] = root_i
            self.num_sets -= 1
            return True
        return False


def merge_clusters(
    manifold: Manifold,
    initial_assignments: np.ndarray,
    centroids: np.ndarray,
    merge_threshold: float,
    verbose: int = 0,
):
    """Merge clusters that are closer than the merge threshold.

    Parameters
    ----------
    manifold : Manifold
        The manifold on which the data lies
    initial_assignments : array-like
        Initial cluster assignments for each point
    centroids : array-like
        Coordinates of cluster centroids
    merge_threshold : float
        Distance threshold below which clusters are merged

    Returns
    -------
    final_assignments : array-like
        Updated cluster assignments after merging
    n_final_clusters : int
        Number of clusters after merging
    """
    k_init = centroids.shape[0]
    if verbose > 0:
        logging.info(f"Starting cluster merging process for {k_init} initial clusters with threshold {merge_threshold}...")
    start_time = time.time()

    distances = manifold.metric.dist_pairwise(centroids)
    
    # Use Union-Find to group clusters to be merged
    uf = UnionFind(k_init)
    indices_i, indices_j = np.where((distances > 0) & (distances < merge_threshold))

    merged_count = 0
    for i, j in zip(indices_i, indices_j):
        if i < j and uf.union(i, j):
            merged_count += 1

    if verbose > 0:
        logging.info(f"Performed {merged_count} merge operations, resulting in {uf.num_sets} final clusters.")

    # Map old cluster IDs to roots of merged sets
    root_map = {i: uf.find(i) for i in range(k_init)}
    
    # Map root IDs to sequential final cluster IDs
    final_cluster_roots = sorted(list(set(root_map.values())))
    final_id_map = {root: idx for idx, root in enumerate(final_cluster_roots)}

    # Remap assignments to final cluster IDs
    final_assignments = np.array([final_id_map[root_map[old_id]] for old_id in initial_assignments])

    n_final_clusters = uf.num_sets
    if verbose > 0:
        logging.info(f"Cluster merging finished in {time.time() - start_time:.2f} seconds.")

    return final_assignments, n_final_clusters