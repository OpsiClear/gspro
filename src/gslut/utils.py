"""
Utility functions for LUT operations

Provides helper functions for linear interpolation, clustering, and other
operations used by LUT classes.
"""

import torch


def linear_interp_1d(
    x: torch.Tensor, centers: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Perform 1D linear interpolation using sorted cluster centers.

    Args:
        x: Input values to interpolate [N]
        centers: Sorted cluster centers [K]
        values: Precomputed output values at cluster centers [K]

    Returns:
        Interpolated values [N]

    Example:
        >>> centers = torch.tensor([0.0, 0.5, 1.0])
        >>> values = torch.tensor([0.0, 0.25, 1.0])
        >>> x = torch.tensor([0.25, 0.75])
        >>> result = linear_interp_1d(x, centers, values)
        >>> print(result)
        tensor([0.1250, 0.6250])
    """
    indices = torch.searchsorted(centers, x)
    indices = indices.clamp(1, len(centers) - 1)

    left_idx = indices - 1
    right_idx = indices

    left_centers = centers[left_idx]
    right_centers = centers[right_idx]
    left_values = values[left_idx]
    right_values = values[right_idx]

    alpha = (x - left_centers) / (right_centers - left_centers + 1e-8)
    result = left_values + alpha * (right_values - left_values)

    return result


def nearest_neighbor_1d(
    x: torch.Tensor, centers: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """
    Perform 1D nearest neighbor lookup.

    Args:
        x: Input values [N]
        centers: Cluster centers [K] (need not be sorted)
        values: Precomputed output values at cluster centers [K]

    Returns:
        Values at nearest neighbors [N]

    Example:
        >>> centers = torch.tensor([0.0, 0.5, 1.0])
        >>> values = torch.tensor([0.0, 0.25, 1.0])
        >>> x = torch.tensor([0.1, 0.7])
        >>> result = nearest_neighbor_1d(x, centers, values)
        >>> print(result)
        tensor([0.0000, 0.2500])
    """
    x_expanded = x.unsqueeze(1)
    distances = torch.cdist(x_expanded, centers.unsqueeze(1))
    nearest_idx = torch.argmin(distances, dim=1)
    return values[nearest_idx]


def build_kmeans_clusters(
    samples: torch.Tensor, num_clusters: int, random_state: int = 42
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build k-means clusters from samples.

    Args:
        samples: Input samples [N, D] or [N] (will be reshaped to [N, 1])
        num_clusters: Number of clusters to create
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cluster_centers, cluster_labels)
        - cluster_centers: [K, D] tensor of cluster centroids
        - cluster_labels: [N] tensor of cluster assignments

    Example:
        >>> samples = torch.randn(1000, 1)
        >>> centers, labels = build_kmeans_clusters(samples, num_clusters=10)
        >>> print(centers.shape, labels.shape)
        torch.Size([10, 1]) torch.Size([1000])
    """
    from sklearn.cluster import MiniBatchKMeans

    # Ensure samples are 2D
    if samples.dim() == 1:
        samples = samples.unsqueeze(1)

    samples_np = samples.cpu().numpy()

    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, random_state=random_state, batch_size=10000
    )
    kmeans.fit(samples_np)

    centers = torch.from_numpy(kmeans.cluster_centers_).float()
    labels = torch.from_numpy(kmeans.labels_)

    return centers, labels
