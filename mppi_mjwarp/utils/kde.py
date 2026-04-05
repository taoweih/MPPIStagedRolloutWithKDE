"""Gaussian Kernel Density Estimator using numpy."""

import numpy as np


class gaussian_kde:
    """Gaussian KDE with fixed per-dimension bandwidth.

    Parameters:
        dataset: shape (n_dimensions, n_data).
        bw: scalar or array of shape (n_dimensions,) for per-dimension bandwidth.
        weights: optional weights of shape (n_data,).
    """

    def __init__(self, dataset: np.ndarray, bw=None, weights=None):
        self.dataset = np.atleast_2d(dataset)
        self.d, self.n = self.dataset.shape

        if weights is not None:
            self.weights = np.atleast_1d(weights).astype(np.float64)
            self.weights /= self.weights.sum()
        else:
            self.weights = np.full(self.n, 1.0 / self.n)

        if bw is None:
            bw = 1.0
        if np.isscalar(bw):
            self.bandwidth = bw * np.ones(self.d)
        else:
            self.bandwidth = np.asarray(bw)
            assert self.bandwidth.shape[0] == self.d

    def pdf(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the KDE at the given points.

        Args:
            points: shape (n_dimensions, n_points).

        Returns:
            Density values, shape (n_points,).
        """
        return self.evaluate(points)

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the KDE at the given points."""
        points = np.atleast_2d(points)
        d, m = points.shape

        if d != self.d:
            if d == 1 and m == self.d:
                points = points.reshape(self.d, 1)
                m = 1
            else:
                raise ValueError(
                    f"points have dimension {d}, dataset has dimension {self.d}"
                )

        log_norm = np.sum(-np.log(self.bandwidth)) - 0.5 * self.d * np.log(2 * np.pi)

        # Fully vectorized: (d, m, n) pairwise distances
        diff = self.dataset[:, None, :] - points[:, :, None]
        scaled = diff / self.bandwidth[:, None, None]
        arg = -0.5 * np.sum(scaled ** 2, axis=0)  # (m, n)
        return np.sum(self.weights[None, :] * np.exp(log_norm + arg), axis=1)  # (m,)

    def __call__(self, points: np.ndarray) -> np.ndarray:
        return self.evaluate(points)
