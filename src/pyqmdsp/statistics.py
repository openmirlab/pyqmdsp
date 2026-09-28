"""PCA and Gaussian HMM algorithms from qm-dsp.

Reads: _native. Matrix inputs have observations in rows and features in columns.
HMM initialization follows upstream's time-seeded RNG; it is not deterministic.
"""

import numpy as np

from . import _native


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or 0 in result.shape or not np.isfinite(result).all():
        raise ValueError("expected a nonempty finite matrix")
    return result.tolist()


def pca(data, components):
    """Project rows onto upstream PCA components; signs follow upstream eigenvectors."""
    return np.asarray(_native.statistics_pca(_matrix(data), components))


def covariance_inverse(covariance):
    """Return (inverse, determinant) for a symmetric positive definite covariance."""
    inverse, determinant = _native.statistics_invert(_matrix(covariance))
    return np.asarray(inverse), determinant


def gaussian(value, mean, covariance, *, logarithmic=False):
    """Evaluate the native multivariate Gaussian density or its natural logarithm."""
    return _native.statistics_gaussian(
        np.asarray(value, dtype=float).tolist(),
        np.asarray(mean, dtype=float).tolist(),
        _matrix(covariance),
        logarithmic,
    )


def forward_backward(initial, transition, emissions):
    """Return posterior (T,N), transition posterior (T-1,N,N), log likelihood.

    Emissions are strictly positive observation likelihoods, shape (T,N).
    Initial and transition rows are nonnegative probabilities summing to one.
    """
    gamma, xi, log_likelihood = _native.statistics_forward_backward(
        np.asarray(initial, dtype=float).tolist(),
        _matrix(transition),
        _matrix(emissions),
    )
    gamma = np.asarray(gamma)
    return (
        gamma,
        np.asarray(xi).reshape(-1, gamma.shape[1], gamma.shape[1]),
        log_likelihood,
    )


class HMM:
    """Gaussian HMM with a shared covariance, preserving qm-dsp training behavior.

    Initialization uses upstream's global time-seeded random generator. Native
    operations keep the GIL to serialize this generator. Supply set_parameters
    for repeatable decoding. Training uses upstream's stopping rule (<=50 steps).
    """

    def __init__(self, data, states):
        self._impl = _native.NativeHMM(_matrix(data), states)

    @property
    def parameters(self):
        """Independent NumPy copies of initial, transition, means and covariance."""
        return {
            key: np.asarray(value) for key, value in self._impl.parameters().items()
        }

    def set_parameters(self, *, initial, transition, means, covariance):
        self._impl.set_parameters(
            np.asarray(initial, dtype=float).tolist(),
            _matrix(transition),
            _matrix(means),
            _matrix(covariance),
        )

    def train(self, data):
        self._impl.train(_matrix(data))
        return self

    def decode(self, data):
        return np.asarray(self._impl.decode(_matrix(data)), dtype=np.int64)

    def update(self, data, posterior, transition_posterior):
        """Perform one upstream Baum-Welch update with caller-supplied posteriors."""
        self._impl.update(
            _matrix(data),
            _matrix(posterior),
            np.asarray(transition_posterior, dtype=float).tolist(),
        )
        return self

    def print(self):
        """Print the upstream parameter representation to native stdout."""
        self._impl.print()

    def close(self):
        """Release the model; repeated close is harmless."""
        self._impl.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


__all__ = ["HMM", "covariance_inverse", "forward_backward", "gaussian", "pca"]
