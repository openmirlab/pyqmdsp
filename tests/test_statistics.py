"""Numerical and lifecycle checks for upstream HMM/PCA. Reads: statistics API."""

import itertools

import numpy as np
import pytest

from pyqmdsp.statistics import HMM, covariance_inverse, forward_backward, gaussian, pca


def test_gaussian_and_inverse_match_closed_form():
    cov = np.array([[2.0, 0.3], [0.3, 1.0]])
    inv, det = covariance_inverse(cov)
    np.testing.assert_allclose(inv, np.linalg.inv(cov), atol=1e-12)
    assert det == pytest.approx(np.linalg.det(cov))
    point = np.array([0.2, -0.6])
    expected = -0.5 * (
        point @ np.linalg.inv(cov) @ point + 2 * np.log(2 * np.pi) + np.log(det)
    )
    assert gaussian(point, [0, 0], cov, logarithmic=True) == pytest.approx(expected)
    assert gaussian(point, [0, 0], cov) == pytest.approx(np.exp(expected))


def test_forward_backward_matches_enumerated_paths():
    initial = np.array([0.6, 0.4])
    transition = np.array([[0.8, 0.2], [0.3, 0.7]])
    emissions = np.array([[0.5, 0.1], [0.3, 0.9], [0.8, 0.2]])
    gamma, xi, ll = forward_backward(initial, transition, emissions)
    expected = np.zeros_like(gamma)
    expected_xi = np.zeros_like(xi)
    total = 0
    for states in itertools.product(range(2), repeat=3):
        weight = initial[states[0]] * emissions[0, states[0]]
        for t in range(1, 3):
            weight *= transition[states[t - 1], states[t]] * emissions[t, states[t]]
        total += weight
        for t, state in enumerate(states):
            expected[t, state] += weight
        for t in range(2):
            expected_xi[t, states[t], states[t + 1]] += weight
    np.testing.assert_allclose(gamma, expected / total)
    np.testing.assert_allclose(xi, expected_xi / total)
    assert ll == pytest.approx(np.log(total))


def test_hmm_known_states_and_lifecycle():
    x = np.array([[-2.0], [-1.8], [2.0], [2.1], [-2.1]])
    hmm = HMM(x, 2)
    hmm.set_parameters(
        initial=[0.5, 0.5],
        transition=[[0.8, 0.2], [0.2, 0.8]],
        means=[[-2], [2]],
        covariance=[[0.1]],
    )
    np.testing.assert_array_equal(hmm.decode(x), [0, 0, 1, 1, 0])
    params = hmm.parameters
    params["means"][:] = 100
    assert hmm.parameters["means"][0, 0] == -2
    hmm.close()
    hmm.close()
    with pytest.raises(RuntimeError, match="closed"):
        hmm.decode(x)


def test_train_and_baum_welch_update():
    x = np.array([[-2.0], [-1.7], [-2.2], [1.8], [2.1], [2.3]])
    with HMM(x, 2) as model:
        model.set_parameters(
            initial=[0.5, 0.5],
            transition=[[0.8, 0.2], [0.2, 0.8]],
            means=[[-2], [2]],
            covariance=[[1.0]],
        )
        emissions = np.array(
            [[gaussian(row, mean, [[1.0]]) for mean in [[-2], [2]]] for row in x]
        )
        gamma, xi, _ = forward_backward([0.5, 0.5], [[0.8, 0.2], [0.2, 0.8]], emissions)
        model.update(x, gamma, xi)
        assert np.isfinite(model.parameters["covariance"]).all()
        model.train(x)
        np.testing.assert_allclose(model.parameters["transition"].sum(axis=1), 1)
        assert len(model.decode(x)) == len(x)


def test_pca_projects_centered_data_and_preserves_variance():
    x = np.random.default_rng(2).normal(size=(20, 3)) @ np.diag([3, 2, 1])
    original = x.copy()
    out = pca(x, 3)
    np.testing.assert_array_equal(x, original)
    assert out.shape == (20, 3)
    np.testing.assert_allclose(out.mean(axis=0), 0, atol=1e-12)
    gram = out.T @ out
    np.testing.assert_allclose(gram, np.diag(np.diag(gram)), atol=1e-10)
    assert np.sum(out**2) == pytest.approx(np.sum((x - x.mean(axis=0)) ** 2))


@pytest.mark.parametrize(
    "cov", [[[0.0]], [[1.0, 2.0], [2.0, 1.0]], [[1.0, 2.0], [0.0, 1.0]]]
)
def test_invalid_covariance_is_rejected(cov):
    with pytest.raises(ValueError):
        covariance_inverse(cov)


def test_invalid_shapes_and_probabilities():
    with pytest.raises(ValueError):
        HMM([[1.0], [1.0]], 2)
    with pytest.raises(ValueError):
        pca([[1.0, 2.0], [2.0, 3.0]], 3)
    with pytest.raises(ValueError):
        forward_backward([1.0], [[1.0]], [[0.0]])
