"""Public PCA/HMM outputs against pristine upstream C executions.

Reads: statistics fixtures from oracle_statistics.cpp and public statistics API.
"""
from pyqmdsp import statistics as st


def test_original_statistics(reference, assert_matches):
    ref = reference("statistics")
    x = ref["inputs"]
    actual = {f"pca_{n}": st.pca(x["pca"], n) for n in (1, 2, 3)}
    actual["inverse"], actual["determinant"] = st.covariance_inverse(x["covariance"])
    actual["gaussian"] = st.gaussian(x["point"], x["mean"], x["covariance"])
    actual["log_gaussian"] = st.gaussian(x["point"], x["mean"], x["covariance"], logarithmic=True)
    actual["posterior"], actual["transition_posterior"], actual["log_likelihood"] = st.forward_backward(
        x["initial"], x["transition"], x["emissions"]
    )
    with st.HMM(x["observations"], 2) as model:
        model.set_parameters(**x["prescribed"])
        actual["set_parameters"] = model.parameters
        actual["decode_before"] = model.decode(x["observations"])
        model.update(x["observations"], x["update_posterior"], x["update_transition_posterior"])
        actual["after_update"] = model.parameters
        actual["decode_updated"] = model.decode(x["observations"])
        model.train(x["observations"])
        actual["after_train"] = model.parameters
        actual["decode_trained"] = model.decode(x["observations"])
    assert_matches(actual, ref["outputs"])


def test_original_random_initialization(reference, assert_matches, seeded_native):
    ref = reference("statistics")
    with st.HMM(ref["inputs"]["observations"], 2) as model:
        assert_matches({"initial_parameters": model.parameters}, ref["stochastic"])
