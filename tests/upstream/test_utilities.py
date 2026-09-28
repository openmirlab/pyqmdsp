"""Parity tests for pyqmdsp.utilities against pristine qm-dsp outputs."""

import numpy as np

from pyqmdsp import utilities as u


def _segmentation_vectors(segmentation):
    return {
        "nsegtypes": segmentation["nsegtypes"],
        "samplerate": segmentation["samplerate"],
        "starts": [item["start"] for item in segmentation["segments"]],
        "ends": [item["end"] for item in segmentation["segments"]],
        "types": [item["type"] for item in segmentation["segments"]],
    }


def _base_actual():
    actual = {
        "frequency_for_pitch": u.frequency_for_pitch(69, 12.5, 440.0),
    }
    pitch, cents = u.pitch_for_frequency(445.0, 440.0)
    actual["pitch_for_frequency"] = pitch
    actual["pitch_cents"] = cents

    ones8 = np.ones(8)
    actual["windows"] = {
        "rectangular": u.window_data("rectangular", 8),
        "bartlett": u.window_data("bartlett", 8),
        "hamming": u.window_data("hamming", 8),
        "hanning": u.window_data("hanning", 8),
        "blackman": u.window_data("blackman", 8),
        "blackman_harris": u.window_data("blackman_harris", 8),
        "hamming_cut": u.cut_window("hamming", ones8),
    }

    length, beta = u.kaiser_parameters_transition_width(60.0, 0.2)
    bandwidth_length, bandwidth_beta = u.kaiser_parameters_bandwidth(60.0, 100.0, 1000.0)
    actual["kaiser"] = {
        "transition_length": length,
        "transition_beta": beta,
        "bandwidth_length": bandwidth_length,
        "bandwidth_beta": bandwidth_beta,
        "window": u.kaiser_window(length, beta),
        "transition_alias_window": u.kaiser_window_transition_width(60.0, 0.2),
        "bandwidth_alias_window": u.kaiser_window_bandwidth(60.0, 100.0, 1000.0),
        "cut": u.cut_kaiser(beta, ones8),
    }
    actual["sinc"] = {
        "window": u.sinc_window(9, 4.0),
        "cut": u.cut_sinc(4.0, np.ones(9)),
    }
    return actual


def _rate_actual(inputs):
    decimator_input = inputs["decimator_input"]
    decimator_input_b = inputs["decimator_input_b"]
    resampler_input = inputs["resampler_input"]
    decimator = u.Decimator(16, 2)
    decimator_first = decimator.process(decimator_input)
    decimator_second = decimator.process(decimator_input_b)
    decimator.reset_filter()
    decimator_after_reset = decimator.process(decimator_input)

    decimator_b = u.DecimatorB(16, 4)
    decimator_b_first = decimator_b.process(decimator_input)
    decimator_b_second = decimator_b.process(decimator_input_b)
    up = u.Resampler(8000, 16000)
    up_chunk_a = up.process(resampler_input[:777])
    up_chunk_b = up.process(resampler_input[777:1554])
    up_chunk_c = up.process(resampler_input[1554:])
    down = u.Resampler(16000, 8000, snr=90.0, bandwidth=3000.0)
    down_out = down.process(resampler_input)
    actual = {
        "decimator_factor": decimator.factor,
        "decimator_first": decimator_first,
        "decimator_second_history": decimator_second,
        "decimator_after_reset": decimator_after_reset,
        "decimator_b_factor": decimator_b.factor,
        "decimator_b_first": decimator_b_first,
        "decimator_b_second_history": decimator_b_second,
        "resample_one_off": u.resample(8000, 11025, resampler_input),
        "resampler_up_latency": up.latency,
        "resampler_up_chunk_a": up_chunk_a,
        "resampler_up_chunk_b": up_chunk_b,
        "resampler_up_chunk_c": up_chunk_c,
        "resampler_down_latency": down.latency,
        "resampler_down": down_out,
    }
    assert np.any(np.abs(np.concatenate([up_chunk_a, up_chunk_b, up_chunk_c])) > 1e-9)
    assert np.any(np.abs(down_out) > 1e-9)
    return actual


def _conditioning_actual(inputs):
    values = inputs
    filt = u.Filter([0.25, 0.5, 0.25])
    filter_first = filt.process(values)
    filter_second = filt.process(values)
    filt.reset()
    filter_after_reset = filt.process(values)
    iir = u.Filter([0.5, 0.25], [1.0, -0.25])
    frames = u.frames(values, 4, 3)
    return {
        "filter_order": filt.order,
        "filter_first": filter_first,
        "filter_second_history": filter_second,
        "filter_after_reset": filter_after_reset,
        "iir_filter": iir.process(values),
        "filtfilt": u.filtfilt(values, [0.25, 0.5, 0.25]),
        "df_process": u.df_process(
            values,
            b=[1.0],
            a=[1.0],
            win_pre=1,
            win_post=1,
            alpha_norm_param=2,
            is_median_positive=True,
            delta=0.01,
        ),
        "framer": {
            "frames": frames.shape[0],
            "data": frames,
        },
    }


def _maths_actual(inputs):
    values = inputs
    summary = u.math_summary(values, alpha=2)
    argmax_index, argmax_value = u.argmax(values)
    median = u.MedianFilter(3, 75.0)
    median_outputs = []
    for value in values:
        median.push(value)
        median_outputs.append(median.get())
    at_25 = median.get_at(25.0)
    median.reset()
    polyfit_x = [-2.0, -1.0, 0.0, 1.0, 2.0]
    polyfit_y = [5.0, 2.5, 1.5, 2.0, 4.0]
    fit = u.polyfit(polyfit_x, polyfit_y, 3)
    return {
        "autocorrelation_unbiased": u.autocorrelation_unbiased(values),
        "cosine_distance": u.cosine_distance([1.0, 2.0, 3.0], [3.0, 2.0, 1.0]),
        "kl_gaussian": u.kl_gaussian([0.0, 1.0], [1.0, 2.0], [0.5, 0.5], [1.5, 2.5]),
        "kl_distribution": u.kl_distribution([0.2, 0.3, 0.5], [0.3, 0.3, 0.4]),
        "kl_distribution_sym": u.kl_distribution(
            [0.2, 0.3, 0.5], [0.3, 0.3, 0.4], symmetrised=True
        ),
        "summary": summary,
        "normalise_sum": u.normalise(values, "unit_sum"),
        "normalise_max": u.normalise(values, "unit_max"),
        "lp_norm": u.lp_norm(values, 3),
        "normalise_lp": u.normalise_lp(values, 2, threshold=1e-6),
        "adaptive_threshold": u.adaptive_threshold(values),
        "circ_shift": u.circ_shift(values, 2),
        "argmax": {"index": argmax_index, "value": argmax_value},
        "scalar": u.scalar_math(-1.6, 2.0),
        "integer": u.integer_math(16, 30),
        "median_filter": u.median_filter(3, values),
        "median_stateful": {
            "outputs": median_outputs,
            "at_25": at_25,
            "after_reset": median.get(),
        },
        "polyfit": {
            "coefficients": fit.coefficients,
            "correlation": fit.correlation,
        },
    }


def _segmentation_deterministic_actual(inputs):
    constq = inputs["constq_matrix"]
    return {
        "mpeg7_constq": u.mpeg7_constq(constq),
        "cq_to_chroma": u.cq_to_chroma(constq, 2),
        "create_histograms": u.create_histograms([0, 1, 1, 0, 1], bins=2, length=3),
    }


def test_utilities_deterministic_upstream_parity(reference, assert_matches):
    fixture = reference("utilities")
    inputs = fixture["inputs"]
    expected = fixture["outputs"]
    actual = {
        "base": _base_actual(),
        "rate": _rate_actual(inputs["rate"]),
        "conditioning": _conditioning_actual(inputs["conditioning"]),
        "maths": _maths_actual(inputs["maths"]),
        "segmentation_deterministic": _segmentation_deterministic_actual(
            inputs["segmentation_deterministic"]
        ),
    }
    assert set(expected) == set(actual) | {"segmentation_stochastic"}
    assert_matches(actual, {key: expected[key] for key in actual})


def _segmentation_stochastic_actual(inputs, seed):
    seed()
    actual = {}
    actual["cluster_melt"] = u.cluster_melt(
        inputs["histograms"], [100.0, 50.0], 2, neighbour_limit=0
    )
    actual["cluster_segment"] = u.cluster_segment(
        inputs["features"], hmm_states=2, histogram_length=3, clusters=2, neighbour_limit=0
    )
    actual["constq_segment_chroma"] = u.constq_segment(
        inputs["chroma_features"],
        bins=12,
        feature_type="chroma",
        hmm_states=6,
        histogram_length=11,
        clusters=3,
        neighbour_limit=0,
    )
    actual["constq_segment_constq"] = u.constq_segment(
        inputs["constq_features"],
        bins=12,
        feature_type="constq",
        hmm_states=6,
        histogram_length=11,
        clusters=3,
        neighbour_limit=0,
    )

    params = u.ClusterMeltSegmenterParams(
        feature_type="chroma",
        hop_size=0.05,
        window_size=0.1,
        hmm_states=2,
        clusters=2,
        histogram_length=3,
        neighbourhood_limit=2,
    )
    segmenter = u.ClusterMeltSegmenter(params)
    segmenter.initialise(11025)
    segmenter.set_features(inputs["features"])
    segmenter.segment()
    actual["stateful_feature"] = {
        "windowsize": segmenter.get_windowsize(),
        "hopsize": segmenter.get_hopsize(),
        "n_segment_types": segmenter.get_n_segment_types(),
        "default_segmentation": _segmentation_vectors(segmenter.get_segmentation()),
    }
    segmenter.clear()
    segmenter.set_features(inputs["features"])
    segmenter.segment(2)
    actual["stateful_feature"]["after_clear_reuse_segmentation"] = _segmentation_vectors(
        segmenter.get_segmentation()
    )

    factory = u.cluster_melt_segmenter(inputs["features"], samplerate=11025, segment_types=2)
    actual["factory_cluster_melt_segmenter"] = _segmentation_vectors(factory)

    audio_params = u.ClusterMeltSegmenterParams(
        feature_type="mfcc",
        hop_size=0.05,
        window_size=0.05,
        fmax=5000,
        hmm_states=2,
        clusters=2,
        histogram_length=3,
        neighbourhood_limit=2,
    )
    audio_segmenter = u.ClusterMeltSegmenter(audio_params)
    audio_segmenter.initialise(11025)
    for window in inputs["stateful_audio_windows"]:
        audio_segmenter.extract_features(window)
    audio_segmenter.segment(2)
    actual["stateful_audio"] = {
        "windowsize": audio_segmenter.get_windowsize(),
        "hopsize": audio_segmenter.get_hopsize(),
        "n_segment_types": audio_segmenter.get_n_segment_types(),
        "segmentation": _segmentation_vectors(audio_segmenter.get_segmentation()),
    }
    return actual


def test_utilities_stochastic_upstream_parity(reference, assert_matches, seeded_native):
    fixture = reference("utilities")
    actual = _segmentation_stochastic_actual(fixture["inputs"]["segmentation_stochastic"], seeded_native)
    assert_matches(actual, fixture["outputs"]["segmentation_stochastic"])
    # The pristine CHROMA path still collapses to one label for the high-contrast
    # pitch-class fixture. Keep parity strict and document the upstream collapse.
    assert len(set(fixture["outputs"]["segmentation_stochastic"]["constq_segment_constq"])) >= 2
