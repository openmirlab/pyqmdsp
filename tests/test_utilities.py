import math

import numpy as np
import pytest

from pyqmdsp import utilities as u


def test_pitch_round_trip_with_cents():
    assert u.frequency_for_pitch(69) == pytest.approx(440.0)
    pitch, cents = u.pitch_for_frequency(445.0)
    assert pitch == 69
    assert 0.0 < cents < 20.0


def test_windows_and_cutters_match_multiplication():
    data = np.ones(8)
    for name in ["rectangular", "bartlett", "hamming", "hanning", "blackman", "blackman_harris"]:
        window = u.window_data(name, 8)
        assert window.shape == (8,)
        assert u.cut_window(name, data) == pytest.approx(window)

    length, beta = u.kaiser_parameters_transition_width(60.0, 0.2)
    assert length > 0
    assert beta > 0
    bandwidth_length, bandwidth_beta = u.kaiser_parameters_bandwidth(60.0, 100.0, 1_000.0)
    assert bandwidth_length > 0
    assert bandwidth_beta == pytest.approx(beta)
    kaiser = u.kaiser_window(length, beta)
    assert kaiser.shape == (length,)
    assert u.cut_kaiser(beta, np.ones(length)) == pytest.approx(kaiser)

    sinc = u.sinc_window(9, 4.0)
    assert sinc.shape == (9,)
    assert sinc[4] == pytest.approx(1.0)
    assert u.cut_sinc(4.0, np.ones(9)) == pytest.approx(sinc)


def test_rate_conversion_algorithms_change_lengths_and_preserve_finite_values():
    impulse = np.zeros(16)
    impulse[0] = 1.0

    decimator = u.Decimator(16, 2)
    decimated = decimator.process(impulse)
    assert decimated.shape == (8,)
    assert np.all(np.isfinite(decimated))
    decimator.reset_filter()

    decimator_b = u.DecimatorB(16, 4)
    decimated_b = decimator_b.process(np.arange(16, dtype=np.float64))
    assert decimated_b.shape == (4,)
    assert np.all(np.isfinite(decimated_b))

    resampled = u.resample(8_000, 4_000, np.sin(np.linspace(0.0, 1.0, 32)))
    assert 12 <= len(resampled) <= 20
    assert np.all(np.isfinite(resampled))

    streaming = u.Resampler(8_000, 16_000)
    doubled = streaming.process(np.ones(16))
    assert len(doubled) >= 16
    assert streaming.latency >= 0


def test_filtering_framing_and_detection_conditioning():
    filt = u.Filter([0.5, 0.5])
    assert filt.order == 1
    assert filt.process([2.0, 4.0, 6.0]) == pytest.approx([1.0, 3.0, 5.0])
    filt.reset()
    assert filt.process([2.0]) == pytest.approx([1.0])

    values = np.array([1.0, 2.0, 4.0, 8.0])
    assert u.filtfilt(values, [1.0], [1.0]) == pytest.approx(values)

    conditioned = u.df_process(
        np.linspace(1.0, 4.0, 8),
        b=[1.0],
        a=[1.0],
        win_pre=1,
        win_post=1,
        is_median_positive=True,
    )
    assert conditioned.shape == (8,)
    assert np.all(conditioned >= 0.0)

    frames = u.frames(np.arange(5, dtype=np.float64), frame_length=3, hop=2)
    assert frames.shape == (3, 3)
    assert frames[0] == pytest.approx([0.0, 1.0, 2.0])
    assert frames[-1] == pytest.approx([4.0, 0.0, 0.0])


def test_math_distance_polyfit_and_median_algorithms():
    data = np.array([1.0, 2.0, 4.0, 8.0])
    assert u.autocorrelation_unbiased(data)[0] == pytest.approx(np.mean(data * data))
    assert u.cosine_distance([1.0, 0.0], [1.0, 0.0]) == pytest.approx(0.0)
    assert u.kl_distribution([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert u.kl_gaussian([0.0], [1.0], [0.0], [1.0]) == pytest.approx(0.0)

    summary = u.math_summary(data)
    assert summary["min"] == pytest.approx(1.0)
    assert summary["max"] == pytest.approx(8.0)
    assert summary["mean"] == pytest.approx(3.75)
    assert summary["sum"] == pytest.approx(15.0)
    assert summary["median"] == pytest.approx(3.0)
    assert summary["alpha_norm"] == pytest.approx(math.sqrt(np.mean(data**2)))

    assert u.normalise([1.0, 2.0, 3.0], "unit_sum") == pytest.approx([1 / 6, 2 / 6, 3 / 6])
    assert u.lp_norm([3.0, 4.0], 2) == pytest.approx(5.0)
    assert u.normalise_lp([3.0, 4.0], 2) == pytest.approx([0.6, 0.8])
    assert u.circ_shift([1.0, 2.0, 3.0, 4.0], 1) == pytest.approx([4.0, 1.0, 2.0, 3.0])
    assert u.argmax([1.0, 5.0, 3.0]) == (1, pytest.approx(5.0))

    scalar = u.scalar_math(-1.6, 2.0)
    assert scalar["round"] == pytest.approx(-2.0)
    assert -math.pi <= scalar["princarg"] < math.pi
    assert scalar["mod"] == pytest.approx(0.4)

    integer = u.integer_math(10, 6)
    assert integer["is_power_of_two"] is False
    assert integer["next_power_of_two"] == 16
    assert integer["previous_power_of_two"] == 8
    assert integer["nearest_power_of_two"] == 8
    assert integer["factorial"] == pytest.approx(math.factorial(10))
    assert integer["gcd"] == 2

    assert u.median_filter(3, [3.0, 1.0, 2.0, 8.0]) == pytest.approx([1.0, 2.0, 2.0, 2.0])
    median = u.MedianFilter(3)
    for value in [3.0, 1.0, 2.0]:
        median.push(value)
    assert median.get() == pytest.approx(2.0)
    assert median.get_at(0.0) == pytest.approx(1.0)
    median.reset()
    assert median.get() == pytest.approx(0.0)

    fit = u.polyfit([0.0, 1.0, 2.0], [1.0, 3.0, 5.0], 2)
    assert fit.coefficients == pytest.approx([1.0, 2.0])
    assert fit.correlation == pytest.approx(1.0)


def test_segmentation_helpers_run_native_algorithms():
    constq = np.array([[1.0, 2.0, 4.0, 8.0], [2.0, 4.0, 8.0, 16.0]], dtype=np.float64)
    mpeg = u.mpeg7_constq(constq)
    assert mpeg.shape == (2, 5)
    assert np.all(np.isfinite(mpeg))

    chroma = u.cq_to_chroma(np.array([[1.0, -2.0, 3.0, -4.0]], dtype=np.float64), bins=2)
    np.testing.assert_allclose(chroma, [[4.0, 6.0]])

    hist = u.create_histograms([0, 1, 1, 0, 1], bins=2, length=3)
    assert hist.shape == (5, 2)
    assert np.all(np.isfinite(hist))

    assignments = u.cluster_melt(hist + 1e-6, [100.0, 50.0], clusters=2)
    assert assignments.shape == (5,)
    assert np.all((1 <= assignments) & (assignments <= 2))

    features = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [5.0, 5.0],
            [5.1, 5.0],
            [5.0, 5.1],
        ],
        dtype=np.float64,
    )
    segmented = u.cluster_segment(features, hmm_states=2, histogram_length=3, clusters=2)
    assert segmented.shape == (6,)
    assert np.all((1 <= segmented) & (segmented <= 2))

    chroma_segmented = u.constq_segment(
        np.abs(np.tile(np.arange(1, 5, dtype=np.float64), (6, 1))),
        bins=2,
        feature_type="chroma",
        hmm_states=2,
        histogram_length=3,
        clusters=2,
    )
    assert chroma_segmented.shape == (6,)
    assert np.all((1 <= chroma_segmented) & (chroma_segmented <= 2))

    constq_segmented = u.constq_segment(
        np.abs(np.tile(np.linspace(1.0, 3.0, 24), (8, 1))),
        bins=8,
        feature_type="constq",
        hmm_states=2,
        histogram_length=3,
        clusters=2,
    )
    assert constq_segmented.shape == (8,)
    assert np.all((1 <= constq_segmented) & (constq_segmented <= 2))

    segmenter = u.ClusterMeltSegmenter(
        u.ClusterMeltSegmenterParams(
            feature_type="chroma",
            clusters=2,
            hmm_states=2,
            histogram_length=3,
            neighbourhood_limit=2,
        )
    )
    segmenter.initialise(11_025)
    assert segmenter.get_windowsize() > 0
    assert segmenter.get_hopsize() > 0
    segmenter.set_features(features)
    segmenter.segment(2)
    stateful = segmenter.get_segmentation()
    assert stateful["samplerate"] == 11_025
    assert stateful["nsegtypes"] == 2
    assert segmenter.get_n_segment_types() == 2
    segmenter.clear()

    audio_segmenter = u.ClusterMeltSegmenter(
        u.ClusterMeltSegmenterParams(
            feature_type="mfcc",
            fmax=5_000,
            clusters=2,
            hmm_states=2,
            histogram_length=3,
            neighbourhood_limit=2,
        )
    )
    audio_segmenter.initialise(11_025)
    audio_segmenter.extract_features(np.sin(np.linspace(0.0, 1.0, audio_segmenter.get_windowsize())))

    high_level = u.cluster_melt_segmenter(features, samplerate=11_025, segment_types=2)
    assert high_level["samplerate"] == 11_025
    assert high_level["nsegtypes"] == 2
    assert isinstance(high_level["segments"], list)


def test_invalid_inputs_are_rejected_before_native_calls():
    with pytest.raises(ValueError, match="one-dimensional"):
        u.autocorrelation_unbiased([[1.0, 2.0]])
    with pytest.raises(ValueError, match="finite"):
        u.cosine_distance([1.0, np.nan], [1.0, 2.0])
    with pytest.raises(Exception, match="same length"):
        u.cosine_distance([1.0], [1.0, 2.0])
    with pytest.raises(Exception, match="power of two"):
        u.Decimator(12, 3)
    with pytest.raises(Exception, match="odd"):
        u.create_histograms([0, 1], bins=2, length=2)
    with pytest.raises(Exception, match="length must not exceed"):
        u.create_histograms([0, 1], bins=2, length=3)
    with pytest.raises(Exception, match="at least two"):
        u.cluster_segment([[0.0, 1.0]], hmm_states=2, histogram_length=1, clusters=1)
    with pytest.raises(Exception, match="histogram_length must not exceed"):
        u.cluster_segment([[0.0], [1.0]], hmm_states=2, histogram_length=3, clusters=1)
    with pytest.raises(Exception, match="clusters must not exceed"):
        u.cluster_segment([[0.0], [1.0]], hmm_states=2, histogram_length=1, clusters=3)
    with pytest.raises(Exception, match="neighbour_limit"):
        u.cluster_segment([[0.0], [1.0]], hmm_states=2, histogram_length=1, clusters=1, neighbour_limit=-1)
    with pytest.raises(Exception, match="multiple of bins"):
        u.constq_segment(
            np.ones((4, 5)),
            bins=3,
            feature_type="chroma",
            hmm_states=2,
            histogram_length=3,
            clusters=2,
        )
    with pytest.raises(Exception, match="at least 21"):
        u.constq_segment(
            np.ones((4, 20)),
            bins=10,
            feature_type="constq",
            hmm_states=2,
            histogram_length=3,
            clusters=2,
        )


def test_stateful_segmenter_rejects_unsafe_state_transitions():
    segmenter = u.ClusterMeltSegmenter(
        u.ClusterMeltSegmenterParams(
            feature_type="mfcc",
            fmax=5_000,
            clusters=2,
            hmm_states=2,
            histogram_length=3,
            neighbourhood_limit=2,
        )
    )
    with pytest.raises(Exception, match="initialised"):
        segmenter.get_windowsize()
    with pytest.raises(Exception, match="segment must be called"):
        segmenter.get_segmentation()

    segmenter.initialise(11_025)
    with pytest.raises(Exception, match="only be initialised once"):
        segmenter.initialise(11_025)
    with pytest.raises(Exception, match="get_windowsize"):
        segmenter.extract_features(np.ones(segmenter.get_windowsize() - 1))
    segmenter.extract_features(np.ones(segmenter.get_windowsize()))
    with pytest.raises(Exception, match="after extract_features"):
        segmenter.set_features(np.ones((3, 2)))

    feature_segmenter = u.ClusterMeltSegmenter(
        u.ClusterMeltSegmenterParams(
            feature_type="chroma",
            clusters=2,
            hmm_states=2,
            histogram_length=3,
            neighbourhood_limit=2,
        )
    )
    feature_segmenter.initialise(11_025)
    feature_segmenter.set_features(np.ones((3, 2)))
    feature_segmenter.clear()
    with pytest.raises(Exception, match="after set_features"):
        feature_segmenter.extract_features(np.ones(feature_segmenter.get_windowsize()))

    low_cqt = u.ClusterMeltSegmenter(
        u.ClusterMeltSegmenterParams(
            feature_type="constq",
            fmin=1_000,
            fmax=1_200,
            nbins=1,
            clusters=2,
            hmm_states=2,
            histogram_length=3,
        )
    )
    with pytest.raises(Exception, match="at least 21"):
        low_cqt.initialise(11_025)
