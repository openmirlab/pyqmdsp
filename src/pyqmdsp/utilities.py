"""Utility, conditioning, segmentation, and maths wrappers for qm-dsp."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from . import _native

WindowType = Literal[
    "rectangular",
    "bartlett",
    "hamming",
    "hanning",
    "hann",
    "blackman",
    "blackman_harris",
]
NormaliseMode = Literal["none", "unit_sum", "unit_max"]
FeatureType = Literal["constq", "chroma", "mfcc"]

FEATURE_TYPE_CONSTQ = 1
FEATURE_TYPE_CHROMA = 2
FEATURE_TYPE_MFCC = 3


def _as_float_vector(values: npt.ArrayLike, *, name: str = "values") -> npt.NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _as_float_matrix(values: npt.ArrayLike, *, name: str = "values") -> npt.NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    if 0 in array.shape:
        raise ValueError(f"{name} dimensions must be non-empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _empty_or_float_vector(values: npt.ArrayLike | None, *, name: str) -> npt.NDArray[np.float64]:
    if values is None:
        return np.empty(0, dtype=np.float64)
    return _as_float_vector(values, name=name)


def _as_int_vector(values: npt.ArrayLike, *, name: str = "values") -> npt.NDArray[np.int64]:
    array = np.ascontiguousarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def frequency_for_pitch(
    midi_pitch: int, cents_offset: float = 0.0, concert_a: float = 440.0
) -> float:
    return float(_native.pitch_frequency(int(midi_pitch), float(cents_offset), float(concert_a)))


def pitch_for_frequency(frequency: float, concert_a: float = 440.0) -> tuple[int, float]:
    pitch, cents = _native.pitch_for_frequency(float(frequency), float(concert_a))
    return int(pitch), float(cents)


def window_data(window_type: WindowType, size: int) -> npt.NDArray[np.float64]:
    return np.asarray(_native.window_data(window_type, int(size)))


def cut_window(window_type: WindowType, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.window_cut(window_type, _as_float_vector(values)))


def kaiser_parameters_transition_width(attenuation: float, transition: float) -> tuple[int, float]:
    length, beta = _native.kaiser_parameters_transition_width(float(attenuation), float(transition))
    return int(length), float(beta)


def kaiser_parameters_bandwidth(
    attenuation: float, bandwidth: float, samplerate: float
) -> tuple[int, float]:
    length, beta = _native.kaiser_parameters_bandwidth(
        float(attenuation), float(bandwidth), float(samplerate)
    )
    return int(length), float(beta)


def kaiser_window(length: int, beta: float) -> npt.NDArray[np.float64]:
    return np.asarray(_native.kaiser_window(int(length), float(beta)))


def kaiser_window_transition_width(attenuation: float, transition: float) -> npt.NDArray[np.float64]:
    return np.asarray(_native.kaiser_window_transition_width(float(attenuation), float(transition)))


def kaiser_window_bandwidth(
    attenuation: float, bandwidth: float, samplerate: float
) -> npt.NDArray[np.float64]:
    return np.asarray(
        _native.kaiser_window_bandwidth(float(attenuation), float(bandwidth), float(samplerate))
    )


def cut_kaiser(beta: float, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.kaiser_cut(float(beta), _as_float_vector(values)))


def sinc_window(length: int, p: float) -> npt.NDArray[np.float64]:
    return np.asarray(_native.sinc_window(int(length), float(p)))


def cut_sinc(p: float, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.sinc_cut(float(p), _as_float_vector(values)))


class Decimator:
    def __init__(self, input_length: int, factor: int):
        self._native = _native.UtilitiesDecimator(int(input_length), int(factor))

    @property
    def factor(self) -> int:
        return int(self._native.factor)

    @property
    def input_length(self) -> int:
        return int(self._native.input_length)

    def process(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(self._native.process(_as_float_vector(values)))

    def reset_filter(self) -> None:
        self._native.reset_filter()


class DecimatorB:
    def __init__(self, input_length: int, factor: int):
        self._native = _native.UtilitiesDecimatorB(int(input_length), int(factor))

    @property
    def factor(self) -> int:
        return int(self._native.factor)

    @property
    def input_length(self) -> int:
        return int(self._native.input_length)

    def process(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(self._native.process(_as_float_vector(values)))


class Resampler:
    def __init__(
        self,
        source_rate: int,
        target_rate: int,
        *,
        snr: float = 100.0,
        bandwidth: float | None = None,
    ):
        if bandwidth is None:
            bandwidth = min(source_rate, target_rate) * 0.45
        self._native = _native.UtilitiesResampler(
            int(source_rate), int(target_rate), float(snr), float(bandwidth)
        )

    @property
    def latency(self) -> int:
        return int(self._native.latency)

    def process(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(self._native.process(_as_float_vector(values)))


def resample(source_rate: int, target_rate: int, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.resample(int(source_rate), int(target_rate), _as_float_vector(values)))


class Filter:
    def __init__(self, b: npt.ArrayLike, a: npt.ArrayLike | None = None):
        self._native = _native.UtilitiesFilter(_empty_or_float_vector(a, name="a"), _as_float_vector(b, name="b"))

    @property
    def order(self) -> int:
        return int(self._native.order)

    def process(self, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
        return np.asarray(self._native.process(_as_float_vector(values)))

    def reset(self) -> None:
        self._native.reset()


def filtfilt(
    values: npt.ArrayLike, b: npt.ArrayLike, a: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    return np.asarray(
        _native.filtfilt(_empty_or_float_vector(a, name="a"), _as_float_vector(b, name="b"), _as_float_vector(values))
    )


def df_process(
    values: npt.ArrayLike,
    *,
    b: npt.ArrayLike,
    a: npt.ArrayLike,
    win_pre: int,
    win_post: int,
    alpha_norm_param: int = 2,
    is_median_positive: bool = True,
    delta: float = 0.0,
) -> npt.NDArray[np.float64]:
    return np.asarray(
        _native.df_process(
            _as_float_vector(values),
            _as_float_vector(a, name="a"),
            _as_float_vector(b, name="b"),
            int(win_pre),
            int(win_post),
            int(alpha_norm_param),
            bool(is_median_positive),
            float(delta),
        )
    )


def frames(values: npt.ArrayLike, frame_length: int, hop: int) -> npt.NDArray[np.float64]:
    return np.asarray(_native.frames(_as_float_vector(values), int(frame_length), int(hop)))


def autocorrelation_unbiased(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.autocorrelation_unbiased(_as_float_vector(values)))


def cosine_distance(a: npt.ArrayLike, b: npt.ArrayLike) -> float:
    return float(_native.cosine_distance(_as_float_vector(a, name="a"), _as_float_vector(b, name="b")))


def kl_gaussian(
    mean_a: npt.ArrayLike, var_a: npt.ArrayLike, mean_b: npt.ArrayLike, var_b: npt.ArrayLike
) -> float:
    return float(
        _native.kl_gaussian(
            _as_float_vector(mean_a, name="mean_a"),
            _as_float_vector(var_a, name="var_a"),
            _as_float_vector(mean_b, name="mean_b"),
            _as_float_vector(var_b, name="var_b"),
        )
    )


def kl_distribution(a: npt.ArrayLike, b: npt.ArrayLike, *, symmetrised: bool = False) -> float:
    return float(
        _native.kl_distribution(
            _as_float_vector(a, name="a"), _as_float_vector(b, name="b"), bool(symmetrised)
        )
    )


def math_summary(values: npt.ArrayLike, *, alpha: int = 2) -> dict[str, float]:
    return {k: float(v) for k, v in _native.math_summary(_as_float_vector(values), int(alpha)).items()}


def normalise(values: npt.ArrayLike, mode: NormaliseMode = "unit_max") -> npt.NDArray[np.float64]:
    return np.asarray(_native.normalise(_as_float_vector(values), mode))


def lp_norm(values: npt.ArrayLike, p: int) -> float:
    return float(_native.lp_norm(_as_float_vector(values), int(p)))


def normalise_lp(
    values: npt.ArrayLike, p: int, *, threshold: float = 1e-6
) -> npt.NDArray[np.float64]:
    return np.asarray(_native.normalise_lp(_as_float_vector(values), int(p), float(threshold)))


def adaptive_threshold(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.adaptive_threshold(_as_float_vector(values)))


def circ_shift(values: npt.ArrayLike, shift: int) -> npt.NDArray[np.float64]:
    return np.asarray(_native.circ_shift(_as_float_vector(values), int(shift)))


def argmax(values: npt.ArrayLike) -> tuple[int, float]:
    index, value = _native.argmax(_as_float_vector(values))
    return int(index), float(value)


def scalar_math(x: float, y: float) -> dict[str, float]:
    return {k: float(v) for k, v in _native.scalar_math(float(x), float(y)).items()}


def integer_math(x: int, y: int) -> dict[str, float | bool]:
    result = dict(_native.integer_math(int(x), int(y)))
    result["is_power_of_two"] = bool(result["is_power_of_two"])
    result["next_power_of_two"] = int(result["next_power_of_two"])
    result["previous_power_of_two"] = int(result["previous_power_of_two"])
    result["nearest_power_of_two"] = int(result["nearest_power_of_two"])
    result["gcd"] = int(result["gcd"])
    result["factorial"] = float(result["factorial"])
    return result


def median_filter(size: int, values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.median_filter(int(size), _as_float_vector(values)))


class MedianFilter:
    def __init__(self, size: int, percentile: float = 50.0):
        self._native = _native.UtilitiesMedianFilter(int(size), float(percentile))

    @property
    def size(self) -> int:
        return int(self._native.size)

    def push(self, value: float) -> None:
        self._native.push(float(value))

    def get(self) -> float:
        return float(self._native.get())

    def get_at(self, percentile: float) -> float:
        return float(self._native.get_at(float(percentile)))

    def reset(self) -> None:
        self._native.reset()


@dataclass(frozen=True)
class PolyfitResult:
    coefficients: npt.NDArray[np.float64]
    correlation: float


def polyfit(x: npt.ArrayLike, y: npt.ArrayLike, terms: int) -> PolyfitResult:
    coefficients, correlation = _native.polyfit(
        _as_float_vector(x, name="x"), _as_float_vector(y, name="y"), int(terms)
    )
    return PolyfitResult(np.asarray(coefficients), float(correlation))


def mpeg7_constq(values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    return np.asarray(_native.mpeg7_constq(_as_float_matrix(values)))


def cq_to_chroma(values: npt.ArrayLike, bins: int) -> npt.NDArray[np.float64]:
    return np.asarray(_native.cq_to_chroma(_as_float_matrix(values), int(bins)))


def create_histograms(labels: npt.ArrayLike, bins: int, length: int) -> npt.NDArray[np.float64]:
    return np.asarray(_native.create_histograms(_as_int_vector(labels, name="labels"), int(bins), int(length)))


def cluster_melt(
    histograms: npt.ArrayLike,
    schedule: npt.ArrayLike,
    clusters: int,
    *,
    neighbour_limit: int = 0,
) -> npt.NDArray[np.int64]:
    return np.asarray(
        _native.cluster_melt(
            _as_float_matrix(histograms, name="histograms"),
            _as_float_vector(schedule, name="schedule"),
            int(clusters),
            int(neighbour_limit),
        )
    )


def cluster_segment(
    features: npt.ArrayLike,
    *,
    hmm_states: int,
    histogram_length: int,
    clusters: int,
    neighbour_limit: int = 0,
) -> npt.NDArray[np.int64]:
    return np.asarray(
        _native.cluster_segment(
            _as_float_matrix(features, name="features"),
            int(hmm_states),
            int(histogram_length),
            int(clusters),
            int(neighbour_limit),
        )
    )


def constq_segment(
    features: npt.ArrayLike,
    *,
    bins: int,
    feature_type: FeatureType,
    hmm_states: int,
    histogram_length: int,
    clusters: int,
    neighbour_limit: int = 0,
) -> npt.NDArray[np.int64]:
    if feature_type == "constq":
        feature_code = FEATURE_TYPE_CONSTQ
    elif feature_type == "chroma":
        feature_code = FEATURE_TYPE_CHROMA
    else:
        raise ValueError("feature_type must be 'constq' or 'chroma'")
    return np.asarray(
        _native.constq_segment(
            _as_float_matrix(features, name="features"),
            int(bins),
            feature_code,
            int(hmm_states),
            int(histogram_length),
            int(clusters),
            int(neighbour_limit),
        )
    )


@dataclass(frozen=True)
class ClusterMeltSegmenterParams:
    feature_type: FeatureType = "constq"
    hop_size: float = 0.2
    window_size: float = 0.6
    fmin: int = 62
    fmax: int = 16_000
    nbins: int = 8
    ncomponents: int = 20
    hmm_states: int = 40
    clusters: int = 10
    histogram_length: int = 15
    neighbourhood_limit: int = 20


def _feature_type_code(feature_type: FeatureType) -> int:
    if feature_type == "constq":
        return FEATURE_TYPE_CONSTQ
    if feature_type == "chroma":
        return FEATURE_TYPE_CHROMA
    if feature_type == "mfcc":
        return FEATURE_TYPE_MFCC
    raise ValueError("feature_type must be 'constq', 'chroma', or 'mfcc'")


class ClusterMeltSegmenter:
    def __init__(self, params: ClusterMeltSegmenterParams | None = None):
        if params is None:
            params = ClusterMeltSegmenterParams()
        self.params = params
        self._native = _native.UtilitiesClusterMeltSegmenter(
            _feature_type_code(params.feature_type),
            float(params.hop_size),
            float(params.window_size),
            int(params.fmin),
            int(params.fmax),
            int(params.nbins),
            int(params.ncomponents),
            int(params.hmm_states),
            int(params.clusters),
            int(params.histogram_length),
            int(params.neighbourhood_limit),
        )

    def initialise(self, samplerate: int) -> None:
        self._native.initialise(int(samplerate))

    def get_windowsize(self) -> int:
        return int(self._native.get_windowsize())

    def get_hopsize(self) -> int:
        return int(self._native.get_hopsize())

    def extract_features(self, samples: npt.ArrayLike) -> None:
        self._native.extract_features(_as_float_vector(samples, name="samples"))

    def set_features(self, features: npt.ArrayLike) -> None:
        self._native.set_features(_as_float_matrix(features, name="features"))

    def segment(self, segment_types: int | None = None) -> None:
        if segment_types is None:
            self._native.segment()
        else:
            self._native.segment_types(int(segment_types))

    def clear(self) -> None:
        self._native.clear()

    def get_segmentation(self) -> dict[str, object]:
        return dict(self._native.get_segmentation())

    def get_n_segment_types(self) -> int:
        return int(self._native.get_n_segment_types())


def cluster_melt_segmenter(
    features: npt.ArrayLike, *, samplerate: int, segment_types: int
) -> dict[str, object]:
    return dict(
        _native.cluster_melt_segmenter(
            _as_float_matrix(features, name="features"), int(samplerate), int(segment_types)
        )
    )
