"""Rhythm, onset, and tempo algorithms from qm-dsp."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native

DF_HFC = int(_native.DF_HFC)
DF_SPECDIFF = int(_native.DF_SPECDIFF)
DF_PHASEDEV = int(_native.DF_PHASEDEV)
DF_COMPLEXSD = int(_native.DF_COMPLEXSD)
DF_BROADBAND = int(_native.DF_BROADBAND)


def _array_1d(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        msg = f"{name} must be one-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain finite values"
        raise ValueError(msg)
    return array


def _matrix(values: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2:
        msg = f"{name} must be two-dimensional"
        raise ValueError(msg)
    if not np.all(np.isfinite(array)):
        msg = f"{name} must contain finite values"
        raise ValueError(msg)
    return array


def _mono_audio(audio: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim == 1:
        mono = array
    elif array.ndim == 2:
        mono = np.mean(array, axis=1)
    else:
        msg = "audio must be 1-D or 2-D (frames[, channels])"
        raise ValueError(msg)
    return _array_1d(mono, "audio")


def _filter_coefficients(values: ArrayLike | None, fallback: NDArray[np.float64]) -> NDArray[np.float64]:
    if values is None:
        return fallback.copy()
    return _array_1d(values, "filter coefficients")


DEFAULT_FILTER_A = np.asarray(_native.rhythm_default_filter_a(), dtype=np.float64)
DEFAULT_FILTER_B = np.asarray(_native.rhythm_default_filter_b(), dtype=np.float64)


class DetectionFunction:
    """Stateful qm-dsp onset detection function."""

    def __init__(
        self,
        step_size: int,
        frame_length: int,
        df_type: int = DF_COMPLEXSD,
        *,
        db_rise: float = 3.0,
        adaptive_whitening: bool = False,
        whitening_relax_coeff: float = -1.0,
        whitening_floor: float = -1.0,
    ) -> None:
        self._native = _native.RhythmDetectionFunction(
            int(step_size),
            int(frame_length),
            int(df_type),
            float(db_rise),
            bool(adaptive_whitening),
            float(whitening_relax_coeff),
            float(whitening_floor),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._native.parameters)

    def process_time_domain(self, samples: ArrayLike) -> float:
        return float(self._native.process_time_domain(_array_1d(samples, "samples")))

    def process_frequency_domain(self, reals: ArrayLike, imags: ArrayLike) -> float:
        return float(
            self._native.process_frequency_domain(
                _array_1d(reals, "reals"),
                _array_1d(imags, "imags"),
            )
        )

    def spectrum_magnitude(self) -> NDArray[np.float64]:
        return np.asarray(self._native.spectrum_magnitude(), dtype=np.float64)


class PeakPicking:
    """Stateful qm-dsp onset peak picker."""

    def __init__(
        self,
        length: int,
        tau: float,
        *,
        alpha: int = 3,
        cutoff: float = 0.4,
        lp_order: int = 2,
        lp_a: ArrayLike | None = None,
        lp_b: ArrayLike | None = None,
        win_pre: int = 3,
        win_post: int = 3,
        quad_a: float = 0.0,
        quad_b: float = 0.0,
        quad_c: float = 0.0,
        delta: float = 0.0,
    ) -> None:
        self._native = _native.RhythmPeakPicking(
            int(length),
            float(tau),
            int(alpha),
            float(cutoff),
            int(lp_order),
            _filter_coefficients(lp_a, DEFAULT_FILTER_A),
            _filter_coefficients(lp_b, DEFAULT_FILTER_B),
            int(win_pre),
            int(win_post),
            float(quad_a),
            float(quad_b),
            float(quad_c),
            float(delta),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        params = dict(self._native.parameters)
        params["lp_a"] = np.asarray(params["lp_a"], dtype=np.float64)
        params["lp_b"] = np.asarray(params["lp_b"], dtype=np.float64)
        return params

    def process(self, detection_function: ArrayLike) -> dict[str, NDArray[np.float64]]:
        onsets, processed = self._native.process(_array_1d(detection_function, "detection_function"))
        return {
            "onsets": np.asarray(onsets, dtype=np.int64),
            "processed": np.asarray(processed, dtype=np.float64),
        }


class TempoTrack:
    """Legacy qm-dsp tempo tracker."""

    def __init__(
        self,
        *,
        win_length: int = 512,
        lag_length: int = 128,
        alpha: int = 3,
        lp_order: int = 2,
        lp_a: ArrayLike | None = None,
        lp_b: ArrayLike | None = None,
        win_pre: int = 3,
        win_post: int = 3,
    ) -> None:
        self._native = _native.RhythmTempoTrack(
            int(win_length),
            int(lag_length),
            int(alpha),
            int(lp_order),
            _filter_coefficients(lp_a, DEFAULT_FILTER_A),
            _filter_coefficients(lp_b, DEFAULT_FILTER_B),
            int(win_pre),
            int(win_post),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        params = dict(self._native.parameters)
        params["lp_a"] = np.asarray(params["lp_a"], dtype=np.float64)
        params["lp_b"] = np.asarray(params["lp_b"], dtype=np.float64)
        return params

    def process(self, detection_function: ArrayLike) -> dict[str, NDArray[np.float64]]:
        beats, tempo = self._native.process(_array_1d(detection_function, "detection_function"))
        return {
            "beats": np.asarray(beats, dtype=np.int64),
            "tempo": np.asarray(tempo, dtype=np.float64),
        }


class TempoTrackV2:
    """Davies and Plumbley qm-dsp tempo tracker."""

    def __init__(self, sample_rate: float, df_increment: int) -> None:
        self._native = _native.RhythmTempoTrackV2(float(sample_rate), int(df_increment))

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._native.parameters)

    def calculate_beat_period(
        self,
        detection_function: ArrayLike,
        *,
        input_tempo: float = 120.0,
        constrain_tempo: bool = False,
    ) -> dict[str, NDArray[np.float64]]:
        beat_period, tempi = self._native.calculate_beat_period(
            _array_1d(detection_function, "detection_function"),
            float(input_tempo),
            bool(constrain_tempo),
        )
        return {
            "beat_period": np.asarray(beat_period, dtype=np.float64),
            "tempi": np.asarray(tempi, dtype=np.float64),
        }

    def calculate_beats(
        self,
        detection_function: ArrayLike,
        beat_period: ArrayLike,
        *,
        alpha: float = 0.9,
        tightness: float = 4.0,
    ) -> NDArray[np.float64]:
        return np.asarray(
            self._native.calculate_beats(
                _array_1d(detection_function, "detection_function"),
                _array_1d(beat_period, "beat_period"),
                float(alpha),
                float(tightness),
            ),
            dtype=np.float64,
        )


class DownBeat:
    """qm-dsp downbeat estimator."""

    def __init__(
        self,
        original_sample_rate: float,
        decimation_factor: int,
        df_increment: int,
    ) -> None:
        self._native = _native.RhythmDownBeat(
            float(original_sample_rate),
            int(decimation_factor),
            int(df_increment),
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return dict(self._native.parameters)

    def set_beats_per_bar(self, beats_per_bar: int) -> None:
        self._native.set_beats_per_bar(int(beats_per_bar))

    def find_down_beats(self, audio: ArrayLike, beats_frames: ArrayLike) -> NDArray[np.int64]:
        return np.asarray(
            self._native.find_down_beats(
                _array_1d(audio, "audio"),
                _array_1d(beats_frames, "beats_frames"),
            ),
            dtype=np.int64,
        )

    def push_audio_block(self, audio: ArrayLike) -> None:
        self._native.push_audio_block(_array_1d(audio, "audio"))

    def buffered_audio(self) -> NDArray[np.float64]:
        return np.asarray(self._native.buffered_audio(), dtype=np.float64)

    def reset_audio_buffer(self) -> None:
        self._native.reset_audio_buffer()

    def beat_sd(self) -> NDArray[np.float64]:
        return np.asarray(self._native.beat_sd(), dtype=np.float64)


class BeatSpectrum:
    """qm-dsp rhythmic self-similarity vector."""

    def __init__(self) -> None:
        self._native = _native.RhythmBeatSpectrum()

    def process(self, matrix: ArrayLike) -> NDArray[np.float64]:
        return np.asarray(self._native.process(_matrix(matrix, "matrix")), dtype=np.float64)


def beats(audio: ArrayLike, sample_rate: int) -> dict[str, NDArray[np.float64] | float]:
    """Return beat times, median BPM, and instantaneous tempo curve.

    `TempoTrackV2` needs more than 640 detection-function frames after the BeatTrack-compatible
    trailing-zero trim and first-two-frame discard. Shorter material returns empty beat arrays and
    `0.0` BPM because upstream's Viterbi period tracker has fewer than two tempo-analysis frames.
    """
    beats_s, bpm, tempo_curve = _native.rhythm_beats(_mono_audio(audio), int(sample_rate))
    return {
        "beats_s": np.asarray(beats_s, dtype=np.float64),
        "bpm": float(bpm),
        "tempo_curve": np.asarray(tempo_curve, dtype=np.float64),
    }


__all__ = [
    "DEFAULT_FILTER_A",
    "DEFAULT_FILTER_B",
    "DF_BROADBAND",
    "DF_COMPLEXSD",
    "DF_HFC",
    "DF_PHASEDEV",
    "DF_SPECDIFF",
    "BeatSpectrum",
    "DetectionFunction",
    "DownBeat",
    "PeakPicking",
    "TempoTrack",
    "TempoTrackV2",
    "beats",
]
