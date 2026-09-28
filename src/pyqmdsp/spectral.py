"""Spectral and tonal qm-dsp bindings.

All array inputs are converted to one-dimensional, C-contiguous ``float64``
NumPy arrays before native dispatch. Frequencies are in Hz, sample rates are in
samples per second, frame durations are in milliseconds, and phase values are in
radians.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import _native

FloatArray = NDArray[np.float64]


def _array1d(values: ArrayLike, *, name: str) -> FloatArray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def _matrix(values: ArrayLike, *, name: str, columns: int) -> FloatArray:
    array = np.ascontiguousarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != columns:
        raise ValueError(f"{name} must have shape (frames, {columns})")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


class FFT:
    """Complex FFT of a fixed frame size.

    Parameters
    ----------
    size:
        Number of complex samples. The upstream qm-dsp FFT accepts non-power of
        two sizes.

    Methods accept ``real`` and optional ``imag`` arrays with shape ``(size,)``.
    ``process(..., inverse=True)`` returns the inverse transform scaled by
    ``1 / size`` following qm-dsp.
    """

    def __init__(self, size: int):
        self._native = _native.SpectralFFT(size)

    @property
    def size(self) -> int:
        return self._native.size

    def process(
        self, real: ArrayLike, imag: ArrayLike | None = None, *, inverse: bool = False
    ) -> tuple[FloatArray, FloatArray]:
        real_array = _array1d(real, name="real")
        imag_array = None if imag is None else _array1d(imag, name="imag")
        return self._native.process(real_array, imag_array, inverse)


class RealFFT:
    """Real-input FFT of a fixed even frame size.

    ``forward`` returns full-length real and imaginary arrays with shape
    ``(size,)``. ``inverse`` accepts only the non-redundant bins with shape
    ``(size // 2 + 1,)`` and returns a real frame with shape ``(size,)``.
    """

    def __init__(self, size: int):
        self._native = _native.SpectralFFTReal(size)

    @property
    def size(self) -> int:
        return self._native.size

    @property
    def bins(self) -> int:
        return self._native.bins

    def forward(self, samples: ArrayLike) -> tuple[FloatArray, FloatArray]:
        return self._native.forward(_array1d(samples, name="samples"))

    def forward_magnitude(self, samples: ArrayLike) -> FloatArray:
        return self._native.forward_magnitude(_array1d(samples, name="samples"))

    def inverse(self, real: ArrayLike, imag: ArrayLike) -> FloatArray:
        return self._native.inverse(_array1d(real, name="real"), _array1d(imag, name="imag"))


class DCT:
    """Type-II and type-III DCT of a fixed size.

    Arrays have shape ``(size,)``. ``unitary=True`` selects qm-dsp's orthogonal
    scaling, where ``inverse(forward(x, unitary=True), unitary=True)`` recovers
    ``x`` within floating-point tolerance.
    """

    def __init__(self, size: int):
        self._native = _native.SpectralDCT(size)

    @property
    def size(self) -> int:
        return self._native.size

    def forward(self, samples: ArrayLike, *, unitary: bool = False) -> FloatArray:
        return self._native.forward(_array1d(samples, name="samples"), unitary)

    def inverse(self, coefficients: ArrayLike, *, unitary: bool = False) -> FloatArray:
        return self._native.inverse(_array1d(coefficients, name="coefficients"), unitary)


class ConstantQ:
    """Constant-Q transform operating on FFT frames.

    ``process_frequency`` accepts full-length FFT real and imaginary arrays with
    shape ``(fft_length,)``. It returns two arrays with shape ``(bins,)`` for
    Constant-Q real and imaginary bins.
    """

    def __init__(
        self,
        sample_rate: float,
        min_frequency: float,
        max_frequency: float,
        *,
        bins_per_octave: int = 12,
        threshold: float = 0.0054,
    ):
        self._native = _native.SpectralConstantQ(
            sample_rate, min_frequency, max_frequency, bins_per_octave, threshold
        )

    @property
    def bins(self) -> int:
        return self._native.bins

    @property
    def fft_length(self) -> int:
        return self._native.fft_length

    @property
    def hop(self) -> int:
        return self._native.hop

    @property
    def q(self) -> float:
        return self._native.q

    def process_frequency(self, real: ArrayLike, imag: ArrayLike) -> tuple[FloatArray, FloatArray]:
        return self._native.process_frequency(
            _array1d(real, name="real"), _array1d(imag, name="imag")
        )


class Chromagram:
    """Constant-Q chromagram.

    Time-domain frames have shape ``(frame_size,)``. Frequency-domain inputs are
    full-length, pre-fftshifted FFT real and imaginary arrays with shape
    ``(frame_size,)``. Output chroma vectors have shape ``(bins_per_octave,)``.
    """

    def __init__(
        self,
        sample_rate: float,
        min_frequency: float,
        max_frequency: float,
        *,
        bins_per_octave: int = 12,
        threshold: float = 0.0054,
        normalise: str = "unit_max",
    ):
        self._native = _native.SpectralChromagram(
            sample_rate, min_frequency, max_frequency, bins_per_octave, threshold, normalise
        )

    @property
    def bins(self) -> int:
        return self._native.bins

    @property
    def cq_bins(self) -> int:
        return self._native.cq_bins

    @property
    def frame_size(self) -> int:
        return self._native.frame_size

    @property
    def hop_size(self) -> int:
        return self._native.hop_size

    def process_time(self, frame: ArrayLike) -> FloatArray:
        return self._native.process_time(_array1d(frame, name="frame"))

    def process_frequency(self, real: ArrayLike, imag: ArrayLike) -> FloatArray:
        return self._native.process_frequency(
            _array1d(real, name="real"), _array1d(imag, name="imag")
        )

    def unity_normalise(self, values: ArrayLike) -> FloatArray:
        """Return qm-dsp unity-normalised chroma values.

        ``values`` must have shape ``(bins_per_octave,)``. qm-dsp divides by the
        maximum value, so an all-zero input is rejected before native dispatch.
        """

        return self._native.unity_normalise(_array1d(values, name="values"))

    def kabs(self, real: float, imag: float) -> float:
        """Return qm-dsp's complex magnitude helper for ``real + imag*j``."""

        return self._native.kabs(real, imag)


class MFCC:
    """Mel-frequency cepstral coefficients.

    Time-domain frames have shape ``(fft_length,)``. Frequency-domain inputs use
    non-redundant FFT bins with shape ``(fft_length // 2 + 1,)``. The output has
    shape ``(n_coefficients + int(want_c0),)``.
    """

    def __init__(
        self,
        sample_rate: int,
        *,
        fft_size: int = 2048,
        n_coefficients: int = 19,
        log_power: float = 1.0,
        want_c0: bool = True,
        window: str = "hamming",
    ):
        self._native = _native.SpectralMFCC(
            sample_rate, fft_size, n_coefficients, log_power, want_c0, window
        )

    @property
    def fft_length(self) -> int:
        return self._native.fft_length

    @property
    def output_size(self) -> int:
        return self._native.output_size

    def process_time(self, frame: ArrayLike) -> FloatArray:
        return self._native.process_time(_array1d(frame, name="frame"))

    def process_frequency(self, real: ArrayLike, imag: ArrayLike) -> FloatArray:
        return self._native.process_frequency(
            _array1d(real, name="real"), _array1d(imag, name="imag")
        )


class PhaseVocoder:
    """Phase vocoder magnitude, phase, and unwrapped phase analysis.

    Time-domain frames have shape ``(size,)``. Frequency-domain inputs use
    non-redundant bins with shape ``(size // 2 + 1,)``. Each result array has
    shape ``(size // 2 + 1,)``.
    """

    def __init__(self, size: int, hop: int):
        self._native = _native.SpectralPhaseVocoder(size, hop)

    @property
    def size(self) -> int:
        return self._native.size

    @property
    def hop(self) -> int:
        return self._native.hop

    @property
    def bins(self) -> int:
        return self._native.bins

    def process_time(self, frame: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        return self._native.process_time(_array1d(frame, name="frame"))

    def process_frequency(
        self, real: ArrayLike, imag: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        return self._native.process_frequency(
            _array1d(real, name="real"), _array1d(imag, name="imag")
        )

    def reset(self) -> None:
        self._native.reset()


class KeyMode:
    """Stateful key detector.

    Process successive time-domain frames with shape ``(block_size,)``. Frames
    should advance by ``hop_size`` samples. Key indices follow qm-dsp: ``0`` is
    no key, ``1`` is C major, and ``13`` is C minor.
    """

    def __init__(
        self,
        sample_rate: float,
        *,
        tuning_frequency: float = 440.0,
        hpcp_average: float = 10.0,
        median_average: float = 10.0,
        frame_overlap_factor: int = 1,
        decimation_factor: int = 8,
    ):
        self._native = _native.SpectralKeyMode(
            sample_rate,
            tuning_frequency,
            hpcp_average,
            median_average,
            frame_overlap_factor,
            decimation_factor,
        )

    @property
    def block_size(self) -> int:
        return self._native.block_size

    @property
    def hop_size(self) -> int:
        return self._native.hop_size

    def process(self, frame: ArrayLike) -> int:
        return self._native.process(_array1d(frame, name="frame"))

    def key_strengths(self) -> FloatArray:
        return self._native.key_strengths()


class TonalEstimator:
    """Map a 12-bin chroma vector into six-dimensional tonal centroid space."""

    def __init__(self):
        self._native = _native.SpectralTonalEstimator()

    def transform_chroma(self, chroma: ArrayLike) -> FloatArray:
        """Return a TCS vector with shape ``(6,)`` for a chroma input ``(12,)``."""

        return self._native.transform_chroma(_array1d(chroma, name="chroma"))


class TCSGram:
    """Mutable tonal centroid sequence used by change detection.

    Each added vector has shape ``(6,)``. ``frame_duration_ms`` controls the
    millisecond timestamps reported by qm-dsp's TCSGram. qm-dsp also has a
    ``setNumBins`` setter, but the stored `TCSVector` type is fixed at six
    dimensions in the pinned upstream code, so this facade keeps the dimension
    fixed and exposes only the meaningful frame-duration setter.
    """

    def __init__(self, frame_duration_ms: float, *, reserve: int = 0):
        self._native = _native.SpectralTCSGram(frame_duration_ms, reserve)

    @property
    def frame_duration_ms(self) -> float:
        return self._native.frame_duration_ms

    @property
    def size(self) -> int:
        return self._native.size

    @property
    def duration(self) -> int:
        return self._native.duration

    def add(self, tcs: ArrayLike) -> None:
        self._native.add(_array1d(tcs, name="tcs"))

    def vector_at(self, index: int) -> FloatArray:
        return self._native.vector_at(index)

    def matrix(self) -> FloatArray:
        return self._native.matrix()

    def time_at(self, index: int) -> int:
        return self._native.time_at(index)

    def set_frame_duration_ms(self, frame_duration_ms: float) -> None:
        """Set the duration used for timestamps of subsequently added vectors."""

        self._native.set_frame_duration_ms(frame_duration_ms)

    def clear(self) -> None:
        self._native.clear()


class ChangeDetection:
    """Tonal change detection over TCS vectors.

    ``process_matrix`` accepts an array with shape ``(frames, 6)`` and returns a
    distance curve with shape ``(frames,)``.
    """

    def __init__(self, smoothing_width: int):
        self._native = _native.SpectralChangeDetection(smoothing_width)

    def process_matrix(self, matrix: ArrayLike, *, frame_duration_ms: float = 10.0) -> FloatArray:
        return self._native.process_matrix(
            _matrix(matrix, name="matrix", columns=6), frame_duration_ms
        )

    def process_tcsgram(self, gram: TCSGram) -> FloatArray:
        return self._native.process_tcsgram(gram._native)


def fft(real: ArrayLike, imag: ArrayLike | None = None, *, inverse: bool = False) -> tuple[FloatArray, FloatArray]:
    """Run a one-shot complex FFT.

    ``real`` and optional ``imag`` must have identical shape ``(n,)``. The
    returned arrays also have shape ``(n,)``.
    """

    real_array = _array1d(real, name="real")
    return FFT(real_array.size).process(real_array, imag, inverse=inverse)


def real_fft(samples: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Run a one-shot real FFT for an even-length input array."""

    samples_array = _array1d(samples, name="samples")
    return RealFFT(samples_array.size).forward(samples_array)


def dct(samples: ArrayLike, *, unitary: bool = False) -> FloatArray:
    """Run a one-shot type-II DCT over a one-dimensional array."""

    samples_array = _array1d(samples, name="samples")
    return DCT(samples_array.size).forward(samples_array, unitary=unitary)


def idct(coefficients: ArrayLike, *, unitary: bool = False) -> FloatArray:
    """Run a one-shot type-III inverse DCT over a one-dimensional array."""

    coefficients_array = _array1d(coefficients, name="coefficients")
    return DCT(coefficients_array.size).inverse(coefficients_array, unitary=unitary)


def tonal_transform(chroma: ArrayLike) -> FloatArray:
    """Return a six-dimensional tonal centroid vector for a 12-bin chroma vector."""

    return _native.spectral_tonal_transform(_array1d(chroma, name="chroma"))


def normalize_chroma(chroma: ArrayLike) -> FloatArray:
    """Return qm-dsp ``ChromaVector.normalizeL1`` for a 12-bin chroma vector."""

    return _native.spectral_normalize_chroma(_array1d(chroma, name="chroma"))


def tonal_magnitude(tcs: ArrayLike) -> float:
    """Return qm-dsp ``TCSVector.magnitude`` for a six-dimensional TCS vector."""

    return _native.spectral_tonal_magnitude(_array1d(tcs, name="tcs"))


def wavelet_names() -> list[str]:
    """Return qm-dsp's supported decomposition wavelet names."""

    return list(_native.spectral_wavelet_names())


def wavelet_filters(wavelet: int | str) -> tuple[FloatArray, FloatArray]:
    """Return low-pass and high-pass decomposition filters for ``wavelet``.

    ``wavelet`` may be a numeric qm-dsp wavelet enum value or an exact name from
    :func:`wavelet_names`.
    """

    return _native.spectral_wavelet_filters(wavelet)


__all__ = [
    "DCT",
    "FFT",
    "MFCC",
    "ChangeDetection",
    "Chromagram",
    "ConstantQ",
    "FloatArray",
    "KeyMode",
    "PhaseVocoder",
    "RealFFT",
    "TCSGram",
    "TonalEstimator",
    "dct",
    "fft",
    "idct",
    "normalize_chroma",
    "real_fft",
    "tonal_magnitude",
    "tonal_transform",
    "wavelet_filters",
    "wavelet_names",
]
