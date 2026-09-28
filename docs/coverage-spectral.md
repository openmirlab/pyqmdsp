# Spectral and tonal binding coverage

This document maps the scoped spectral/tonal qm-dsp headers to the public
Python surface in `pyqmdsp.spectral`. All listed bindings call qm-dsp native
classes or static functions directly through nanobind. Python facades only
normalise NumPy dtype/layout and do not reimplement the DSP algorithms.

## Header coverage

| Upstream header | Native binding | Python API | Coverage notes |
|---|---|---|---|
| `dsp/transforms/FFT.h` `FFT` | `SpectralFFT.process` | `FFT.process`, `fft` | Complex forward and inverse transforms. `imag=None` maps to qm-dsp's real-input complex FFT path. |
| `dsp/transforms/FFT.h` `FFTReal` | `SpectralFFTReal.forward`, `forward_magnitude`, `inverse` | `RealFFT`, `real_fft` | Real forward, magnitude-only forward, and inverse from non-redundant bins. |
| `dsp/transforms/DCT.h` | `SpectralDCT.forward`, `inverse` | `DCT`, `dct`, `idct` | Type-II forward, type-II unitary forward, type-III inverse, and type-III unitary inverse. |
| `dsp/chromagram/ConstantQ.h` | `SpectralConstantQ` | `ConstantQ` | Constructor parameters, sparse kernel generation, FFT length, hop, Q, bin count, and complex FFT-frame processing. |
| `dsp/chromagram/Chromagram.h` | `SpectralChromagram` | `Chromagram` | Time-domain and frequency-domain chroma processing, frame/hop/bin metadata, normalisation mode selection, `unityNormalise`, and `kabs`. |
| `dsp/keydetection/GetKeyMode.h` | `SpectralKeyMode` | `KeyMode` | Stateful block processing, block/hop metadata, key index result, and post-process key strengths. |
| `dsp/mfcc/MFCC.h` | `SpectralMFCC` | `MFCC` | Time-domain and frequency-domain MFCC extraction, sample rate, FFT size, coefficient count, log power, C0, and window selection. |
| `dsp/phasevocoder/PhaseVocoder.h` | `SpectralPhaseVocoder` | `PhaseVocoder` | Time-domain and frequency-domain magnitude/phase/unwrapped phase analysis plus `reset`. |
| `dsp/tonal/TonalEstimator.h` | `SpectralTonalEstimator`, `spectral_tonal_transform`, `spectral_normalize_chroma`, `spectral_tonal_magnitude` | `TonalEstimator`, `tonal_transform`, `normalize_chroma`, `tonal_magnitude` | 12-bin chroma to six-dimensional tonal centroid transform, `ChromaVector::normalizeL1`, and `TCSVector::magnitude`. |
| `dsp/tonal/TCSgram.h` | `SpectralTCSGram` | `TCSGram` | Add, clear, reserve-at-construction, frame-duration setter, size, time, duration, vector access, and matrix snapshot for change detection. |
| `dsp/tonal/ChangeDetectionFunction.h` | `SpectralChangeDetection` | `ChangeDetection` | Smoothing-width configuration and processing from either a TCS matrix or a `TCSGram`. |
| `dsp/wavelet/Wavelet.h` | `spectral_wavelet_names`, `spectral_wavelet_filters` | `wavelet_names`, `wavelet_filters` | All upstream decomposition wavelet enum values by integer id or exact qm-dsp name. |

## Safety and state

- Python converts public array inputs to one-dimensional or fixed-column
  `float64` C-contiguous NumPy arrays and rejects non-finite values before
  native dispatch.
- Native bindings validate sizes and parameters before calling qm-dsp raw
  pointer APIs.
- Mutable stateful native objects use a per-instance mutex around qm-dsp state.
  Long-running mutable processing releases the GIL while holding that mutex.
- `KeyMode.key_strengths()` raises before the first `process()` call because
  upstream strength buffers are not meaningful until a key frame has been
  processed.
- `ChangeDetection.process_tcsgram()` snapshots the `TCSGram` before processing
  so the change detector does not read a sequence while Python mutates it.

## Implementation-internal exclusions

- `ConstantQ::process(const double *FFTData)` is not exposed as a separate raw
  interleaved-pointer API because the pinned upstream implementation clears
  `m_CQdata` with `row < 2 * m_uK` while writing both `m_CQdata[row]` and
  `m_CQdata[row + 1]`, which writes one element past the allocated
  `2 * m_uK` buffer at the final row. The public binding keeps the safe
  separate `process_frequency(real, imag)` path, which calls the other upstream
  overload and covers the Constant-Q transform capability.
- `TCSGram::printDebug` is diagnostic stdout output and is omitted from the
  numerical API.
- `TCSGram::normalize` is declared in the upstream header but has no definition
  in the pinned qm-dsp sources, so binding it would create an unresolved symbol.
- `TCSGram::setNumBins` is not exposed because the pinned upstream storage and
  processing path use fixed six-dimensional `TCSVector` values; changing this
  field does not alter vector shape or change-detection semantics in the exposed
  API.
- `ChromaVector::printDebug`, `TCSVector::printDebug`, and direct subclassing of
  qm-dsp `std::valarray` types are not exposed. Their numerical capabilities are
  covered by `normalize_chroma`, `tonal_magnitude`, `TonalEstimator`,
  `TCSGram`, matrix snapshots, and change detection.

## Test evidence

`tests/test_spectral.py` exercises each exposed algorithm family:

- Complex FFT and real FFT against `numpy.fft` plus inverse round trips.
- DCT-II against the explicit DCT definition plus unitary inverse identity.
- Constant-Q and chromagram nonzero sinusoid peak bins using the upstream
  chromagram pitch-bin rule, plus `unityNormalise` and `kabs`.
- MFCC exact zero-input coefficients.
- Phase vocoder time-domain outputs against the upstream full-cycle fixture and
  frequency-domain magnitudes against FFT magnitudes.
- Key detection on a nonzero sinusoid using the upstream tonic rule.
- Tonal centroid transform against the explicit basis, `normalizeL1`, TCS
  magnitude, constant-vector change detection against qm-dsp's Gaussian
  edge-padding rule, TCSGram storage/frame-duration setter, and Haar wavelet
  filters.
- Unsafe-input rejection for invalid sizes, non-finite arrays, bad matrix shape,
  negative smoothing width, and unknown wavelet names.
