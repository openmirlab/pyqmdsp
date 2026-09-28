# Utilities upstream parity coverage

This parity fixture compares `pyqmdsp.utilities` against an executable linked to
the pristine, unpatched qm-dsp sources under `extern/qm-dsp`. The fixture stores
the exact original inputs under `inputs` and expected outputs under `outputs`.

## Deterministic coverage

- Pitch conversion: frequency for MIDI pitch and pitch plus cents for frequency.
- Windowing: rectangular, Bartlett, Hamming, Hanning, Blackman,
  Blackman-Harris, Kaiser parameter helpers, direct Kaiser window/cut,
  `byTransitionWidth`/`byBandwidth` alias window coefficients, Sinc window/cut.
- Rate conversion: `Decimator` first frame, second sequential frame, reset,
  `DecimatorB` first and second sequential frames, one-off `Resampler::resample`,
  streaming `Resampler` over more than 2048 nonzero samples in three chunks,
  latency, and custom downsampling SNR/bandwidth.
- Signal conditioning: stateful FIR history/reset, IIR filtering, `FiltFilt`,
  `DFProcess`, and `Framer`.
- Maths: unbiased autocorrelation, cosine distance, KL Gaussian/distribution,
  scalar/vector `MathUtilities`, normalization modes, Lp norm/normalization,
  adaptive threshold, circular shift, argmax, power/factorial/gcd helpers,
  static and stateful `MedianFilter`, and `TPolyFit::PolyFit2` coefficients and
  correlation.
- Segmentation helpers without randomness: `mpeg7_constq`, `cq2chroma`, and
  `create_histograms`.

## Stochastic coverage

The stochastic tests run when Linux/glibc has `fixed_time` preloaded, matching
the oracle command. Each original random case seeds `srand(20260928)` before the
direct random path; HMM-backed paths use `fixed_time` so upstream
`srand(time(0))` receives `1780000000`.

- `cluster_melt`
- `cluster_segment`
- `constq_segment` for high-contrast 60x24 CHROMA-path and full-rank 60x24
  CONSTQ-path inputs. The CONSTQ fixture produces at least two upstream labels;
  the CHROMA fixture still collapses to one upstream label despite four distinct
  pitch-class blocks, so the test records strict parity and reports that
  upstream behavior instead of forcing a split.
- `ClusterMeltSegmenter` with supplied feature matrices, default `segment()`,
  `clear()` followed by reuse, getters, and explicit segment count
- `ClusterMeltSegmenter` audio feature extraction path through MFCC
- `cluster_melt_segmenter` convenience factory

## Intentional API-shape differences

- Python `Resampler` defaults its streaming bandwidth to
  `0.45 * min(source_rate, target_rate)`. The oracle uses the same explicit
  bandwidth for streaming parity. The one-off `resample` fixture uses upstream's
  static `Resampler::resample` directly.
- Python segmentation returns dictionaries/lists; the oracle stores segmentation
  as `nsegtypes`, `samplerate`, `starts`, `ends`, and `types` so every field is
  compared exactly.
- Direct HMM and PCA public APIs are covered by `pyqmdsp.statistics`; utilities
  parity covers their use only through upstream segmentation helpers.

## Current gaps

No scoped utilities API is intentionally unpaired. Production safety guards that
reject upstream-unsafe inputs are covered by ordinary unit tests, not by this
pristine parity fixture, because the original executable would read out of bounds
or leave uninitialized fields for those cases.
