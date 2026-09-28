# Utilities binding coverage

Source of truth: qm-dsp public headers under `base/`, `dsp/rateconversion/`,
`dsp/signalconditioning/`, `dsp/segmentation/`, and `maths/` except PCA. Direct
HMM and PCA interfaces live in `pyqmdsp.statistics`; segmentation uses those
upstream algorithms internally.

## Base

| Upstream API | Python API | Coverage | Notes |
|---|---|---|---|
| `Pitch::getFrequencyForPitch` | `frequency_for_pitch` | Covered | Validates positive `concert_a`. |
| `Pitch::getPitchForFrequency` | `pitch_for_frequency` | Covered | Returns `(pitch, cents_offset)`. |
| `Window<double>` | `window_data`, `cut_window` | Covered | Exposes rectangular, Bartlett, Hamming, Hanning/Hann, Blackman, Blackman-Harris. |
| `KaiserWindow::Parameters` and constructors | `kaiser_parameters_*`, `kaiser_window*`, `cut_kaiser` | Covered | Parameter helpers and direct/custom window application are exposed. |
| `SincWindow` | `sinc_window`, `cut_sinc` | Covered | Validates positive length and `p`. |

## Rate Conversion

| Upstream API | Python API | Coverage | Notes |
|---|---|---|---|
| `Decimator` | `Decimator` | Covered | Stateful block processor with `process` and `reset_filter`. Validates power-of-two factor, supported range, and exact input length. |
| `DecimatorB` | `DecimatorB` | Covered | Stateful block processor for repeated factor-of-two decimation. |
| `Resampler` | `Resampler`, `resample` | Covered | Exposes streaming process, latency, custom SNR/bandwidth, and one-off latency-compensated resampling. |

## Signal Conditioning

| Upstream API | Python API | Coverage | Notes |
|---|---|---|---|
| `Filter` | `Filter` | Covered | Stateful FIR/IIR filtering with reset and order. Coefficients are validated before native construction. |
| `FiltFilt` | `filtfilt` | Covered | One-off zero-phase filtering. |
| `DFProcess` | `df_process` | Covered | One-off detection-function conditioning. Requires IIR coefficients because upstream config expects both `LPACoeffs` and `LPBCoeffs`. |
| `Framer` | `frames` | Covered | Returns a dense 2-D array of frames. Wrapper configures frame length and hop before calling `setSource`, matching upstream state requirements. |

## Maths

| Upstream API | Python API | Coverage | Notes |
|---|---|---|---|
| `Correlation::doAutoUnBiased` | `autocorrelation_unbiased` | Covered | Returns the same length as input. |
| `CosineDistance::distance` | `cosine_distance` | Covered | Validates equal non-empty vector sizes. |
| `KLDivergence::distanceGaussian` | `kl_gaussian` | Covered | Validates equal sizes and non-negative variances. |
| `KLDivergence::distanceDistribution` | `kl_distribution` | Covered | Supports symmetric and asymmetric variants. |
| `MathUtilities` scalar helpers | `scalar_math`, `integer_math` | Covered | Includes round, principal argument, floating modulus, powers of two, factorial, and gcd. |
| `MathUtilities` vector helpers | `math_summary`, `normalise`, `lp_norm`, `normalise_lp`, `adaptive_threshold`, `circ_shift`, `argmax` | Covered | Wraps finite 1-D NumPy arrays and returns arrays or dictionaries. |
| `MedianFilter` | `median_filter`, `MedianFilter` | Covered | Exposes static whole-array filtering and stateful push/get/reset/get-at-percentile. |
| `TPolyFit::PolyFit2` | `polyfit` | Covered | Calls upstream implementation through the build-generated `polyfit_decl.h`, which contains the exact upstream class declaration without the header's out-of-class function definitions. Returns coefficients and correlation. |
| `maths/pca/pca.h` | `statistics.pca` | Out of this packet | See [Statistics](coverage-statistics.md). |

## Segmentation

| Upstream API | Python API | Coverage | Notes |
|---|---|---|---|
| `mpeg7_constq` | `mpeg7_constq` | Covered | Copies input and returns the extra envelope column. |
| `cq2chroma` | `cq_to_chroma` | Covered | Validates `bins` divides coefficient count. |
| `create_histograms` | `create_histograms` | Covered | Validates labels, bin range, and odd histogram length. |
| `cluster_melt` | `cluster_melt` | Covered | Direct row-major histogram clustering. Labels preserve upstream's 1-based cluster numbering. |
| `cluster_segment` | `cluster_segment` | Covered | Uses upstream HMM internally; wrapper retains the GIL and takes a mutex because upstream HMM uses global random state. Labels preserve upstream's 1-based cluster numbering. |
| `constq_segment` | `constq_segment` | Covered | Exposes CONSTQ and CHROMA upstream paths. The CONSTQ path uses upstream PCA internally and receives an extra envelope column in native storage because upstream writes `features[i][ncoeff]`. Labels preserve upstream's 1-based cluster numbering. |
| `ClusterMeltSegmenterParams` | `ClusterMeltSegmenterParams` | Covered | All public parameter fields are represented. Upstream currently ignores `ncomponents` in `constq_segment` and hardcodes 20 PCA components. |
| `ClusterMeltSegmenter` | `ClusterMeltSegmenter` | Covered | Exposes initialise, get window/hop size, extract audio features, set feature matrices, segment default, segment with explicit type count, clear, get segmentation, and get segment type count. Instances are one-shot initialized and choose either audio extraction or supplied feature matrices for their lifetime, matching upstream's irreversible `setFeatures` mode change. |
| `Segmenter` abstract base | Through `ClusterMeltSegmenter` | Covered where instantiable | Abstract base is not separately constructible. Its public operations are available through the concrete segmenter. |

## Safety and limitations

- Python wrappers coerce numeric inputs to contiguous `float64` or `int64` arrays
  before native calls and reject non-finite floats.
- Native wrappers validate sizes, coefficient compatibility, power-of-two
  factors, positive rates, positive windows, histogram lengths, segment counts,
  neighbour limits, ConstantQ bin counts, and label ranges before calling
  upstream code.
- Stateful segmenters reject reads before initialization, segmentation reads
  before a successful segment call, repeated initialization, short audio
  buffers, and feature-source mixing.
- Compute-only, stateless routines release the GIL where they do not touch
  upstream global state. Segmentation routines keep the GIL and use a mutex
  because upstream HMM initialisation uses `srand(time(0))` and `rand()`.
- Direct `hmm/hmm.h` and `maths/pca/pca.h` APIs are excluded from this document
  because they are covered in [Statistics](coverage-statistics.md).
