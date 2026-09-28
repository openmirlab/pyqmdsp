# pyqmdsp

**Direct Python and NumPy access to qm-dsp's music analysis and signal processing algorithms.**

[![Tests and Linux wheels](https://github.com/openmirlab/pyqmdsp/actions/workflows/wheels.yml/badge.svg)](https://github.com/openmirlab/pyqmdsp/actions/workflows/wheels.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![GPL-2.0-or-later](https://img.shields.io/badge/license-GPL--2.0--or--later-blue)](LICENSE)

## Why this exists

[QM-DSP](https://github.com/c4dm/qm-dsp) is a C++ library developed at the
Centre for Digital Music, Queen Mary, University of London. Its algorithms
cover beats, onsets, spectral features, key, tonal change, structural segmentation
and general DSP. It also supplies algorithms used by the QM Vamp Plugins.

pyqmdsp makes those algorithms callable directly from Python with NumPy arrays.
The pinned C/C++ sources compile into the extension, preserving the upstream
implementations and stateful processing. Installation and analysis require no
Vamp SDK, plugin host, plugin discovery, system qm-dsp library or intermediate
WAV files. NumPy is the only Python runtime dependency.

**This is an early 0.x API, available from GitHub source and Actions wheels.
There is no PyPI release.**

## Acknowledgments

The analysis and DSP algorithms belong to the upstream authors. OpenMIRLab
maintains the Python bindings, packaging and verification:

- **Queen Mary, University of London, Centre for Digital Music** —
  [QM-DSP](https://github.com/c4dm/qm-dsp), the underlying library.
- **Christian Landone and Matthew Davies** — credited in the
  [tempo-tracking implementation](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/dsp/tempotracking/TempoTrack.cpp).
- **Chris Cannam** — credited in the
  [pitch utilities](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/base/Pitch.cpp)
  and other DSP infrastructure.
- **Mark Levy** — credited in the
  [HMM implementation](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/hmm/hmm.c)
  and segmentation code.
- **Nicolas Chetry**, **Martin Gasser**, **Kurt Jacobson** and **Thomas Wilmering** —
  credited respectively in [MFCC](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/dsp/mfcc/MFCC.cpp),
  [tonal estimation](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/dsp/tonal/TonalEstimator.cpp),
  [beat spectrum](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/dsp/rhythm/BeatSpectrum.cpp)
  and [wavelet filters](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/dsp/wavelet/Wavelet.cpp).
- **Mark Borgerding** — [Kiss FFT](https://github.com/mborgerding/kissfft), bundled by qm-dsp.
  **Fionn Murtagh** — the PCA implementation; **Allen Miller, David J Taylor and
  others** — polynomial fitting, as recorded in the [upstream notices](https://github.com/c4dm/qm-dsp/blob/e34a3cc188332ed7c33cd9257ef164de5b587191/README.md#licence).
- **Wenzel Jakob and contributors** — [nanobind](https://github.com/wjakob/nanobind),
  the binding layer.
- **OpenMIRLab's [pytimestretch](https://github.com/openmirlab/pytimestretch)** —
  the precedent for direct native bindings, pinned engines and wheel packaging.

Additional authors and bundled numerical-library notices remain in the original
source files. [NOTICE](NOTICE) records the pinned revision and component provenance.
No model weights are involved.

## Citation

For research using these algorithms, credit [qm-dsp](https://github.com/c4dm/qm-dsp)
and the relevant original algorithm authors above. The pinned upstream repository
does not supply a library-wide BibTeX citation; pyqmdsp does not introduce a new
analysis algorithm. Record the qm-dsp revision
`e34a3cc188332ed7c33cd9257ef164de5b587191`, the pyqmdsp version and
`pyqmdsp.engine_info()` with reproducibility materials.

## Features

| Task | Public API |
|---|---|
| Whole-buffer beat times, median BPM and tempo between beats | `pyqmdsp.beats` |
| Five onset detection functions, peak picking, two tempo trackers, downbeats and beat spectrum | `pyqmdsp.rhythm` |
| Complex/real FFT, DCT, Constant-Q, chromagram, MFCC and phase-vocoder analysis | `pyqmdsp.spectral` |
| Key detection, tonal centroids, tonal change and wavelet decomposition filters | `pyqmdsp.spectral` |
| Pitch conversion, windows, filtering, framing, decimation and resampling | `pyqmdsp.utilities` |
| Correlation, distances, normalization, median filters and polynomial fitting | `pyqmdsp.utilities` |
| Clustering and structural segmentation from audio frames or supplied features | `pyqmdsp.utilities` |
| PCA, Gaussian densities, Gaussian HMM initialization/training/decoding and posterior inference | `pyqmdsp.statistics` |

## Scope

The package exposes all upstream **algorithm families**, including capabilities
whose musical accuracy is limited. [API coverage](docs/coverage.md) maps original
methods to Python and records exceptions. In particular, an undefined upstream
`TCSGram.normalize` method is unavailable, and Constant-Q uses the safe
real/imaginary-input overload because the legacy interleaved overload writes
outside its buffer.

Most APIs are low-level frame or feature processors. `beats()` supplies a
whole-buffer pipeline; the other modules expose upstream building blocks with
explicit parameters. Wavelet support supplies the filters implemented upstream,
not a complete wavelet transform. The phase vocoder exposes analysis outputs.
HMM training fits this classical statistical model; no neural models or weights
are used.

File decoding, channel mixing, frame/timestamp policy for custom pipelines,
playback and application-specific musical decisions belong to the caller.
Raw BLAS, C++ allocation and thread-management APIs remain implementation details.

## Install

### From source

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/),
and provide a C compiler and a C++17 compiler (GCC, Clang or MSVC). Python 3.10
or newer is required. The isolated build installs nanobind and scikit-build-core
and obtains CMake when needed.

```bash
git clone --recurse-submodules https://github.com/openmirlab/pyqmdsp.git
cd pyqmdsp
uv sync --python 3.12 --no-dev
```

For an existing checkout cloned without submodules, run
`git submodule update --init --recursive` first. The build compiles the pinned
sources under `extern/qm-dsp`, including KissFFT and the numerical routines used
by segmentation.

### Built wheels

Download and unzip the artifact from [GitHub Actions](https://github.com/openmirlab/pyqmdsp/actions),
then select the wheel matching your Python version and architecture. A wheel
includes the native engine, so installation needs no compiler. Outside the
source checkout:

```bash
uv venv --python 3.12
uv pip install "/path/to/the-matching-cp312-wheel.whl"
```

CI builds and tests **CPython 3.10–3.13** on:

| Platform | Architecture | Artifact |
|---|---|---|
| Linux | x86_64 | `wheels-linux` |
| macOS | arm64 and x86_64 | `wheels-macos` |
| Windows | AMD64 | `wheels-windows` |

macOS and Windows artifacts come from **Wheels (all platforms)**, triggered by
an `all-platforms` PR label or manual dispatch. Actions artifacts are development
builds with limited retention; downloading through GitHub requires sign-in.

## Quick start

This example needs only pyqmdsp and NumPy. Save it as `quick_start.py` beside the
installed `.venv`, then run `uv run --no-project python quick_start.py`.

```python
import numpy as np
import pyqmdsp

sr = 44_100
t = np.arange(12 * sr, dtype=np.float64) / sr
audio = 0.05 * np.sin(2 * np.pi * 220 * t)
audio[::sr // 2] += 1.0

result = pyqmdsp.beats(audio, sr)
print("Beat times (seconds):", result["beats_s"])
print("Median BPM:", result["bpm"])
print("Tempo between consecutive beats:", result["tempo_curve"])

mfcc = pyqmdsp.spectral.MFCC(sr, fft_size=1024, n_coefficients=13)
coefficients = mfcc.process_time(audio[:mfcc.fft_length])
print("MFCC vector:", coefficients)
print(pyqmdsp.engine_info())
```

MFCC's default `want_c0=True` includes the energy coefficient C0, so requesting
13 coefficients above returns a 14-value vector.

For a file, install `soundfile` separately with `uv pip install soundfile`:

```python
import soundfile as sf
import pyqmdsp

audio, sr = sf.read("input.wav", dtype="float64", always_2d=True)
mono = audio.mean(axis=1)  # Explicit caller-selected channel mix.
result = pyqmdsp.beats(mono, sr)
print(result["beats_s"])
```

### Beat results and short clips

| Field | Meaning |
|---|---|
| `beats_s` | One-dimensional float64 array of beat times in seconds |
| `bpm` | Median of `tempo_curve`; `0.0` when no beat interval is available |
| `tempo_curve` | BPM between consecutive beats; normally one value fewer than `beats_s` |

`beats()` uses uncentered frames, discards the first two detection-function
frames and retains the evaluated QM tracker's timestamp convention. Its period
estimator requires more than 640 detection-function frames after trimming:
roughly **7.45 seconds of usable context at 44.1 kHz**. Shorter inputs return empty
beat/tempo arrays and zero BPM. Trailing silence does not supply usable context.
Each call creates independent analysis state.

## Working with frames and state

Create one processor per stream and feed frames in order. Read the processor's
frame and hop sizes rather than assuming every algorithm uses the same sizes.
For example, this produces an onset detection function from a synthetic signal:

```python
import numpy as np
from pyqmdsp import rhythm, utilities

sr, size, hop = 44_100, 1024, 512
audio = np.zeros(2 * sr, dtype=np.float64)
audio[::sr // 4] = 1.0
detector = rhythm.DetectionFunction(hop, size, rhythm.DF_COMPLEXSD)
novelty = np.array([
    detector.process_time_domain(frame)
    for frame in utilities.frames(audio, size, hop)
])
print(novelty.shape)  # One value per analysis frame; not onset timestamps.
```

`KeyMode.process()` returns the upstream key index: `0` means no key,
`1–12` are C through B major, and `13–24` are C through B minor.
`key_strengths()` returns 24 profile scores after processing; these are not
calibrated probabilities. Frames advance by `KeyMode.hop_size` and have
`KeyMode.block_size` samples.

For an independent stream, create a new object unless its API offers reset.
`ClusterMeltSegmenter` is initialized once and uses either audio-frame extraction
or supplied features. Its segment labels follow upstream's 1-based convention.
HMMs offer `close()` and context-manager support; initialization and segmentation
retain upstream's time-seeded/global randomness.

## Processing contract

| Input or output | Contract |
|---|---|
| Audio/frame layout | Mono one-dimensional arrays; caller handles channel selection or mixing |
| Features | Two-dimensional arrays with rows as observations and columns as features |
| Numeric input | Converted to contiguous float64; invalid shapes and non-finite inputs are rejected before native computation |
| Precision | Most arithmetic uses double; APIs with native float parameters/buffers retain those conversions |
| Output ownership | New NumPy arrays or Python values; inputs are not modified |
| Sample rate | Explicit in APIs that require it; no implicit file loading, channel mixing or resampling |
| Units | `beats_s` is seconds; low-level trackers use detection-function frame indices; `TCSGram` times are milliseconds |
| History | Stateful processors preserve upstream history across calls; instances serialize mutable native operations |
| Parameters | Explicit upstream options; unsupported/unsafe combinations raise rather than run undefined native behavior |

Docstrings describe individual shapes and units. Streaming `Resampler` uses
`bandwidth=0.45 * min(source_rate, target_rate)` when omitted, retains latency,
and accepts consecutive sample blocks. The one-off `utilities.resample()` calls
upstream's latency-compensated helper. HMM `set_parameters()` provides repeatable
inference from a prescribed model.

## Fidelity and musical accuracy

Numerical compatibility is checked independently from musical quality:

- **Original qm-dsp comparison:** standalone C++ executables compile pristine
  upstream sources, generate nonzero inputs and expected outputs, and compare
  public Python results across all four modules. Tests cover state, non-default
  parameters and secondary outputs. Shapes and discrete labels are checked
  exactly; floating comparisons use documented tolerances. See the
  [verification method, coverage and measured report](verification/upstream/README.md).
- **Saved Vamp beat comparison:** on 72 annotated files, reference mean beat F
  was **0.86534**, versus **0.86485** for pyqmdsp. The difference, **0.00049**, is
  within the **0.005** acceptance gate. Five files differ individually; this is
  a score-level result, not sample-identical beat output. See the
  [per-file report](verification/beats/report.json).
- **Native safety and portability:** installed wheels are exercised on the
  platform matrix above; Linux also runs AddressSanitizer checks.

These are bounded tests at a pinned source revision. They do not prove identical
results for every signal, parameter combination, compiler or platform.
Randomized algorithms need the same random state for reproducible comparisons.
The reference harness controls that state in test processes only.

The bindings preserve qm-dsp's musical limitations: in the recorded comparison,
onset detection did not beat the alternatives and downbeat detection performed
poorly. Availability of an API is not a claim that it is the best estimator.

## Development and verification

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv build
```

`uv build` creates an sdist and then builds a wheel from it. After native changes,
rebuild with `uv sync --reinstall-package pyqmdsp`. The normal test suite includes
committed original-output fixtures. Randomized original-output tests require the
Linux glibc reference harness and are explicitly skipped in ordinary runs.

Run the complete original-versus-binding check on Linux with CMake and a compiler:

```bash
uv run python verification/upstream/verify.py
```

This independently builds qm-dsp, compares both committed and freshly generated
original outputs, and writes `verification/upstream/report.json`. It is also a CI
gate. Fixture regeneration commands and environment provenance are documented in
the [verification guide](verification/upstream/README.md).

The upstream submodule remains unmodified. Recorded build-directory patches free
omitted allocations and replace a legacy Fortran diagnostic-output dependency;
they preserve numerical calculations. The reference executables use the original
sources without those patches.

## License

**GPL-2.0-or-later**, with upstream and bundled component notices retained.
See [LICENSE](LICENSE), [NOTICE](NOTICE) and the license files included in source
and wheel distributions. KissFFT and nanobind retain their BSD-style notices;
PCA, polynomial fitting and bundled BLAS/LAPACK routines retain upstream provenance.

## Support

Report reproducible issues at [GitHub Issues](https://github.com/openmirlab/pyqmdsp/issues).
Include Python/OS/architecture, `pyqmdsp.engine_info()`, the API and parameters,
input shape/dtype and a small reproducer. Prefer synthetic input when the original
audio cannot be shared.
