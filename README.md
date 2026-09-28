# pyqmdsp

**Direct NumPy bindings for qm-dsp music analysis and digital signal processing.**

## Why this exists

[QM-DSP](https://github.com/c4dm/qm-dsp) provides C++ algorithms for beat and
onset analysis, spectral features, tonality, segmentation and signal processing.
pyqmdsp makes these algorithms available through Python and NumPy, including
stateful processing and upstream parameters.

The native sources compile directly into the Python extension. Installation and
processing need no Vamp SDK, plugin host, plugin discovery, command-line audio
process or intermediate WAV files. This is an early 0.x API. There is no PyPI release.

## Acknowledgments

- **Queen Mary, University of London, Centre for Digital Music**, and the individual
  authors credited in [qm-dsp](https://github.com/c4dm/qm-dsp)'s source files:
  the upstream DSP and music analysis algorithms.
- **Mark Borgerding**: [Kiss FFT](https://github.com/mborgerding/kissfft), bundled by qm-dsp.
- **Wenzel Jakob and contributors**: [nanobind](https://github.com/wjakob/nanobind),
  the C++/Python binding layer.
- **OpenMIRLab's [pytimestretch](https://github.com/openmirlab/pytimestretch)**:
  the direct-binding, pinned-source and wheel-packaging precedent.

Original source notices, references and component provenance are retained; see
[NOTICE](NOTICE). The beat convenience pipeline follows the evaluated QM Vamp
tracker's framing and defaults. Vamp is only the origin of saved reference results.

## Features

| Module | Functionality |
|---|---|
| `pyqmdsp.beats` | Whole-buffer beat times in seconds, median BPM and inter-beat tempo |
| `pyqmdsp.rhythm` | Five onset detection functions, peak picking, original and V2 tempo trackers, downbeats and beat spectrum |
| `pyqmdsp.spectral` | Complex/real FFT, DCT, Constant-Q, chromagram, MFCC, phase vocoder, key, tonal-space/change detection and wavelet decomposition filters |
| `pyqmdsp.utilities` | Pitch conversion, windows, decimation/resampling, filters, framing, statistical helpers, polynomial fitting, clustering and structural segmentation |
| `pyqmdsp.statistics` | PCA, Gaussian densities and covariance inverse, HMM training/decoding and posterior inference |

The [coverage inventory](docs/coverage.md) maps upstream APIs to Python. C/C++
thread/allocation machinery and raw third-party BLAS APIs stay internal. Wavelet
support provides the filters that upstream implements; it does not invent a
wavelet transform engine.

## Install

Source builds require Git, a C/C++17 compiler and Python 3.10 or newer:

```bash
git clone --recurse-submodules https://github.com/openmirlab/pyqmdsp.git
cd pyqmdsp
uv sync --python 3.12 --no-dev
```

For a checkout cloned without submodules, first run
`git submodule update --init --recursive`. The build uses scikit-build-core,
CMake and nanobind. It compiles pinned sources including KissFFT and the numerical
routines used by segmentation; a system qm-dsp installation is not consulted.

Wheel workflows target CPython 3.10–3.13: Linux x86_64 by default, macOS arm64/x86_64
and Windows AMD64 via the `all-platforms` PR label or manual dispatch. Download
matching wheels from [Actions](https://github.com/openmirlab/pyqmdsp/actions), then:

```bash
uv venv --python 3.12
uv pip install /path/to/matching-wheel.whl
```

A matching wheel includes the compiled engine and needs no C++ compiler.
Actions artifacts are development builds rather than a PyPI release.

## Quick start

```python
import numpy as np
import pyqmdsp

sr = 44_100
audio = np.zeros(12 * sr, dtype=np.float64)
audio[::sr // 2] = 1.0
result = pyqmdsp.beats(audio, sr)
print(result["beats_s"])
print(result["bpm"])
print(pyqmdsp.engine_info())

# Work with the underlying algorithms and their state directly.
tracker = pyqmdsp.rhythm.DetectionFunction(step_size=512, frame_length=1024)
value = tracker.process_time_domain(audio[:1024])

fft = pyqmdsp.spectral.RealFFT(1024)
real, imag = fft.forward(audio[:1024])
restored = fft.inverse(real[:fft.bins], imag[:fft.bins])
np.testing.assert_allclose(restored, audio[:1024], atol=1e-12)
```

Run scripts in the installed environment with `uv run --no-project python script.py`.
File reading and resampling decisions belong to the caller; `soundfile` can read
files, while `pyqmdsp.utilities.resample` exposes qm-dsp's resampler.

## Contracts and limitations

Domain docstrings specify frame lengths, shapes and units. Inputs are copied or
normalized to contiguous float64 arrays before native computation; caller arrays
are not modified. Returned NumPy arrays own their buffers. Stateful processors
preserve upstream history across calls. Create a new instance to start an
independent stream unless its API provides reset. Independent one-shot beat calls
release the GIL during computation.

`beats()` returns `beats_s` (seconds), `bpm` (median inter-beat BPM, zero when no
interval is available), and `tempo_curve` (BPM per consecutive beat pair, normally
one shorter than the beat array). It preserves uncentered framing, removes the
first two detection-function frames, and uses the original tracker's timestamp
conversion. The period estimator needs more than 640 trimmed detection-function frames
(roughly 7.45 seconds of context at 44.1 kHz). Shorter clips return empty beat
and tempo arrays with BPM zero. Trailing silence does not supply usable context.

Algorithm fidelity is distinct from musical quality. In the original 72-file
comparison, QM beat tracking performed well; onset detection did not beat the
alternatives, and downbeats performed poorly on the evaluated steady 4/4 material.
These capabilities are nevertheless available. HMM initialization and segmentation
use upstream's time-seeded random generator. Numerical behavior can vary slightly
with compiler/platform. See [verification](verification/beats/README.md).

## Development

```bash
uv sync --group dev
uv run pytest -q
uv run ruff check .
uv build
```

`uv build` builds an sdist and then a wheel from that sdist. Tests execute the
native algorithms and numerical contracts. Linux CI also checks the installed
wheel with AddressSanitizer for native out-of-bounds accesses. Wheel CI tests installed distributions.
After native edits, rebuild with `uv sync --reinstall-package pyqmdsp`.
The upstream submodule stays unmodified; recorded patches are
applied to build-directory copies for allocation cleanup and portable diagnostic
output. They preserve numerical calculations.

## License

GPL-2.0-or-later, with upstream and bundled component notices retained. See
[LICENSE](LICENSE) and [NOTICE](NOTICE). No model weights are used.

## Support

Report reproducible issues at [GitHub Issues](https://github.com/openmirlab/pyqmdsp/issues),
including the input shape, parameters and `pyqmdsp.engine_info()`.
