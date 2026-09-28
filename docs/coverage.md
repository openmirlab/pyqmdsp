# qm-dsp API coverage

The target is the public DSP and numerical functionality of qm-dsp at
`e34a3cc188332ed7c33cd9257ef164de5b587191`. This inventory separates callable
algorithms from C/C++ support machinery. A Python binding preserves algorithm
behavior; it does not assert musical accuracy on arbitrary recordings.

| Domain | Python module | Detailed upstream mapping |
|---|---|---|
| Onset functions and peak picking; both tempo trackers; downbeats; beat spectrum | `pyqmdsp.rhythm` | [Rhythm](coverage-rhythm.md) |
| FFT, DCT, CQT, chromagram, MFCC, phase vocoder, key, tonal space/change, wavelet filters | `pyqmdsp.spectral` | [Spectral](coverage-spectral.md) |
| Pitch, windows, filters, framing, resampling, decimation, maths, polynomial fit, clustering and segmentation | `pyqmdsp.utilities` | [Utilities](coverage-utilities.md) |
| PCA, Gaussian HMM initialization/training/decoding/update, posterior inference and density | `pyqmdsp.statistics` | [Statistics](coverage-statistics.md) |

## Infrastructure boundary

- `base/Restrict.h`, `maths/MathAliases.h`, `maths/nan-inf.h`: compiler macros and
  C++ container aliases; NumPy arrays and Python scalars carry their public data.
- `dsp/segmentation/segment.h`: feature selector constants used by segmentation.
- `thread/{Thread,AsynchronousTask,BlockAllocator}.h`: execution/allocation machinery
  used internally by native algorithms; Python uses its own concurrency and lifetime.
- `ext/` and `include/`: bundled FFT/BLAS/LAPACK/f2c implementation dependencies,
  compiled into the extension. Their qm-dsp-facing algorithms are exposed above.
- Constructors/configuration structs become Python constructor parameters. C++
  buffer pointers and output parameters become owned NumPy results. Destructors
  run with Python lifetime; explicit close is available for HMM models.

The package never loads a Vamp plugin. Recorded Vamp beat outputs are historical
verification data only. Audio fixtures and evaluation dependencies do not ship in
runtime dependencies.
