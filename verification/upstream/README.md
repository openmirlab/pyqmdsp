# Original qm-dsp versus pyqmdsp

This harness checks numerical binding fidelity independently of the existing
mathematical/contract tests and the saved Vamp beat-score comparison. It compares
public Python calls with standalone executables built from pristine qm-dsp
`e34a3cc188332ed7c33cd9257ef164de5b587191`.

## What is compared

| Domain | Original driver | Comparison inventory |
|---|---|---|
| Rhythm/onsets/beats | [oracle_rhythm.cpp](oracle_rhythm.cpp) | [Rhythm coverage](rhythm-coverage.md) |
| Spectral/tonal analysis | [oracle_spectral.cpp](oracle_spectral.cpp) | [Spectral coverage](spectral-coverage.md) |
| DSP/maths/segmentation | [oracle_utilities.cpp](oracle_utilities.cpp) | [Utilities coverage](utilities-coverage.md) |
| PCA/Gaussian/HMM | [oracle_statistics.cpp](oracle_statistics.cpp) | [Statistics coverage](statistics-coverage.md) |

Original inputs and outputs are committed in `fixtures/*.json`. Python replays
the input doubles directly, avoiding differences from generating the same signal
with a different trigonometric implementation. Cases include nonzero signals,
non-default parameters, repeated frames, resets, intermediate outputs, full model
parameters and discrete labels. Convenience APIs are checked against the same
original operations. Inputs copied unchanged are not counted as output evidence.

The original drivers include qm-dsp headers and call its APIs. Their independent
CMake project compiles the original source list from upstream's
`build/general/Makefile.inc`. It links no nanobind code, Python facades, package
native adapters or package patches. The original f2c warning-I/O references have
link-only guards that abort if reached; they are not numerical replacements.

## Run the full comparison

On Linux with glibc, a C/C++ compiler and CMake:

```bash
uv sync --group dev
uv run python verification/upstream/verify.py
```

The command builds the independent executables in `build/upstream-reference`,
generates fresh original outputs, then runs the public API comparisons twice:
once against the committed golden fixtures and once against the fresh outputs.
Either a failure or a skipped test fails the full check. The report is written to
[`report.json`](report.json); CI uploads its own report as `upstream-parity-report`.

To test an installed wheel, use that environment's Python to run this script by
absolute path. The script executes that same Python's pytest and imports its
installed pyqmdsp. It does not install or rebuild the Python package.

### Tolerances and environment

- Shapes and discrete integer outputs must match exactly.
- Floating comparisons use `rtol=1e-10`, `atol=1e-12`. The report also records
  exact numeric equality and maximum absolute error for every compared field.
- JSON uses 17 significant decimal digits. Exact numeric equality here is not a
  bit-pattern claim about signed zero or NaN payloads.
- [Fixture provenance](fixtures/provenance.json) records upstream revision,
  compiler, architecture, libc, driver/harness hashes and fixture hashes. The
  runtime report records Python, NumPy and the installed binding's engine info.
- Reference and package use double KissFFT with no fast-math. Cross-platform
  golden tests allow the stated floating tolerance; platform/compiler bit
  identity is not promised.

HMM initialization calls `srand(time(0))`; clustering also uses libc's global RNG.
For the full Linux check, a **test-only** shared library fixes `time()` at
`1780000000` in both reference and Python subprocesses. Explicit clustering seeds
use `20260928`; native HMM initialization still performs its original seeding.
The shim changes neither package sources nor installed/runtime behavior.
Randomized golden tests require Linux glibc and this fixed clock. Ordinary pytest
and other platforms explicitly skip those two tests; deterministic golden tests
still run. The dedicated Linux CI job requires all six comparison tests to pass.

## Recorded result

The committed report compares **357 numerical output fields, containing 18,479
values**, across all four domains. Both the saved-original and freshly-built-original
comparisons passed with exact numeric equality and **maximum absolute error 0**
on the recorded Linux/GCC environment. See the report's environment and
`comparisons.*.measurements` for the precise scope. A successful finite test suite
does not establish equality for every possible recording or parameter combination.

The 72-file Vamp result remains a separate [beat-score check](../beats/README.md):
mean F differs by 0.0004923023, and five files have individual score differences.
That result cannot be described as identical beat outputs.

## Regenerate original fixtures

```bash
uv run python verification/upstream/verify.py --record
uv run python verification/upstream/verify.py
```

`--record` runs only the pristine C/C++ originals and writes their fixtures and
provenance. It never imports pyqmdsp to calculate expected values. Inspect original
outputs before correcting a binding mismatch; retain the original reference as
the expected behavior. Review fixture changes with their driver changes.

## Boundaries

The [API inventory](../../docs/coverage.md) remains authoritative for exposed
methods. Undefined upstream methods, its unsafe interleaved CQT overload, debug
printing, raw allocation/thread APIs and invalid-input behavior are not numerical
parity targets. Memory cleanup and invalid-input rejection are covered separately.
These tests measure fidelity to upstream, not accuracy of musical interpretation.
