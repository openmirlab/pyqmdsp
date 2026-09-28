# pyqmdsp maintainer context

This standalone audio-tool package binds qm-dsp directly through nanobind. Paul authorized its full DSP scope under OpenMIRLab following pytimestretch. Keep upstream extern/qm-dsp unchanged and pinned. Vamp is not a build or runtime dependency. Threading/allocation helpers are native infrastructure; all public numerical algorithms are in scope.

QMUL/Centre for Digital Music and the credited original authors own the DSP work:
Landone/Davies, Cannam, Levy, Chetry, Gasser, Jacobson, Wilmering and bundled
numerical-library authors. This project supplies bindings, packaging and
verification. Preserve individual source notices and verified README credits.
Load-bearing source files use a purpose header and Reads dependencies.

Native bindings live in native/, NumPy interfaces in src/pyqmdsp/. Read docs/coverage.md for coverage and README.md for the current user contract. Tests must execute the real algorithms and cover invalid inputs before they can reach unsafe native code. Keep README, coverage, NOTICE and CHANGELOG aligned with changes. Local docs/blueprints follows pytimestretch's ignored planning convention.

Verification: uv sync --group dev; uv run pytest -q; uv run ruff check .; uv build. After C++ edits: uv sync --reinstall-package pyqmdsp. Test the resulting wheel in an isolated environment. No PyPI publication is authorized.

For numerical fidelity, run `uv run python verification/upstream/verify.py` on
Linux glibc. It independently builds pristine original sources and compares both
golden and live outputs through the public Python APIs. All six comparisons must
pass with no skips, including controlled-RNG HMM/segmentation cases. Default
pytest runs deterministic golden cases everywhere and explicitly skips the two
RNG cases without the test-only clock shim. Read verification/upstream/README.md
before changing fixtures or claiming equivalence; `--record` generates originals
without importing the binding. Every numerical output is compared, with shapes
and discrete values exact and documented floating tolerances. Mathematical
identities, import success and beat F-score alone cannot establish full fidelity.

## Current implementation

All upstream algorithm families have NumPy interfaces in rhythm, spectral,
utilities and statistics. The exact coverage and upstream-defect exceptions live
in docs/coverage.md. beats() uses the evaluated framing and timestamp convention;
its period estimator requires more than 640 trimmed onset frames. Keep the
72-file mean beat-F gate within 0.005 of the recorded Vamp reference, and report
individual differences. Vamp remains absent from the installed package.

Build-directory patches free omitted HMM/MFCC allocations and replace a legacy
Fortran warning-output path with stderr. They do not change numerical algorithms.
Linux AddressSanitizer CI guards native memory access. HMM/segmentation preserve
upstream time-seeded randomness; mutable bindings serialize native state.

Use a clean installed wheel for final acceptance. CI covers Linux x86_64 and
opt-in macOS arm64/x86_64 and Windows AMD64, CPython 3.10–3.13. PyPI remains
separately gated with the Private :: Do Not Upload classifier.
