# Changelog

## 0.1.0 — unreleased

- Direct nanobind interfaces to the full qm-dsp DSP/numerical algorithm families.
- NumPy-facing rhythm, spectral, utilities and statistics modules, with beats()
  preserving the evaluated QM tracker framing and timestamps.
- Pinned upstream source, bundled native numerical dependencies, source and wheel
  builds, build provenance, and executable numerical/packaging checks.
- Allocation-only HMM patch frees two buffers omitted by upstream.
- No Vamp dependency. GitHub source and wheel CI; no PyPI publication.
