# pyqmdsp maintainer context

This standalone audio-tool package binds qm-dsp directly through nanobind. Paul authorized its full DSP scope under OpenMIRLab following pytimestretch. Keep upstream extern/qm-dsp unchanged and pinned. Vamp is not a build or runtime dependency. Threading/allocation helpers are native infrastructure; all public numerical algorithms are in scope.

Native bindings live in native/, NumPy interfaces in src/pyqmdsp/. Read docs/coverage.md for coverage and README.md for the current user contract. Tests must execute the real algorithms and cover invalid inputs before they can reach unsafe native code. Keep README, coverage, NOTICE and CHANGELOG aligned with changes. Local docs/blueprints follows pytimestretch's ignored planning convention.

Verification: uv sync --group dev; uv run pytest -q; uv run ruff check .; uv build. After C++ edits: uv sync --reinstall-package pyqmdsp. Test the resulting wheel in an isolated environment. No PyPI publication is authorized.
