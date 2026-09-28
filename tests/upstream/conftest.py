"""Original outputs and numerical comparisons for public binding tests.

Reads: committed golden JSON or freshly generated original executable outputs.
"""
import ctypes
import json
import os
import platform
import sys
from pathlib import Path

import numpy as np
import pytest

_measurements = []
_outcomes = []


@pytest.fixture
def reference():
    root = Path(os.environ.get("PYQMDSP_REFERENCE_DIR", Path(__file__).resolve().parents[2] / "verification/upstream/fixtures"))

    def read(domain):
        return json.loads((root / f"{domain}.json").read_text())

    return read


@pytest.fixture
def assert_matches(request):
    def check(actual, expected, *, rtol=1e-10, atol=1e-12, _path="output"):
        if isinstance(expected, dict):
            assert set(actual) == set(expected)
            for key in expected:
                check(actual[key], expected[key], rtol=rtol, atol=atol, _path=f"{_path}.{key}")
        elif isinstance(expected, (str, bool)):
            assert actual == expected
        else:
            want, got = np.asarray(expected), np.asarray(actual)
            assert got.shape == want.shape
            exact = got.dtype.kind in "biuUS" and want.dtype.kind in "biuUS"
            if exact:
                np.testing.assert_array_equal(got, want)
            else:
                np.testing.assert_allclose(got, want, rtol=rtol, atol=atol)
            error = np.abs(got.astype(float) - want.astype(float)) if got.dtype.kind not in "US" else np.zeros(got.shape)
            _measurements.append({
                "test": request.node.name, "field": _path, "values": int(got.size),
                "exact": bool(np.array_equal(got, want)),
                "comparison": "exact" if exact else "tolerance",
                "rtol": 0 if exact else rtol, "atol": 0 if exact else atol,
                "max_abs_error": float(error.max()) if error.size else 0,
            })
    return check


def pytest_runtest_logreport(report):
    if report.when == "call" or report.skipped or report.failed:
        _outcomes.append({"test": report.nodeid, "outcome": report.outcome})


def pytest_sessionfinish(session, exitstatus):
    if path := os.environ.get("PYQMDSP_PARITY_REPORT"):
        import pyqmdsp

        Path(path).write_text(json.dumps({
            "exit_status": int(exitstatus), "tests": _outcomes,
            "binding": {"version": pyqmdsp.__version__, "module": pyqmdsp.__file__, "engine": pyqmdsp.engine_info()},
            "python": sys.version, "numpy": np.__version__,
            "fields_compared": len(_measurements),
            "values_compared": sum(m["values"] for m in _measurements),
            "all_fields_exact": all(m["exact"] for m in _measurements),
            "max_abs_error": max((m["max_abs_error"] for m in _measurements), default=0),
            "measurements": _measurements,
        }, indent=2) + "\n")


@pytest.fixture
def seeded_native():
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        pytest.skip("recorded RNG fixtures require Linux glibc; deterministic algorithms run everywhere")
    libc = ctypes.CDLL(None)
    libc.time.argtypes = [ctypes.c_void_p]
    libc.time.restype = ctypes.c_long
    if libc.time(None) != 1780000000:
        pytest.skip("run verification/upstream/verify.py for clock-controlled HMM/segmentation parity")
    libc.srand.argtypes = [ctypes.c_uint]

    def seed(value=20260928):
        libc.srand(value)

    seed()
    return seed
