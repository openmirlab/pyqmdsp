"""Installed distribution integrity. Reads: importlib metadata, pyqmdsp."""

from importlib.metadata import files, metadata, requires, version
from pathlib import Path

import pyqmdsp


def test_version_and_build_provenance():
    assert pyqmdsp.__version__ == version("pyqmdsp")
    info = pyqmdsp.engine_info()
    assert info["source_revision"] == "e34a3cc188332ed7c33cd9257ef164de5b587191"
    assert info["fast_math"] is False
    assert info["fft"] == "kissfft-double"


def test_package_licenses_and_typing():
    paths = [str(p) for p in files("pyqmdsp")]
    assert (Path(pyqmdsp.__file__).parent / "py.typed").is_file()
    for notice in ("LICENSE", "NOTICE", "nanobind-LICENSE", "COPYING"):
        assert any(p.endswith("/" + notice) for p in paths)
    assert metadata("pyqmdsp")["License-Expression"] == "GPL-2.0-or-later"


def test_only_numpy_is_a_runtime_dependency():
    deps = requires("pyqmdsp") or []
    assert len(deps) == 1
    assert deps[0].startswith("numpy")
