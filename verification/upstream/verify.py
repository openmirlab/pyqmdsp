"""Build pristine qm-dsp references and compare public Python outputs.

Reads: independent CMake project, oracle drivers, golden fixtures, pytest reports.
The recording path never imports pyqmdsp; the clock shim is test-process-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DOMAINS = ("rhythm", "spectral", "utilities", "statistics")
UPSTREAM = "e34a3cc188332ed7c33cd9257ef164de5b587191"


def run(args, **kwargs):
    subprocess.run([str(arg) for arg in args], check=True, cwd=ROOT, **kwargs)


def output(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="replace golden fixtures using original C/C++ only; does not import the binding")
    parser.add_argument("--output", type=Path, default=HERE / "report.json")
    args = parser.parse_args()
    if platform.system() != "Linux" or platform.libc_ver()[0] != "glibc":
        parser.error("the full time/RNG-controlled reference runner requires Linux glibc; ordinary pytest runs deterministic fixtures on other platforms")
    if output("git", "-C", str(ROOT / "extern/qm-dsp"), "rev-parse", "HEAD") != UPSTREAM:
        parser.error("unexpected upstream revision")
    if output("git", "-C", str(ROOT / "extern/qm-dsp"), "status", "--porcelain", "--untracked-files=no"):
        parser.error("upstream tracked sources must be pristine")

    build = ROOT / "build/upstream-reference"
    run(["cmake", "-S", HERE, "-B", build, "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", build, "--parallel", "4"])
    cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines() if "=" in line and not line.startswith(("//", "#")))
    compiler = next(value for key, value in cache.items() if key.split(":", 1)[0] == "CMAKE_CXX_COMPILER")
    env = os.environ.copy()
    env["LD_PRELOAD"] = str(build / "libfixed_time.so") + (":" + env["LD_PRELOAD"] if env.get("LD_PRELOAD") else "")
    fixture_dir = HERE / "fixtures" if args.record else build / "live-fixtures"
    fixture_dir.mkdir(parents=True, exist_ok=True)
    for domain in DOMAINS:
        run([build / f"oracle_{domain}", fixture_dir / f"{domain}.json"], env=env, timeout=120)
    provenance = {
        "upstream_revision": UPSTREAM,
        "upstream_sources": "pristine; no package patches or binding source linked",
        "reference_source_inventory": "qm-dsp/build/general/Makefile.inc",
        "compiler": output(compiler, "--version").splitlines()[0],
        "system": platform.system(), "machine": platform.machine(),
        "libc": list(platform.libc_ver()),
        "floating_point": "double KissFFT; Release optimization; no fast-math",
        "random_environment": {"time": 1780000000, "srand_seed": 20260928, "scope": "reference/test processes only"},
        "fixtures_sha256": {d: sha256(fixture_dir / f"{d}.json") for d in DOMAINS},
        "drivers_sha256": {d: sha256(HERE / f"oracle_{d}.cpp") for d in DOMAINS},
        "harness_sha256": {p: sha256(HERE / p) for p in ("CMakeLists.txt", "oracle.h", "fixed_time.c", "f2c_guard.c")},
        "binding_before_recording": output("git", "rev-parse", "HEAD"),
    }
    if args.record:
        (fixture_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print("Recorded pristine upstream fixtures; no Python binding was imported.")
        return

    reports = {}
    for label, directory in (("golden", HERE / "fixtures"), ("live", fixture_dir)):
        report_path = build / f"{label}-comparison.json"
        test_env = {**env, "PYQMDSP_REFERENCE_DIR": str(directory), "PYQMDSP_PARITY_REPORT": str(report_path)}
        result = subprocess.run([sys.executable, "-m", "pytest", str(ROOT / "tests/upstream"), "-q"], cwd=ROOT, env=test_env, check=False)
        reports[label] = json.loads(report_path.read_text()) if report_path.exists() else {"exit_status": result.returncode}
        incomplete = not reports[label].get("tests") or any(test["outcome"] != "passed" for test in reports[label].get("tests", []))
        if result.returncode or incomplete:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"reference": provenance, "comparisons": reports}, indent=2) + "\n")
            raise SystemExit(result.returncode or 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"reference": provenance, "comparisons": reports}, indent=2) + "\n")
    print(f"Original-versus-binding report: {args.output}")


if __name__ == "__main__":
    main()
