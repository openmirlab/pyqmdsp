"""Compare direct binding against saved reference on annotated audio.

Reads: package beats(), saved truth/reference and an external audio directory.
"""

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

import pyqmdsp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    here = Path(__file__).parent
    audio = {p.stem: p for p in args.audio_root.rglob("*.wav")}
    rows = []
    for path in sorted((here / "truth").glob("*.json")):
        truth = json.loads(path.read_text())
        reference = json.loads((here / "reference" / path.name).read_text())
        samples, rate = sf.read(audio[path.stem], dtype="float64")
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        actual = pyqmdsp.beats(samples, rate)

        def score(events, reference_beats=truth["beats"]):
            return float(
                mir_eval.beat.f_measure(
                    np.asarray(reference_beats),
                    np.asarray(events),
                    f_measure_threshold=0.07,
                )
            )

        expected_f, actual_f = score(reference["beats"]), score(actual["beats_s"])
        rows.append(
            {
                "file": path.stem,
                "reference_f": expected_f,
                "actual_f": actual_f,
                "delta": actual_f - expected_f,
                "beats_s": actual["beats_s"].tolist(),
                "bpm": actual["bpm"],
            }
        )
    reference_mean = float(np.mean([r["reference_f"] for r in rows]))
    actual_mean = float(np.mean([r["actual_f"] for r in rows]))
    report = {
        "engine": pyqmdsp.engine_info(),
        "files": len(rows),
        "reference_mean_f": reference_mean,
        "actual_mean_f": actual_mean,
        "passed": len(rows) == 72 and abs(actual_mean - reference_mean) <= 0.005,
        "results": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
    for row in rows:
        if abs(row["delta"]) > 1e-12:
            print(row["file"], row["delta"])
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
