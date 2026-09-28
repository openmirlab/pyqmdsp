# Recorded beats reference

These 72 annotated cases and QM Vamp output JSON files were recorded for the
Tactus comparison (qmdsp-beats commit 69afd4d, 2026-09-28). No audio is redistributed.
The files preserve the historical reference; normal package tests do not need Vamp.

Run `uv run --with soundfile --with mir_eval python verification/beats/check.py AUDIO_ROOT --output report.json`.
AUDIO_ROOT may contain nested directories; WAV basenames must match truth stems.
The acceptance gate is absolute difference of mean mir_eval beat F (70 ms) <=0.005.
Individual file differences are reported, including the float-sensitive decode cases.

Ballroom sources are documented in SOURCES.md. Synthetic reconstruction scripts
remain in Tactus's qmdsp-beats branch, experiments/beat-eval/material/scripts/.
