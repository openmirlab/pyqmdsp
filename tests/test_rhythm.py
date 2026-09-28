from __future__ import annotations

import numpy as np
import pytest

from pyqmdsp import rhythm


def impulse_train(length: int, period: int, *, amplitude: float = 1.0) -> np.ndarray:
    values = np.zeros(length, dtype=np.float64)
    values[::period] = amplitude
    return values


def test_detection_function_processes_time_and_frequency_domains_statefully() -> None:
    detector = rhythm.DetectionFunction(4, 8, rhythm.DF_HFC)
    frame = np.ones(8, dtype=np.float64)

    with pytest.raises(RuntimeError, match="after processing one frame"):
        detector.spectrum_magnitude()

    first = detector.process_time_domain(frame)
    second = detector.process_time_domain(frame)
    spectrum = detector.spectrum_magnitude()
    freq_value = detector.process_frequency_domain(np.ones(5), np.zeros(5))

    assert first > 0.0
    assert second == pytest.approx(first)
    assert spectrum.shape == (5,)
    assert freq_value > 0.0
    assert detector.parameters["frame_length"] == 8


@pytest.mark.parametrize(
    "df_type",
    [
        rhythm.DF_HFC,
        rhythm.DF_SPECDIFF,
        rhythm.DF_PHASEDEV,
        rhythm.DF_COMPLEXSD,
        rhythm.DF_BROADBAND,
    ],
)
def test_detection_function_all_upstream_types_execute(df_type: int) -> None:
    detector = rhythm.DetectionFunction(16, 32, df_type, db_rise=0.5)
    values = [detector.process_time_domain(np.hanning(32) * scale) for scale in (1.0, 0.5, 1.5)]
    assert np.asarray(values).shape == (3,)
    assert np.all(np.isfinite(values))


def test_peak_picking_returns_onset_indices_and_processed_curve() -> None:
    detection_function = np.zeros(64, dtype=np.float64)
    detection_function[[12, 32, 52]] = [1.0, 0.8, 1.2]
    picker = rhythm.PeakPicking(
        len(detection_function),
        tau=512 / 44100,
        win_pre=1,
        win_post=1,
        quad_a=0.0,
        quad_c=0.0,
    )

    result = picker.process(detection_function)

    assert result["processed"].shape == detection_function.shape
    assert result["onsets"].ndim == 1
    assert np.all(result["onsets"] >= 0)
    assert np.all(result["onsets"] < len(detection_function))


def test_tempo_track_v2_periods_beats_and_constrained_tempo() -> None:
    df = impulse_train(900, 43)
    tracker = rhythm.TempoTrackV2(44100, 512)

    unconstrained = tracker.calculate_beat_period(df)
    constrained = tracker.calculate_beat_period(df, input_tempo=120.0, constrain_tempo=True)
    beats = tracker.calculate_beats(df, unconstrained["beat_period"])

    assert unconstrained["beat_period"].shape == df.shape
    assert unconstrained["tempi"].shape == df.shape
    assert constrained["beat_period"].shape == df.shape
    assert np.all(unconstrained["beat_period"] > 0.0)
    assert np.all(unconstrained["tempi"] > 0.0)
    assert beats.ndim == 1
    assert np.all(np.diff(beats) > 0)


def test_tempo_track_v2_minimum_length_guard_matches_upstream_windowing() -> None:
    tracker = rhythm.TempoTrackV2(44100, 512)

    with pytest.raises(ValueError, match="more than 640"):
        tracker.calculate_beat_period(impulse_train(640, 43))

    result = tracker.calculate_beat_period(impulse_train(641, 43))

    assert result["beat_period"].shape == (641,)
    assert result["tempi"].shape == (641,)
    assert np.all(result["beat_period"] > 0.0)


def test_legacy_tempo_track_executes_and_returns_tempo_curve() -> None:
    df = impulse_train(1024, 43)
    tracker = rhythm.TempoTrack()

    result = tracker.process(df)

    assert result["beats"].ndim == 1
    assert result["tempo"].ndim == 1
    assert result["tempo"].size >= 1
    assert np.all(np.isfinite(result["tempo"]))


def test_downbeat_finds_downbeats_buffers_audio_and_reports_spectral_difference() -> None:
    tracker = rhythm.DownBeat(44100, 16, 512)
    tracker.set_beats_per_bar(4)
    audio = np.sin(np.linspace(0.0, 16 * np.pi, 4096, dtype=np.float64))
    beats = np.arange(0, 128, 16, dtype=np.float64)

    downbeats = tracker.find_down_beats(audio, beats)
    beat_sd = tracker.beat_sd()
    tracker.push_audio_block(np.linspace(-1.0, 1.0, 512, dtype=np.float64))
    buffered = tracker.buffered_audio()
    tracker.reset_audio_buffer()

    assert downbeats.ndim == 1
    assert np.all(downbeats >= 0)
    assert np.all(downbeats < beats.size)
    assert beat_sd.shape == (beats.size - 2,)
    assert buffered.shape == (32,)
    assert tracker.buffered_audio().shape == (0,)


def test_beat_spectrum_matches_known_cosine_distance_shape_and_range() -> None:
    matrix = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=np.float64,
    )
    result = rhythm.BeatSpectrum().process(matrix)

    assert result.shape == (2,)
    assert np.max(result) == pytest.approx(1.0)
    assert np.all(result >= 0.0)


def test_beats_returns_empty_result_below_tempo_track_v2_minimum() -> None:
    sample_rate = 44100
    audio = np.zeros(sample_rate * 6, dtype=np.float64)
    audio[:: sample_rate // 2] = 1.0

    result = rhythm.beats(audio, sample_rate)

    assert result["beats_s"].shape == (0,)
    assert result["tempo_curve"].shape == (0,)
    assert result["bpm"] == 0.0


def test_beats_returns_documented_numpy_result_for_120_bpm_click_track() -> None:
    sample_rate = 44100
    audio = np.zeros(sample_rate * 20, dtype=np.float64)
    audio[:: sample_rate // 2] = 1.0

    result = rhythm.beats(audio, sample_rate)

    assert set(result) == {"beats_s", "bpm", "tempo_curve"}
    assert result["beats_s"].ndim == 1
    assert result["tempo_curve"].ndim == 1
    assert result["beats_s"].size >= 20
    assert result["tempo_curve"].size + 1 == result["beats_s"].size
    assert result["bpm"] == pytest.approx(120.0, abs=8.0)
    assert float(np.median(result["tempo_curve"])) == pytest.approx(120.0, abs=8.0)


def test_python_and_native_guards_reject_bad_shapes_and_params() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        rhythm.DetectionFunction(4, 8).process_time_domain(np.zeros((2, 4)))
    with pytest.raises(ValueError, match="frame_length"):
        rhythm.DetectionFunction(4, 7)
    with pytest.raises(ValueError, match="df must contain more than 640"):
        rhythm.TempoTrackV2(44100, 512).calculate_beat_period(np.zeros(100))
    with pytest.raises(ValueError, match="decimation_factor"):
        rhythm.DownBeat(44100, 3, 512)
    with pytest.raises(ValueError, match="matrix"):
        rhythm.BeatSpectrum().process(np.zeros(4))
