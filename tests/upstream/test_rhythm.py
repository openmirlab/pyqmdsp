"""Parity tests against pristine qm-dsp rhythm/onset outputs."""

from __future__ import annotations

import numpy as np

from pyqmdsp import rhythm


def _array(values):
    return np.asarray(values, dtype=np.float64)


def _detection_outputs(inputs):
    outputs = {}
    frame_a = _array(inputs["detection_frame_a"])
    frame_b = _array(inputs["detection_frame_b"])
    reals = _array(inputs["detection_freq_reals"])
    imags = _array(inputs["detection_freq_imags"])
    for mode in inputs["df_modes"]:
        time_detector = rhythm.DetectionFunction(
            32,
            64,
            mode,
            db_rise=1.5,
            adaptive_whitening=True,
            whitening_relax_coeff=0.991,
            whitening_floor=0.02,
        )
        time = [
            time_detector.process_time_domain(frame_a),
            time_detector.process_time_domain(frame_b),
        ]
        spectrum_after_time = time_detector.spectrum_magnitude()

        freq_detector = rhythm.DetectionFunction(
            32,
            64,
            mode,
            db_rise=1.5,
            adaptive_whitening=True,
            whitening_relax_coeff=0.991,
            whitening_floor=0.02,
        )
        frequency = [
            freq_detector.process_frequency_domain(reals, imags),
            freq_detector.process_frequency_domain(reals, imags),
        ]
        spectrum_after_frequency = freq_detector.spectrum_magnitude()

        outputs[str(mode)] = {
            "time": time,
            "spectrum_after_time": spectrum_after_time,
            "frequency": frequency,
            "spectrum_after_frequency": spectrum_after_frequency,
        }
    return outputs


def _detection_frequency_rise_outputs(inputs):
    outputs = {}
    reals = [_array(row) for row in inputs["detection_rise_reals"]]
    imags = [_array(row) for row in inputs["detection_rise_imags"]]
    for mode in inputs["df_modes"]:
        detector = rhythm.DetectionFunction(
            32,
            64,
            mode,
            db_rise=1.0,
            adaptive_whitening=False,
            whitening_relax_coeff=-1.0,
            whitening_floor=-1.0,
        )
        frequency = [
            detector.process_frequency_domain(real, imag)
            for real, imag in zip(reals, imags, strict=True)
        ]
        outputs[str(mode)] = {
            "frequency": frequency,
            "spectrum_after_frequency": detector.spectrum_magnitude(),
        }
    return outputs


def _peak_picking_output(inputs):
    picker = rhythm.PeakPicking(
        len(inputs["peak_df"]),
        tau=512.0 / 44100.0,
        alpha=4,
        cutoff=0.35,
        win_pre=2,
        win_post=4,
        quad_a=0.01,
        quad_b=0.0,
        quad_c=0.03,
        delta=0.005,
    )
    return picker.process(_array(inputs["peak_df"]))


def _tempo_track_output(inputs):
    tracker = rhythm.TempoTrack(
        win_length=512,
        lag_length=128,
        alpha=4,
        win_pre=2,
        win_post=5,
    )
    return tracker.process(_array(inputs["tempo_df"]))


def _tempo_track_v2_output(inputs):
    tracker = rhythm.TempoTrackV2(44100, 512)
    periods = tracker.calculate_beat_period(
        _array(inputs["tempo_v2_df"]),
        input_tempo=132.0,
        constrain_tempo=True,
    )
    beats = tracker.calculate_beats(
        _array(inputs["tempo_v2_df"]),
        periods["beat_period"],
        alpha=0.72,
        tightness=5.5,
    )
    return {
        "beat_period": periods["beat_period"],
        "tempi": periods["tempi"],
        "beats": beats,
    }


def _downbeat_output(inputs):
    tracker = rhythm.DownBeat(8000, 8, 256)
    tracker.set_beats_per_bar(3)
    downbeats = tracker.find_down_beats(
        _array(inputs["downbeat_audio"]),
        _array(inputs["downbeat_beats"]),
    )
    beat_sd = tracker.beat_sd()
    tracker.push_audio_block(_array(inputs["downbeat_block_a"]))
    tracker.push_audio_block(_array(inputs["downbeat_block_b"]))
    buffered_before_reset = tracker.buffered_audio()
    tracker.reset_audio_buffer()
    buffered_after_reset = tracker.buffered_audio()
    return {
        "downbeats": downbeats,
        "beat_sd": beat_sd,
        "buffered_before_reset": buffered_before_reset,
        "buffered_after_reset": buffered_after_reset,
    }


def _beat_spectrum_output(inputs):
    return {
        "values": rhythm.BeatSpectrum().process(_array(inputs["beat_spectrum_matrix"])),
    }


def _extract_frame(audio, start, frame_length):
    frame = np.zeros(frame_length, dtype=np.float64)
    available = min(frame_length, len(audio) - start)
    if available > 0:
        frame[:available] = audio[start : start + available]
    return frame


def _beats_pipeline_output(inputs):
    audio = _array(inputs["beats_audio"])
    sample_rate = int(inputs["beats_sample_rate"])
    step = int(sample_rate * 0.01161 + 0.0001)
    frame_length = step * 2
    detector = rhythm.DetectionFunction(
        step,
        frame_length,
        rhythm.DF_COMPLEXSD,
        db_rise=3.0,
        adaptive_whitening=False,
        whitening_relax_coeff=-1.0,
        whitening_floor=-1.0,
    )
    raw_df = []
    for start in range(0, len(audio), step):
        raw_df.append(detector.process_time_domain(_extract_frame(audio, start, frame_length)))
    non_zero_count = len(raw_df)
    while non_zero_count > 0 and raw_df[non_zero_count - 1] <= 0.0:
        non_zero_count -= 1
    trimmed_df = np.asarray(raw_df[2:non_zero_count] if non_zero_count > 2 else [], dtype=np.float64)

    beat_period = np.zeros(len(trimmed_df), dtype=np.float64)
    tempi = np.asarray([], dtype=np.float64)
    beat_frames = np.asarray([], dtype=np.float64)
    if len(trimmed_df) > 640:
        tracker = rhythm.TempoTrackV2(sample_rate, step)
        period_result = tracker.calculate_beat_period(trimmed_df)
        beat_period = period_result["beat_period"]
        tempi = period_result["tempi"]
        if np.all(beat_period > 0.0):
            beat_frames = tracker.calculate_beats(trimmed_df, beat_period)

    beats_s = beat_frames * float(step) / float(sample_rate)
    tempo_curve = np.asarray(
        [60.0 / dt for dt in np.diff(beats_s) if dt > 0.0],
        dtype=np.float64,
    )
    bpm = 0.0
    if tempo_curve.size:
        bpm = float(np.median(tempo_curve))

    public = rhythm.beats(audio, sample_rate)
    np.testing.assert_allclose(public["beats_s"], beats_s, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(public["tempo_curve"], tempo_curve, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(public["bpm"], bpm, rtol=1e-12, atol=1e-12)

    return {
        "step": step,
        "frame_length": frame_length,
        "raw_df": np.asarray(raw_df, dtype=np.float64),
        "trimmed_df": trimmed_df,
        "beat_period": beat_period,
        "tempi": tempi,
        "beat_frames": beat_frames,
        "beats_s": beats_s,
        "bpm": bpm,
        "tempo_curve": tempo_curve,
    }


def test_rhythm_upstream_parity(reference, assert_matches):
    expected = reference("rhythm")
    inputs = expected["inputs"]
    broadband_rise = np.asarray(
        expected["outputs"]["detection_frequency_rise"][str(rhythm.DF_BROADBAND)]["frequency"],
        dtype=np.float64,
    )
    assert np.any(broadband_rise > 0.0)

    actual_outputs = {
        "detection": _detection_outputs(inputs),
        "detection_frequency_rise": _detection_frequency_rise_outputs(inputs),
        "peak_picking": _peak_picking_output(inputs),
        "tempo_track": _tempo_track_output(inputs),
        "tempo_track_v2": _tempo_track_v2_output(inputs),
        "downbeat": _downbeat_output(inputs),
        "beat_spectrum": _beat_spectrum_output(inputs),
        "beats_pipeline": _beats_pipeline_output(inputs),
    }
    assert_matches(actual_outputs, expected["outputs"])
