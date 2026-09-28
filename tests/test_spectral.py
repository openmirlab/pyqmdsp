import numpy as np
import pytest

from pyqmdsp import spectral


def _frequency_for_midi(midi_pitch: int, concert_a: float = 440.0) -> float:
    return concert_a * 2 ** ((midi_pitch - 69.0) / 12.0)


def _sinusoid(frequency: float, sample_rate: float, length: int) -> np.ndarray:
    samples = np.arange(length, dtype=np.float64)
    return np.sin(samples * 2.0 * np.pi * frequency / sample_rate)


def _periodic_hamming(length: int) -> np.ndarray:
    samples = np.arange(length, dtype=np.float64)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * samples / length)


def _fftshift_frame(frame: np.ndarray) -> np.ndarray:
    midpoint = frame.size // 2
    return np.concatenate([frame[midpoint:], frame[:midpoint]])


def _dct_type_ii(values: np.ndarray, *, unitary: bool) -> np.ndarray:
    n = values.size
    index = np.arange(n, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    for k in range(n):
        result[k] = np.sum(values * np.cos(np.pi * k * (index + 0.5) / n))
        if unitary:
            result[k] *= np.sqrt(2.0 / n)
            if k == 0:
                result[k] /= np.sqrt(2.0)
        else:
            result[k] *= 2.0
    return result


def test_complex_fft_matches_numpy_and_round_trips():
    real = np.array([0.5, -1.0, 0.25, 2.0, -0.75, 0.125, 1.5, -0.25])
    imag = np.array([0.0, 0.25, -0.5, 0.75, -1.0, 0.5, 0.125, -0.25])
    expected = np.fft.fft(real + 1j * imag)

    fft = spectral.FFT(real.size)
    freq_real, freq_imag = fft.process(real, imag)
    out_real, out_imag = fft.process(freq_real, freq_imag, inverse=True)

    assert np.allclose(freq_real, expected.real, atol=1e-12)
    assert np.allclose(freq_imag, expected.imag, atol=1e-12)
    assert np.allclose(out_real, real, atol=1e-12)
    assert np.allclose(out_imag, imag, atol=1e-12)


def test_real_fft_matches_numpy_and_round_trips_from_non_redundant_bins():
    samples = np.array([1.0, 0.0, -1.0, 0.5, 0.25, -0.75, 0.125, -0.5])
    expected = np.fft.fft(samples)

    fft = spectral.RealFFT(samples.size)
    real, imag = fft.forward(samples)
    restored = fft.inverse(real[: fft.bins], imag[: fft.bins])

    assert np.allclose(real, expected.real, atol=1e-12)
    assert np.allclose(imag, expected.imag, atol=1e-12)
    assert np.allclose(restored, samples, atol=1e-12)
    assert np.allclose(fft.forward_magnitude(samples), np.hypot(real, imag), atol=1e-12)


def test_dct_matches_explicit_definition_and_round_trips():
    samples = np.array([0.0, 1.0, -2.0, 0.5, 3.0, -1.0], dtype=np.float64)

    dct = spectral.DCT(samples.size)
    coefficients = dct.forward(samples)
    unitary_coefficients = dct.forward(samples, unitary=True)
    restored = dct.inverse(unitary_coefficients, unitary=True)

    assert np.allclose(coefficients, _dct_type_ii(samples, unitary=False), atol=1e-12)
    assert np.allclose(
        unitary_coefficients, _dct_type_ii(samples, unitary=True), atol=1e-12
    )
    assert np.allclose(restored, samples, atol=1e-12)
    assert np.allclose(spectral.idct(spectral.dct(samples, unitary=True), unitary=True), samples)


def test_constant_q_and_chromagram_peak_at_expected_pitch_class():
    sample_rate = 44100.0
    bins_per_octave = 36
    min_pitch = 36
    max_pitch = 96
    probe_pitch = 60
    min_frequency = _frequency_for_midi(min_pitch)
    max_frequency = _frequency_for_midi(max_pitch)
    frequency = _frequency_for_midi(probe_pitch)
    expected_cq_bin = (probe_pitch - min_pitch) * (bins_per_octave // 12)
    expected_chroma_bin = expected_cq_bin % bins_per_octave

    cq = spectral.ConstantQ(
        sample_rate, min_frequency, max_frequency, bins_per_octave=bins_per_octave
    )
    fft = spectral.RealFFT(cq.fft_length)
    frame = _sinusoid(frequency, sample_rate, cq.fft_length)
    shifted = _fftshift_frame(frame * _periodic_hamming(cq.fft_length))
    real, imag = fft.forward(shifted)

    cq_real, cq_imag = cq.process_frequency(real, imag)
    cq_magnitude = np.hypot(cq_real, cq_imag)

    assert cq_real.shape == (cq.bins,)
    assert cq_imag.shape == (cq.bins,)
    assert np.argmax(cq_magnitude) == expected_cq_bin

    chroma = spectral.Chromagram(
        sample_rate,
        min_frequency,
        max_frequency,
        bins_per_octave=bins_per_octave,
        normalise="none",
    )
    chroma_frame = chroma.process_time(_sinusoid(frequency, sample_rate, chroma.frame_size))

    assert chroma_frame.shape == (bins_per_octave,)
    assert np.argmax(chroma_frame) == expected_chroma_bin
    assert np.allclose(
        chroma.unity_normalise(np.array([0.0, 2.0, 4.0] * 12)), [0.0, 0.5, 1.0] * 12
    )
    assert chroma.kabs(3.0, 4.0) == 5.0

    with pytest.raises(ValueError, match="non-zero maximum"):
        chroma.unity_normalise(np.zeros(chroma.bins))


def test_mfcc_zero_input_has_exact_zero_coefficients():
    mfcc = spectral.MFCC(8000, fft_size=64, n_coefficients=5, want_c0=True)
    coefficients = mfcc.process_time(np.zeros(mfcc.fft_length))

    assert coefficients.shape == (6,)
    assert np.array_equal(coefficients, np.zeros(6))


def test_phase_vocoder_matches_upstream_fullcycle_fixture():
    frame = np.array([1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0, 0.0])
    pv = spectral.PhaseVocoder(size=8, hop=4)

    mag0, phase0, unwrapped0 = pv.process_time(frame)
    mag1, phase1, unwrapped1 = pv.process_time(frame)
    mag2, phase2, unwrapped2 = pv.process_time(frame)

    assert np.array_equal(mag0, [0.0, 0.0, 4.0, 0.0, 0.0])
    assert np.array_equal(mag1, [0.0, 0.0, 4.0, 0.0, 0.0])
    assert np.array_equal(mag2, [0.0, 0.0, 4.0, 0.0, 0.0])
    assert np.allclose(phase0, np.zeros(5), atol=1e-7)
    assert np.allclose(phase1, np.zeros(5), atol=1e-7)
    assert np.allclose(phase2, np.zeros(5), atol=1e-7)
    assert np.allclose(unwrapped0, np.zeros(5), atol=1e-7)
    assert np.allclose(unwrapped1, [0.0, 2 * np.pi, 2 * np.pi, 4 * np.pi, 4 * np.pi])
    assert np.allclose(unwrapped2, [0.0, 4 * np.pi, 4 * np.pi, 8 * np.pi, 8 * np.pi])


def test_phase_vocoder_frequency_path_matches_fft_magnitudes():
    frame = np.array([0.0, 1.0, 0.0, -1.0, 0.0, 1.0, 0.0, -1.0])
    pv = spectral.PhaseVocoder(size=8, hop=4)
    real, imag = spectral.RealFFT(8).forward(frame)
    mag, phase, unwrapped = pv.process_frequency(real[: pv.bins], imag[: pv.bins])

    assert np.allclose(mag, np.abs(np.fft.fft(frame)[: pv.bins]), atol=1e-12)
    assert phase.shape == unwrapped.shape == (pv.bins,)


def test_key_mode_tracks_single_tone_tonic_like_upstream_fixture():
    sample_rate = 44100
    midi_pitch = 60
    detector = spectral.KeyMode(sample_rate)
    block_size = detector.block_size
    hop_size = detector.hop_size
    signal = _sinusoid(_frequency_for_midi(midi_pitch), sample_rate, block_size * 4)

    keys = []
    for offset in range(0, signal.size - block_size, hop_size):
        keys.append(detector.process(signal[offset : offset + block_size]))

    tonics = [key - 12 if key > 12 else key for key in keys]
    strengths = detector.key_strengths()

    assert tonics
    assert set(tonics) == {1 + (midi_pitch % 12)}
    assert strengths.shape == (24,)
    assert np.all(np.isfinite(strengths))


def test_tonal_transform_matches_explicit_basis():
    chroma = np.zeros(12)
    chroma[0] = 1.0
    unnormalised = np.arange(1.0, 13.0)
    tcs = spectral.tonal_transform(chroma)
    normalised = spectral.normalize_chroma(unnormalised)
    pitch_classes = np.arange(12, dtype=np.float64)
    expected = np.array(
        [
            np.sum(np.sin((7.0 / 6.0) * pitch_classes * np.pi) * chroma),
            np.sum(np.cos((7.0 / 6.0) * pitch_classes * np.pi) * chroma),
            np.sum(0.6 * np.sin((2.0 / 3.0) * pitch_classes * np.pi) * chroma),
            np.sum(0.6 * np.cos((2.0 / 3.0) * pitch_classes * np.pi) * chroma),
            np.sum(1.1 * np.sin((3.0 / 2.0) * pitch_classes * np.pi) * chroma),
            np.sum(1.1 * np.cos((3.0 / 2.0) * pitch_classes * np.pi) * chroma),
        ]
    )

    assert np.allclose(tcs, expected, atol=1e-12)
    assert np.allclose(normalised, unnormalised / np.sum(np.abs(unnormalised)))
    assert spectral.tonal_magnitude([3.0, 4.0, 0.0, 0.0, 0.0, 12.0]) == 13.0


def test_change_detection_tcsgram_and_wavelet_filters_have_expected_values():
    tcs = np.array([0.0, 1.0, 0.0, 0.6, 0.0, 1.1])

    gram = spectral.TCSGram(frame_duration_ms=10.0)
    gram.add(tcs)
    gram.add(tcs)
    gram.add(tcs)

    assert gram.size == 3
    assert gram.time_at(2) == 20
    assert gram.matrix().shape == (3, 6)

    timed = spectral.TCSGram(frame_duration_ms=10.0)
    timed.add(tcs)
    timed.set_frame_duration_ms(20.0)
    timed.add(tcs)
    assert timed.time_at(1) == 20

    changes = spectral.ChangeDetection(1).process_tcsgram(gram)
    changes_from_matrix = spectral.ChangeDetection(1).process_matrix(gram.matrix())
    sigma = 3.0 / (2.0 * 2.3548)
    gaussian = np.array(
        [
            (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-(x * x) / (2.0 * sigma * sigma))
            for x in (-1.0, 0.0, 1.0)
        ]
    )
    expected_edge = np.sum(gaussian) * np.linalg.norm(tcs)

    assert changes.shape == (3,)
    assert np.allclose(changes, [expected_edge, 0.0, expected_edge])
    assert np.allclose(changes, changes_from_matrix)

    names = spectral.wavelet_names()
    low, high = spectral.wavelet_filters("Haar")

    assert names[0] == "Haar"
    assert np.allclose(low, [2**-0.5, 2**-0.5])
    assert np.allclose(high, [-2**-0.5, 2**-0.5])


def test_invalid_inputs_are_rejected_before_unsafe_native_calls():
    with pytest.raises(ValueError, match="positive"):
        spectral.FFT(0)

    with pytest.raises(ValueError, match="even"):
        spectral.RealFFT(7)

    with pytest.raises(ValueError, match="wrong length"):
        spectral.FFT(4).process(np.ones(3))

    with pytest.raises(ValueError, match="finite"):
        spectral.dct([1.0, np.nan])

    with pytest.raises(ValueError, match="shape"):
        spectral.ChangeDetection(1).process_matrix(np.ones((3, 5)))

    with pytest.raises(ValueError, match="unknown wavelet"):
        spectral.wavelet_filters("No Such Wavelet")

    with pytest.raises(ValueError, match="smoothing_width"):
        spectral.ChangeDetection(-1)
