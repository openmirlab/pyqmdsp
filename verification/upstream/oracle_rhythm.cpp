// Pristine qm-dsp rhythm/onset oracle; does not include binding code.
// Reads: oracle.h and upstream dsp/onsets, dsp/tempotracking, dsp/rhythm.
#include "oracle.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <stdexcept>
#include <vector>

#include "dsp/onsets/DetectionFunction.h"
#include "dsp/onsets/PeakPicking.h"
#include "dsp/rhythm/BeatSpectrum.h"
#include "dsp/tempotracking/DownBeat.h"
#include "dsp/tempotracking/TempoTrack.h"
#include "dsp/tempotracking/TempoTrackV2.h"

namespace {

constexpr double kStepSecs = 0.01161;

std::vector<double> default_lpa() { return {1.0, -0.3695, 0.1958}; }
std::vector<double> default_lpb() { return {0.2066, 0.4131, 0.2066}; }

std::vector<float> to_float(const std::vector<double> &values) {
    return std::vector<float>(values.begin(), values.end());
}

std::vector<double> float_to_double(const float *values, std::size_t length) {
    std::vector<double> out(length);
    for (std::size_t i = 0; i < length; ++i) out[i] = values[i];
    return out;
}

std::vector<double> musical_frame(std::size_t n, double phase, double accent) {
    std::vector<double> out = signal(n, phase);
    for (std::size_t i = 0; i < n; ++i) {
        out[i] += accent * std::sin(0.021 * double(i * i + 3));
        if (i % 19 == 0) out[i] += 0.4;
    }
    return out;
}

std::vector<double> detection_curve(std::size_t n, int period, double phase) {
    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i) {
        double pulse = (i % std::size_t(period) == 0) ? 1.0 : 0.0;
        double neighbor = (i % std::size_t(period) == 1) ? 0.45 : 0.0;
        out[i] = pulse + neighbor + 0.09 * std::sin(0.037 * double(i) + phase)
                 + 0.04 * std::cos(0.131 * double(i));
        if (out[i] < 0.0) out[i] = 0.0;
    }
    return out;
}

std::vector<double> beat_audio(std::size_t n, int sample_rate) {
    std::vector<double> out(n);
    int half_second = sample_rate / 2;
    for (std::size_t i = 0; i < n; ++i) {
        double base = 0.08 * std::sin(2.0 * M_PI * 220.0 * double(i) / sample_rate)
                      + 0.04 * std::sin(2.0 * M_PI * 331.0 * double(i) / sample_rate);
        int offset = int(i % std::size_t(half_second));
        double click = offset < 12 ? 1.0 - double(offset) / 12.0 : 0.0;
        out[i] = base + click;
    }
    return out;
}

DFConfig df_config(int type) {
    DFConfig config;
    config.stepSize = 32;
    config.frameLength = 64;
    config.DFType = type;
    config.dbRise = 1.5;
    config.adaptiveWhitening = true;
    config.whiteningRelaxCoeff = 0.991;
    config.whiteningFloor = 0.02;
    return config;
}

DFConfig df_config_no_whitening(int type) {
    DFConfig config = df_config(type);
    config.dbRise = 1.0;
    config.adaptiveWhitening = false;
    config.whiteningRelaxCoeff = -1.0;
    config.whiteningFloor = -1.0;
    return config;
}

std::vector<double> spectrum(DetectionFunction &df, int frame_length) {
    double *src = df.getSpectrumMagnitude();
    return std::vector<double>(src, src + frame_length / 2 + 1);
}

Json detection_outputs(const std::vector<double> &frame_a,
                       const std::vector<double> &frame_b,
                       const std::vector<double> &freq_reals,
                       const std::vector<double> &freq_imags) {
    Json all;
    for (int mode : {DF_HFC, DF_SPECDIFF, DF_PHASEDEV, DF_COMPLEXSD, DF_BROADBAND}) {
        DetectionFunction time_df(df_config(mode));
        std::vector<double> time_values;
        time_values.push_back(time_df.processTimeDomain(frame_a.data()));
        time_values.push_back(time_df.processTimeDomain(frame_b.data()));

        DetectionFunction freq_df(df_config(mode));
        std::vector<double> freq_values;
        freq_values.push_back(freq_df.processFrequencyDomain(freq_reals.data(), freq_imags.data()));
        freq_values.push_back(freq_df.processFrequencyDomain(freq_reals.data(), freq_imags.data()));

        Json one;
        one.set("time", time_values);
        one.set("spectrum_after_time", spectrum(time_df, 64));
        one.set("frequency", freq_values);
        one.set("spectrum_after_frequency", spectrum(freq_df, 64));
        all.set(std::to_string(mode), one);
    }
    return all;
}

Json detection_frequency_rise_outputs(
    const std::vector<std::vector<double>> &reals,
    const std::vector<std::vector<double>> &imags) {
    Json all;
    for (int mode : {DF_HFC, DF_SPECDIFF, DF_PHASEDEV, DF_COMPLEXSD, DF_BROADBAND}) {
        DetectionFunction df(df_config_no_whitening(mode));
        std::vector<double> values;
        for (std::size_t i = 0; i < reals.size(); ++i) {
            values.push_back(df.processFrequencyDomain(reals[i].data(), imags[i].data()));
        }
        Json one;
        one.set("frequency", values);
        one.set("spectrum_after_frequency", spectrum(df, 64));
        all.set(std::to_string(mode), one);
    }
    return all;
}

Json peak_picking_output(std::vector<double> curve) {
    std::vector<double> a = default_lpa();
    std::vector<double> b = default_lpb();
    PPickParams params;
    params.length = int(curve.size());
    params.tau = 512.0 / 44100.0;
    params.alpha = 4;
    params.cutoff = 0.35;
    params.LPOrd = 2;
    params.LPACoeffs = a.data();
    params.LPBCoeffs = b.data();
    params.WinT = PPWinThresh(2, 4);
    params.QuadThresh = QFitThresh(0.01, 0.0, 0.03);
    params.delta = 0.005f;
    PeakPicking picker(params);
    std::vector<int> onsets;
    picker.process(curve.data(), int(curve.size()), onsets);
    Json out;
    out.set("processed", curve);
    out.set("onsets", onsets);
    return out;
}

Json tempo_track_output(const std::vector<double> &curve) {
    std::vector<double> a = default_lpa();
    std::vector<double> b = default_lpb();
    WinThresh win;
    win.pre = 2;
    win.post = 5;
    TTParams params;
    params.winLength = 512;
    params.lagLength = 128;
    params.alpha = 4;
    params.LPOrd = 2;
    params.LPACoeffs = a.data();
    params.LPBCoeffs = b.data();
    params.WinT = win;
    TempoTrack tracker(params);
    std::vector<double> tempo;
    std::vector<int> beats = tracker.process(curve, &tempo);
    Json out;
    out.set("beats", beats);
    out.set("tempo", tempo);
    return out;
}

Json tempo_track_v2_output(const std::vector<double> &curve) {
    TempoTrackV2 tracker(44100.0f, 512);
    std::vector<double> periods(curve.size(), 0.0);
    std::vector<double> tempi;
    tracker.calculateBeatPeriod(curve, periods, tempi, 132.0, true);
    std::vector<double> beats;
    tracker.calculateBeats(curve, periods, beats, 0.72, 5.5);
    Json out;
    out.set("beat_period", periods);
    out.set("tempi", tempi);
    out.set("beats", beats);
    return out;
}

Json downbeat_output(const std::vector<double> &audio_double,
                     const std::vector<double> &beats,
                     const std::vector<double> &block_a,
                     const std::vector<double> &block_b) {
    DownBeat tracker(8000.0f, 8, 256);
    tracker.setBeatsPerBar(3);
    std::vector<float> audio = to_float(audio_double);
    std::vector<int> downbeats;
    tracker.findDownBeats(audio.data(), audio.size(), beats, downbeats);
    std::vector<double> beat_sd;
    tracker.getBeatSD(beat_sd);

    std::vector<float> a = to_float(block_a);
    std::vector<float> b = to_float(block_b);
    tracker.pushAudioBlock(a.data());
    tracker.pushAudioBlock(b.data());
    std::size_t buffered_length = 0;
    const float *buffered = tracker.getBufferedAudio(buffered_length);
    std::vector<double> buffered_before_reset = float_to_double(buffered, buffered_length);
    tracker.resetAudioBuffer();
    buffered = tracker.getBufferedAudio(buffered_length);
    std::vector<double> buffered_after_reset = float_to_double(buffered, buffered_length);

    Json out;
    out.set("downbeats", downbeats);
    out.set("beat_sd", beat_sd);
    out.set("buffered_before_reset", buffered_before_reset);
    out.set("buffered_after_reset", buffered_after_reset);
    return out;
}

Json beat_spectrum_output(const std::vector<std::vector<double>> &matrix) {
    BeatSpectrum spectrum;
    Json out;
    out.set("values", spectrum.process(matrix));
    return out;
}

void extract_frame(const std::vector<double> &mono, std::size_t start, std::vector<double> &frame) {
    std::fill(frame.begin(), frame.end(), 0.0);
    if (start >= mono.size()) return;
    std::size_t available = std::min(frame.size(), mono.size() - start);
    std::copy_n(mono.begin() + static_cast<std::ptrdiff_t>(start), available, frame.begin());
}

Json beats_pipeline_output(const std::vector<double> &audio, int sample_rate) {
    int step = static_cast<int>(sample_rate * kStepSecs + 0.0001);
    int frame_length = step * 2;
    DFConfig config;
    config.stepSize = step;
    config.frameLength = frame_length;
    config.DFType = DF_COMPLEXSD;
    config.dbRise = 3.0;
    config.adaptiveWhitening = false;
    config.whiteningRelaxCoeff = -1.0;
    config.whiteningFloor = -1.0;
    DetectionFunction df(config);
    std::vector<double> frame(std::size_t(frame_length), 0.0);
    std::vector<double> raw_df;
    for (std::size_t k = 0;; ++k) {
        std::size_t start = k * std::size_t(step);
        if (start >= audio.size()) break;
        extract_frame(audio, start, frame);
        raw_df.push_back(df.processTimeDomain(frame.data()));
    }

    std::size_t non_zero_count = raw_df.size();
    while (non_zero_count > 0 && raw_df[non_zero_count - 1] <= 0.0) --non_zero_count;
    std::vector<double> trimmed_df;
    if (non_zero_count > 2) {
        trimmed_df.assign(raw_df.begin() + 2, raw_df.begin() + static_cast<long>(non_zero_count));
    }

    std::vector<double> beat_period(trimmed_df.size(), 0.0);
    std::vector<double> tempi;
    std::vector<double> beat_frames;
    if (trimmed_df.size() > 640) {
        TempoTrackV2 tracker(static_cast<float>(sample_rate), step);
        tracker.calculateBeatPeriod(trimmed_df, beat_period, tempi, 120.0, false);
        bool usable = std::all_of(beat_period.begin(), beat_period.end(), [](double x) {
            return x > 0.0;
        });
        if (usable) tracker.calculateBeats(trimmed_df, beat_period, beat_frames, 0.9, 4.0);
    }

    std::vector<double> beats_s;
    for (double beat : beat_frames) {
        beats_s.push_back((beat * double(step)) / double(sample_rate));
    }
    std::vector<double> tempo_curve;
    for (std::size_t i = 0; i + 1 < beats_s.size(); ++i) {
        double dt = beats_s[i + 1] - beats_s[i];
        if (dt > 0.0) tempo_curve.push_back(60.0 / dt);
    }
    double bpm = 0.0;
    if (!tempo_curve.empty()) {
        std::vector<double> sorted = tempo_curve;
        std::size_t mid = sorted.size() / 2;
        std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(mid), sorted.end());
        bpm = sorted[mid];
        if (sorted.size() % 2 == 0) {
            std::nth_element(sorted.begin(), sorted.begin() + static_cast<long>(mid - 1), sorted.end());
            bpm = (bpm + sorted[mid - 1]) / 2.0;
        }
    }

    Json out;
    out.set("step", step);
    out.set("frame_length", frame_length);
    out.set("raw_df", raw_df);
    out.set("trimmed_df", trimmed_df);
    out.set("beat_period", beat_period);
    out.set("tempi", tempi);
    out.set("beat_frames", beat_frames);
    out.set("beats_s", beats_s);
    out.set("bpm", bpm);
    out.set("tempo_curve", tempo_curve);
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) throw std::runtime_error("usage: oracle_rhythm OUT.json");

    std::vector<double> frame_a = musical_frame(64, 0.2, 0.03);
    std::vector<double> frame_b = musical_frame(64, 0.8, 0.05);
    std::vector<double> freq_reals = musical_frame(33, 0.4, 0.02);
    std::vector<double> freq_imags = musical_frame(33, 1.1, 0.01);
    std::vector<std::vector<double>> rise_reals;
    std::vector<std::vector<double>> rise_imags;
    for (double scale : {0.08, 0.35, 1.4}) {
        std::vector<double> real = musical_frame(33, 0.6 + scale, 0.01);
        std::vector<double> imag = musical_frame(33, 1.4 + scale, 0.01);
        for (std::size_t i = 0; i < real.size(); ++i) {
            real[i] = scale * (std::abs(real[i]) + 0.05);
            imag[i] = scale * (std::abs(imag[i]) + 0.03);
        }
        rise_reals.push_back(real);
        rise_imags.push_back(imag);
    }
    std::vector<double> peak_df = detection_curve(96, 17, 0.3);
    std::vector<double> tempo_df = detection_curve(1152, 43, 0.7);
    std::vector<double> tempo_v2_df = detection_curve(900, 43, 1.2);
    std::vector<double> downbeat_audio = musical_frame(2048, 0.5, 0.08);
    std::vector<double> downbeat_beats = {0, 8, 15, 23, 31, 40, 48, 57, 65, 74};
    std::vector<double> downbeat_block_a = musical_frame(256, 0.9, 0.04);
    std::vector<double> downbeat_block_b = musical_frame(256, 1.7, 0.06);
    std::vector<std::vector<double>> beat_matrix;
    for (int r = 0; r < 6; ++r) {
        beat_matrix.push_back(signal(4, 0.25 * r));
        for (int c = 0; c < 4; ++c) beat_matrix.back()[c] += 0.1 * r + 0.03 * c;
    }
    int beats_sample_rate = 8000;
    std::vector<double> beats_audio = beat_audio(76000, beats_sample_rate);

    Json inputs;
    inputs.set("df_modes", std::vector<int>{DF_HFC, DF_SPECDIFF, DF_PHASEDEV, DF_COMPLEXSD, DF_BROADBAND});
    inputs.set("detection_frame_a", frame_a);
    inputs.set("detection_frame_b", frame_b);
    inputs.set("detection_freq_reals", freq_reals);
    inputs.set("detection_freq_imags", freq_imags);
    inputs.set("detection_rise_reals", rise_reals);
    inputs.set("detection_rise_imags", rise_imags);
    inputs.set("peak_df", peak_df);
    inputs.set("tempo_df", tempo_df);
    inputs.set("tempo_v2_df", tempo_v2_df);
    inputs.set("downbeat_audio", downbeat_audio);
    inputs.set("downbeat_beats", downbeat_beats);
    inputs.set("downbeat_block_a", downbeat_block_a);
    inputs.set("downbeat_block_b", downbeat_block_b);
    inputs.set("beat_spectrum_matrix", beat_matrix);
    inputs.set("beats_audio", beats_audio);
    inputs.set("beats_sample_rate", beats_sample_rate);

    Json outputs;
    outputs.set("detection", detection_outputs(frame_a, frame_b, freq_reals, freq_imags));
    outputs.set("detection_frequency_rise", detection_frequency_rise_outputs(rise_reals, rise_imags));
    outputs.set("peak_picking", peak_picking_output(peak_df));
    outputs.set("tempo_track", tempo_track_output(tempo_df));
    outputs.set("tempo_track_v2", tempo_track_v2_output(tempo_v2_df));
    outputs.set("downbeat", downbeat_output(downbeat_audio, downbeat_beats, downbeat_block_a, downbeat_block_b));
    outputs.set("beat_spectrum", beat_spectrum_output(beat_matrix));
    outputs.set("beats_pipeline", beats_pipeline_output(beats_audio, beats_sample_rate));

    Json root;
    root.set("inputs", inputs);
    root.set("outputs", outputs);
    root.save(argv[1]);
}
