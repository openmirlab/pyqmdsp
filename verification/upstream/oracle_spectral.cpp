// Original spectral and tonal outputs with full input frames for Python replay.
// Reads: pristine qm-dsp APIs and fixture-only oracle.h; no binding code.
#include "oracle.h"

#include "base/Window.h"
#include "dsp/chromagram/Chromagram.h"
#include "dsp/chromagram/ConstantQ.h"
#include "dsp/keydetection/GetKeyMode.h"
#include "dsp/mfcc/MFCC.h"
#include "dsp/phasevocoder/PhaseVocoder.h"
#include "dsp/tonal/ChangeDetectionFunction.h"
#include "dsp/tonal/TCSgram.h"
#include "dsp/tonal/TonalEstimator.h"
#include "dsp/transforms/DCT.h"
#include "dsp/transforms/FFT.h"
#include "dsp/wavelet/Wavelet.h"
#include "maths/MathUtilities.h"

#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<double> ramp(size_t n, double offset, double scale) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = offset + scale * double(int(i % 11) - 5) + 0.07 * std::sin(0.31 * double(i));
    }
    return x;
}

std::vector<double> sinusoid(double frequency, double sample_rate, size_t n, double phase = 0.0) {
    std::vector<double> x(n);
    for (size_t i = 0; i < n; ++i) {
        x[i] = std::sin((double(i) * M_PI * 2.0 * frequency / sample_rate) + phase);
    }
    return x;
}

double frequency_for_midi(int midi_pitch, double concert_a = 440.0) {
    return concert_a * std::pow(2.0, (double(midi_pitch) - 69.0) / 12.0);
}

std::vector<double> fft_shifted_hamming(const std::vector<double> &frame) {
    std::vector<double> shifted(frame.size());
    std::vector<double> windowed(frame);
    Window<double> window(HammingWindow, int(frame.size()));
    window.cut(windowed.data());
    size_t half = frame.size() / 2;
    for (size_t i = 0; i < half; ++i) {
        shifted[i] = windowed[i + half];
        shifted[i + half] = windowed[i];
    }
    return shifted;
}

Json triple_json(const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &c) {
    Json j;
    j.set("mag", a);
    j.set("phase", b);
    j.set("unwrapped", c);
    return j;
}

Json pair_json(const std::vector<double> &a, const std::vector<double> &b) {
    Json j;
    j.set("real", a);
    j.set("imag", b);
    return j;
}

std::vector<double> tcs_vector_to_std(const TCSVector &v) {
    std::vector<double> out(6);
    for (size_t i = 0; i < 6; ++i) out[i] = v[i];
    return out;
}

TCSVector make_tcs_vector(const std::vector<double> &values) {
    if (values.size() != 6) throw std::runtime_error("tcs input must have six values");
    TCSVector v;
    for (size_t i = 0; i < 6; ++i) v[i] = values[i];
    return v;
}

TCSGram make_tcsgram(const std::vector<std::vector<double>> &rows, double frame_duration_ms) {
    TCSGram gram;
    gram.setFrameDuration(frame_duration_ms);
    gram.setNumBins(6);
    gram.reserve(rows.size());
    for (const auto &row : rows) gram.addTCSVector(make_tcs_vector(row));
    return gram;
}

std::vector<double> tcsgram_vector(const TCSGram &gram, int index) {
    TCSVector v;
    gram.getTCSVector(index, v);
    return tcs_vector_to_std(v);
}

std::vector<std::vector<double>> tcsgram_matrix(const TCSGram &gram) {
    std::vector<std::vector<double>> out;
    for (int i = 0; i < gram.getSize(); ++i) out.push_back(tcsgram_vector(gram, i));
    return out;
}

std::vector<double> change_to_std(const ChangeDistance &distance) {
    std::vector<double> out(distance.size());
    for (size_t i = 0; i < distance.size(); ++i) out[i] = distance[i];
    return out;
}

std::vector<double> chroma_normalize_l1(const std::vector<double> &values) {
    ChromaVector chroma(12);
    for (size_t i = 0; i < 12; ++i) chroma[i] = values[i];
    chroma.normalizeL1();
    std::vector<double> out(12);
    for (size_t i = 0; i < 12; ++i) out[i] = chroma[i];
    return out;
}

Json wavelet_output(int wavelet) {
    Wavelet::Type type = static_cast<Wavelet::Type>(wavelet);
    std::vector<double> low;
    std::vector<double> high;
    Wavelet::createDecompositionFilters(type, low, high);
    Json j;
    j.set("name", Wavelet::getWaveletName(type));
    j.set("low", low);
    j.set("high", high);
    return j;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc != 2) throw std::runtime_error("usage: oracle_spectral output.json");

    Json inputs;
    Json outputs;
    Json convenience;

    {
        const int n = 9;
        std::vector<double> real = ramp(n, 0.2, 0.11);
        std::vector<double> imag = ramp(n, -0.15, -0.08);
        FFT fft(n);
        std::vector<double> out_real(n), out_imag(n), inv_real(n), inv_imag(n), real_only(n), real_only_imag(n);
        fft.process(false, real.data(), imag.data(), out_real.data(), out_imag.data());
        fft.process(true, out_real.data(), out_imag.data(), inv_real.data(), inv_imag.data());
        fft.process(false, real.data(), nullptr, real_only.data(), real_only_imag.data());
        Json in;
        in.set("real", real);
        in.set("imag", imag);
        inputs.set("fft", in);
        Json out;
        out.set("forward", pair_json(out_real, out_imag));
        out.set("inverse", pair_json(inv_real, inv_imag));
        out.set("real_only_forward", pair_json(real_only, real_only_imag));
        outputs.set("fft", out);
        convenience.set("fft", pair_json(out_real, out_imag));
    }

    {
        const int n = 16;
        std::vector<double> frame = signal(n, 0.37);
        FFTReal fft(n);
        std::vector<double> real(n), imag(n), mag(n), inverse(n);
        fft.forward(frame.data(), real.data(), imag.data());
        fft.forwardMagnitude(frame.data(), mag.data());
        fft.inverse(real.data(), imag.data(), inverse.data());
        inputs.set("real_fft", frame);
        Json out;
        out.set("forward", pair_json(real, imag));
        out.set("magnitude", mag);
        out.set("inverse", inverse);
        outputs.set("real_fft", out);
        convenience.set("real_fft", pair_json(real, imag));
    }

    {
        std::vector<double> frame = ramp(7, -0.1, 0.13);
        DCT dct(int(frame.size()));
        std::vector<double> forward(frame.size()), forward_unitary(frame.size());
        std::vector<double> inverse(frame.size()), inverse_unitary(frame.size());
        dct.forward(frame.data(), forward.data());
        dct.forwardUnitary(frame.data(), forward_unitary.data());
        dct.inverse(forward.data(), inverse.data());
        dct.inverseUnitary(forward_unitary.data(), inverse_unitary.data());
        inputs.set("dct", frame);
        Json out;
        out.set("forward", forward);
        out.set("forward_unitary", forward_unitary);
        out.set("inverse", inverse);
        out.set("inverse_unitary", inverse_unitary);
        outputs.set("dct", out);
        convenience.set("dct", forward_unitary);
        convenience.set("idct", inverse_unitary);
    }

    {
        CQConfig config{22050.0, frequency_for_midi(40), frequency_for_midi(88), 24, 0.004};
        ConstantQ cq(config);
        cq.sparsekernel();
        std::vector<double> time = sinusoid(frequency_for_midi(52), config.FS, cq.getFFTLength(), 0.2);
        std::vector<double> shifted = fft_shifted_hamming(time);
        FFTReal fft(cq.getFFTLength());
        std::vector<double> real(cq.getFFTLength()), imag(cq.getFFTLength());
        std::vector<double> cq_real(cq.getK()), cq_imag(cq.getK());
        fft.forward(shifted.data(), real.data(), imag.data());
        cq.process(real.data(), imag.data(), cq_real.data(), cq_imag.data());
        Json in;
        in.set("time", time);
        in.set("fft_real", real);
        in.set("fft_imag", imag);
        inputs.set("constant_q", in);
        Json out;
        out.set("bins", cq.getK());
        out.set("fft_length", cq.getFFTLength());
        out.set("hop", cq.getHop());
        out.set("q", cq.getQ());
        out.set("process_frequency", pair_json(cq_real, cq_imag));
        outputs.set("constant_q", out);
    }

    {
        ChromaConfig config{22050.0, frequency_for_midi(36), frequency_for_midi(84), 24, 0.004,
                            MathUtilities::NormaliseNone};
        Chromagram chroma_none(config);
        ChromaConfig sum_config = config;
        sum_config.normalise = MathUtilities::NormaliseUnitSum;
        Chromagram chroma_sum(sum_config);
        ChromaConfig max_config = config;
        max_config.normalise = MathUtilities::NormaliseUnitMax;
        Chromagram chroma_max(max_config);
        std::vector<double> time = sinusoid(frequency_for_midi(57), config.FS, chroma_none.getFrameSize(), 0.4);
        std::vector<double> shifted = fft_shifted_hamming(time);
        FFTReal fft(chroma_none.getFrameSize());
        std::vector<double> real(chroma_none.getFrameSize()), imag(chroma_none.getFrameSize());
        fft.forward(shifted.data(), real.data(), imag.data());
        double *none_time_ptr = chroma_none.process(time.data());
        std::vector<double> none_time(none_time_ptr, none_time_ptr + config.BPO);
        double *none_freq_ptr = chroma_none.process(real.data(), imag.data());
        std::vector<double> none_freq(none_freq_ptr, none_freq_ptr + config.BPO);
        double *sum_ptr = chroma_sum.process(time.data());
        std::vector<double> sum_time(sum_ptr, sum_ptr + config.BPO);
        double *max_ptr = chroma_max.process(time.data());
        std::vector<double> max_time(max_ptr, max_ptr + config.BPO);
        std::vector<double> unity{0.0, 2.0, 4.0, 1.0, 3.0, 6.0, 5.0, 0.5,
                                  1.5, 2.5, 3.5, 4.5, 1.25, 2.25, 3.25, 4.25,
                                  5.25, 6.25, 7.25, 8.25, 9.25, 10.25, 11.25, 12.25};
        chroma_none.unityNormalise(unity.data());
        Json in;
        in.set("time", time);
        in.set("fft_real", real);
        in.set("fft_imag", imag);
        in.set("unity_values", std::vector<double>{0.0, 2.0, 4.0, 1.0, 3.0, 6.0, 5.0, 0.5,
                                                   1.5, 2.5, 3.5, 4.5, 1.25, 2.25, 3.25, 4.25,
                                                   5.25, 6.25, 7.25, 8.25, 9.25, 10.25, 11.25, 12.25});
        inputs.set("chromagram", in);
        Json out;
        out.set("bins", config.BPO);
        out.set("cq_bins", chroma_none.getK());
        out.set("frame_size", chroma_none.getFrameSize());
        out.set("hop_size", chroma_none.getHopSize());
        out.set("time_none", none_time);
        out.set("frequency_none", none_freq);
        out.set("time_unit_sum", sum_time);
        out.set("time_unit_max", max_time);
        out.set("unity_normalise", unity);
        out.set("kabs", chroma_none.kabs(3.5, -4.25));
        outputs.set("chromagram", out);
    }

    {
        MFCCConfig config(16000);
        config.fftsize = 128;
        config.nceps = 7;
        config.logpower = 2.0;
        config.want_c0 = false;
        config.window = BlackmanWindow;
        MFCC mfcc(config);
        std::vector<double> frame = signal(config.fftsize, 1.2);
        FFTReal fft(config.fftsize);
        std::vector<double> full_real(config.fftsize), full_imag(config.fftsize);
        fft.forward(frame.data(), full_real.data(), full_imag.data());
        std::vector<double> real(full_real.begin(), full_real.begin() + config.fftsize / 2 + 1);
        std::vector<double> imag(full_imag.begin(), full_imag.begin() + config.fftsize / 2 + 1);
        std::vector<double> time_out(config.nceps);
        std::vector<double> freq_out(config.nceps);
        mfcc.process(frame.data(), time_out.data());
        mfcc.process(real.data(), imag.data(), freq_out.data());
        Json in;
        in.set("frame", frame);
        in.set("fft_real", real);
        in.set("fft_imag", imag);
        inputs.set("mfcc", in);
        Json out;
        out.set("fft_length", mfcc.getfftlength());
        out.set("output_size", config.nceps);
        out.set("time", time_out);
        out.set("frequency", freq_out);
        outputs.set("mfcc", out);
    }

    {
        MFCCConfig config(16000);
        MFCC mfcc(config);
        std::vector<double> frame = signal(config.fftsize, 0.85);
        FFTReal fft(config.fftsize);
        std::vector<double> full_real(config.fftsize), full_imag(config.fftsize);
        fft.forward(frame.data(), full_real.data(), full_imag.data());
        std::vector<double> real(full_real.begin(), full_real.begin() + config.fftsize / 2 + 1);
        std::vector<double> imag(full_imag.begin(), full_imag.begin() + config.fftsize / 2 + 1);
        int output_size = config.nceps + (config.want_c0 ? 1 : 0);
        std::vector<double> time_out(output_size);
        std::vector<double> freq_out(output_size);
        mfcc.process(frame.data(), time_out.data());
        mfcc.process(real.data(), imag.data(), freq_out.data());
        Json in;
        in.set("frame", frame);
        in.set("fft_real", real);
        in.set("fft_imag", imag);
        inputs.set("mfcc_default", in);
        Json out;
        out.set("fft_length", mfcc.getfftlength());
        out.set("output_size", output_size);
        out.set("time", time_out);
        out.set("frequency", freq_out);
        outputs.set("mfcc_default", out);
    }

    {
        const int n = 16;
        PhaseVocoder pv(n, 4);
        std::vector<double> frame0 = signal(n, 0.1);
        std::vector<double> frame1 = signal(n, 0.6);
        std::vector<double> frame2 = signal(n, 1.1);
        std::vector<double> mag(n / 2 + 1), phase(n / 2 + 1), unwrapped(n / 2 + 1);
        Json time_out;
        pv.processTimeDomain(frame0.data(), mag.data(), phase.data(), unwrapped.data());
        time_out.set("frame0", triple_json(mag, phase, unwrapped));
        pv.processTimeDomain(frame1.data(), mag.data(), phase.data(), unwrapped.data());
        time_out.set("frame1", triple_json(mag, phase, unwrapped));
        pv.reset();
        pv.processTimeDomain(frame2.data(), mag.data(), phase.data(), unwrapped.data());
        time_out.set("after_reset", triple_json(mag, phase, unwrapped));
        FFTReal fft(n);
        std::vector<double> full_real(n), full_imag(n);
        fft.forward(frame1.data(), full_real.data(), full_imag.data());
        std::vector<double> real(full_real.begin(), full_real.begin() + n / 2 + 1);
        std::vector<double> imag(full_imag.begin(), full_imag.begin() + n / 2 + 1);
        pv.processFrequencyDomain(real.data(), imag.data(), mag.data(), phase.data(), unwrapped.data());
        Json in;
        in.set("frame0", frame0);
        in.set("frame1", frame1);
        in.set("frame2", frame2);
        in.set("freq_real", real);
        in.set("freq_imag", imag);
        inputs.set("phase_vocoder", in);
        Json out;
        out.set("size", n);
        out.set("hop", 4);
        out.set("bins", n / 2 + 1);
        out.set("time", time_out);
        out.set("frequency", triple_json(mag, phase, unwrapped));
        outputs.set("phase_vocoder", out);
    }

    {
        GetKeyMode::Config config(22050.0, 442.0f);
        config.hpcpAverage = 0.2;
        config.medianAverage = 0.2;
        config.frameOverlapFactor = 2;
        config.decimationFactor = 4;
        GetKeyMode key(config);
        int block = key.getBlockSize();
        int hop = key.getHopSize();
        std::vector<double> stream = sinusoid(frequency_for_midi(64, 442.0), config.sampleRate, block + 4 * hop, 0.3);
        std::vector<double> keys;
        for (int offset = 0; offset + block <= int(stream.size()); offset += hop) {
            keys.push_back(double(key.process(stream.data() + offset)));
        }
        double *strength_ptr = key.getKeyStrengths();
        std::vector<double> strengths(strength_ptr, strength_ptr + 24);
        inputs.set("key", stream);
        Json out;
        out.set("block_size", block);
        out.set("hop_size", hop);
        out.set("keys", keys);
        out.set("strengths", strengths);
        outputs.set("key", out);
    }

    {
        std::vector<double> chroma{0.4, -0.2, 0.1, 0.0, 0.7, -0.3, 0.2, 0.5, -0.1, 0.6, 0.05, -0.45};
        TonalEstimator estimator;
        ChromaVector cv(12);
        for (size_t i = 0; i < 12; ++i) cv[i] = chroma[i];
        TCSVector tcs = estimator.transform2TCS(cv);
        std::vector<double> tcs_values = tcs_vector_to_std(tcs);
        std::vector<double> to_normalize{1.0, -2.0, 3.5, 0.0, -4.5, 6.0, 7.5, -8.0, 9.0, -10.0, 11.0, -12.0};
        TCSVector mag_vector = make_tcs_vector({3.0, -4.0, 5.0, -6.0, 7.0, -8.0});
        Json in;
        in.set("chroma", chroma);
        in.set("normalize_chroma", to_normalize);
        in.set("magnitude_tcs", std::vector<double>{3.0, -4.0, 5.0, -6.0, 7.0, -8.0});
        inputs.set("tonal", in);
        Json out;
        out.set("transform", tcs_values);
        out.set("normalize_chroma", chroma_normalize_l1(to_normalize));
        out.set("magnitude", mag_vector.magnitude());
        outputs.set("tonal", out);
        convenience.set("tonal_transform", tcs_values);
    }

    {
        std::vector<std::vector<double>> rows{
            {0.1, 0.2, -0.1, 0.3, -0.2, 0.4},
            {0.3, 0.1, 0.0, -0.2, 0.5, -0.4},
            {-0.2, 0.4, 0.2, 0.1, -0.3, 0.2},
            {0.0, -0.1, 0.6, -0.5, 0.2, 0.3},
        };
        TCSGram gram = make_tcsgram(rows, 12.5);
        Json before_clear;
        before_clear.set("size", gram.getSize());
        before_clear.set("duration", gram.getDuration());
        before_clear.set("time_2", gram.getTime(2));
        before_clear.set("vector_1", tcsgram_vector(gram, 1));
        before_clear.set("matrix", tcsgram_matrix(gram));
        gram.setFrameDuration(25.0);
        gram.addTCSVector(make_tcs_vector({0.9, -0.8, 0.7, -0.6, 0.5, -0.4}));
        Json after_setter;
        after_setter.set("size", gram.getSize());
        after_setter.set("duration", gram.getDuration());
        after_setter.set("time_4", gram.getTime(4));
        gram.clear();
        Json after_clear;
        after_clear.set("size", gram.getSize());
        after_clear.set("duration", gram.getDuration());
        inputs.set("tcsgram", rows);
        Json out;
        out.set("before_clear", before_clear);
        out.set("after_setter", after_setter);
        out.set("after_clear", after_clear);
        outputs.set("tcsgram", out);
    }

    {
        std::vector<std::vector<double>> rows{
            {0.1, 0.2, -0.1, 0.3, -0.2, 0.4},
            {0.3, 0.1, 0.0, -0.2, 0.5, -0.4},
            {-0.2, 0.4, 0.2, 0.1, -0.3, 0.2},
            {0.0, -0.1, 0.6, -0.5, 0.2, 0.3},
            {0.2, -0.3, 0.4, -0.1, 0.0, 0.1},
        };
        TCSGram gram = make_tcsgram(rows, 10.0);
        TCSGram gram_again = make_tcsgram(rows, 10.0);
        ChangeDetectionFunction change0(ChangeDFConfig{0});
        ChangeDetectionFunction change2(ChangeDFConfig{2});
        ChangeDistance distance0 = change0.process(gram);
        ChangeDistance distance2 = change2.process(gram);
        ChangeDetectionFunction change0_again(ChangeDFConfig{0});
        ChangeDetectionFunction change2_again(ChangeDFConfig{2});
        ChangeDistance distance0_again = change0_again.process(gram_again);
        ChangeDistance distance2_again = change2_again.process(gram_again);
        inputs.set("change_detection", rows);
        Json out;
        out.set("width_0_tcsgram", change_to_std(distance0));
        out.set("width_2_tcsgram", change_to_std(distance2));
        out.set("width_0_matrix", change_to_std(distance0_again));
        out.set("width_2_matrix", change_to_std(distance2_again));
        outputs.set("change_detection", out);
    }

    {
        Json names;
        Json filters;
        for (int i = int(Wavelet::Haar); i <= int(Wavelet::LastType); ++i) {
            std::string key = std::to_string(i);
            names.set(key, Wavelet::getWaveletName(static_cast<Wavelet::Type>(i)));
            filters.set(key, wavelet_output(i));
        }
        outputs.set("wavelet_names", names);
        outputs.set("wavelet_filters", filters);
    }

    outputs.set("convenience", convenience);

    Json root;
    root.set("inputs", inputs);
    root.set("outputs", outputs);
    root.save(argv[1]);
    return 0;
}
