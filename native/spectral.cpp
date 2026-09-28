#include "common.h"

#include <nanobind/stl/optional.h>

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

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

namespace {

void check_positive(int value, const char *name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
}

int positive_int(int value, const char *name) {
    check_positive(value, name);
    return value;
}

int positive_even_int(int value, const char *name) {
    check_positive(value, name);
    if (value % 2 != 0) throw std::invalid_argument(std::string(name) + " must be even");
    return value;
}

void check_non_negative(int value, const char *name) {
    if (value < 0) throw std::invalid_argument(std::string(name) + " must be non-negative");
}

int non_negative_int(int value, const char *name) {
    check_non_negative(value, name);
    return value;
}

void check_finite_positive(double value, const char *name) {
    if (!std::isfinite(value) || value <= 0.0) {
        throw std::invalid_argument(std::string(name) + " must be finite and positive");
    }
}

void check_length(const std::vector<double> &v, size_t expected, const char *name) {
    if (v.size() != expected) {
        throw std::invalid_argument(std::string(name) + " has the wrong length");
    }
}

std::vector<double> vector_input_named(const ArrayD &a, const char *name) {
    try {
        return vector_input(a);
    } catch (const std::invalid_argument &e) {
        throw std::invalid_argument(std::string(name) + ": " + e.what());
    }
}

nb::ndarray<nb::numpy, double> array2d_output(std::vector<double> values, size_t rows, size_t cols) {
    if (values.size() != rows * cols) throw std::invalid_argument("internal output shape mismatch");
    auto *v = new std::vector<double>(std::move(values));
    nb::capsule owner(v, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    return nb::ndarray<nb::numpy, double>(v->data(), {rows, cols}, owner);
}

MathUtilities::NormaliseType parse_normalise(const std::string &value) {
    if (value == "none") return MathUtilities::NormaliseNone;
    if (value == "unit_sum") return MathUtilities::NormaliseUnitSum;
    if (value == "unit_max") return MathUtilities::NormaliseUnitMax;
    throw std::invalid_argument("normalise must be 'none', 'unit_sum', or 'unit_max'");
}

WindowType parse_window(const std::string &value) {
    if (value == "rectangular") return RectangularWindow;
    if (value == "bartlett") return BartlettWindow;
    if (value == "hamming") return HammingWindow;
    if (value == "hanning" || value == "hann") return HanningWindow;
    if (value == "blackman") return BlackmanWindow;
    if (value == "blackman_harris") return BlackmanHarrisWindow;
    throw std::invalid_argument("unknown window type");
}

Wavelet::Type parse_wavelet(int value) {
    if (value < int(Wavelet::Haar) || value > int(Wavelet::LastType)) {
        throw std::invalid_argument("wavelet type out of range");
    }
    return static_cast<Wavelet::Type>(value);
}

Wavelet::Type parse_wavelet_name(const std::string &name) {
    for (int i = int(Wavelet::Haar); i <= int(Wavelet::LastType); ++i) {
        Wavelet::Type type = static_cast<Wavelet::Type>(i);
        if (Wavelet::getWaveletName(type) == name) return type;
    }
    throw std::invalid_argument("unknown wavelet name");
}

TCSGram tcsgram_from_matrix(const ArrayD &a, double frame_duration_ms) {
    if (a.ndim() != 2) throw std::invalid_argument("tcs matrix must be two-dimensional");
    if (a.shape(1) != 6) throw std::invalid_argument("tcs matrix must have shape (frames, 6)");
    check_finite_positive(frame_duration_ms, "frame_duration_ms");
    TCSGram gram;
    gram.setFrameDuration(frame_duration_ms);
    gram.setNumBins(6);
    gram.reserve(size_t(a.shape(0)));
    const double *p = a.data();
    for (size_t row = 0; row < size_t(a.shape(0)); ++row) {
        TCSVector v;
        for (size_t col = 0; col < 6; ++col) {
            double value = p[row * 6 + col];
            if (!std::isfinite(value)) throw std::invalid_argument("tcs matrix must contain finite values");
            v[col] = value;
        }
        gram.addTCSVector(v);
    }
    return gram;
}

class SpectralFFT {
public:
    explicit SpectralFFT(int size) : m_size(positive_int(size, "size")), m_fft(m_size) {}

    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>>
    process(const ArrayD &real, std::optional<ArrayD> imag, bool inverse) {
        std::vector<double> real_in = vector_input_named(real, "real");
        check_length(real_in, size_t(m_size), "real");
        std::vector<double> imag_in(size_t(m_size), 0.0);
        if (imag.has_value()) {
            imag_in = vector_input_named(*imag, "imag");
            check_length(imag_in, size_t(m_size), "imag");
        }
        std::vector<double> real_out(static_cast<size_t>(m_size));
        std::vector<double> imag_out(static_cast<size_t>(m_size));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_fft.process(inverse, real_in.data(), imag.has_value() ? imag_in.data() : nullptr,
                          real_out.data(), imag_out.data());
        }
        return {array_output(std::move(real_out)), array_output(std::move(imag_out))};
    }

    int size() const { return m_size; }

private:
    int m_size;
    FFT m_fft;
    std::mutex m_mutex;
};

class SpectralFFTReal {
public:
    explicit SpectralFFTReal(int size) : m_size(positive_even_int(size, "size")), m_fft(m_size) {}

    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>>
    forward(const ArrayD &samples) {
        std::vector<double> in = vector_input_named(samples, "samples");
        check_length(in, size_t(m_size), "samples");
        std::vector<double> real(static_cast<size_t>(m_size));
        std::vector<double> imag(static_cast<size_t>(m_size));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_fft.forward(in.data(), real.data(), imag.data());
        }
        return {array_output(std::move(real)), array_output(std::move(imag))};
    }

    nb::ndarray<nb::numpy, double> forward_magnitude(const ArrayD &samples) {
        std::vector<double> in = vector_input_named(samples, "samples");
        check_length(in, size_t(m_size), "samples");
        std::vector<double> mag(static_cast<size_t>(m_size));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_fft.forwardMagnitude(in.data(), mag.data());
        }
        return array_output(std::move(mag));
    }

    nb::ndarray<nb::numpy, double> inverse(const ArrayD &real, const ArrayD &imag) {
        std::vector<double> real_in = vector_input_named(real, "real");
        std::vector<double> imag_in = vector_input_named(imag, "imag");
        size_t bins = size_t(m_size / 2 + 1);
        check_length(real_in, bins, "real");
        check_length(imag_in, bins, "imag");
        std::vector<double> out(static_cast<size_t>(m_size));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_fft.inverse(real_in.data(), imag_in.data(), out.data());
        }
        return array_output(std::move(out));
    }

    int size() const { return m_size; }
    int bins() const { return m_size / 2 + 1; }

private:
    int m_size;
    FFTReal m_fft;
    std::mutex m_mutex;
};

class SpectralDCT {
public:
    explicit SpectralDCT(int size) : m_size(positive_int(size, "size")), m_dct(m_size) {}

    nb::ndarray<nb::numpy, double> forward(const ArrayD &samples, bool unitary) {
        return process(samples, unitary ? Mode::ForwardUnitary : Mode::Forward);
    }

    nb::ndarray<nb::numpy, double> inverse(const ArrayD &coefficients, bool unitary) {
        return process(coefficients, unitary ? Mode::InverseUnitary : Mode::Inverse);
    }

    int size() const { return m_size; }

private:
    enum class Mode { Forward, ForwardUnitary, Inverse, InverseUnitary };

    nb::ndarray<nb::numpy, double> process(const ArrayD &input, Mode mode) {
        std::vector<double> in = vector_input_named(input, "input");
        check_length(in, size_t(m_size), "input");
        std::vector<double> out(static_cast<size_t>(m_size));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            if (mode == Mode::Forward) m_dct.forward(in.data(), out.data());
            else if (mode == Mode::ForwardUnitary) m_dct.forwardUnitary(in.data(), out.data());
            else if (mode == Mode::Inverse) m_dct.inverse(in.data(), out.data());
            else m_dct.inverseUnitary(in.data(), out.data());
        }
        return array_output(std::move(out));
    }

    int m_size;
    DCT m_dct;
    std::mutex m_mutex;
};

class SpectralConstantQ {
public:
    SpectralConstantQ(double sample_rate, double min_frequency, double max_frequency,
                      int bins_per_octave, double threshold)
        : m_config(make_config(sample_rate, min_frequency, max_frequency, bins_per_octave, threshold)),
          m_cq(m_config) {
        nb::gil_scoped_release release;
        m_cq.sparsekernel();
    }

    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>>
    process_frequency(const ArrayD &real, const ArrayD &imag) {
        std::vector<double> real_in = vector_input_named(real, "real");
        std::vector<double> imag_in = vector_input_named(imag, "imag");
        check_length(real_in, size_t(m_cq.getFFTLength()), "real");
        check_length(imag_in, size_t(m_cq.getFFTLength()), "imag");
        std::vector<double> real_out(size_t(m_cq.getK()));
        std::vector<double> imag_out(size_t(m_cq.getK()));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_cq.process(real_in.data(), imag_in.data(), real_out.data(), imag_out.data());
        }
        return {array_output(std::move(real_out)), array_output(std::move(imag_out))};
    }

    int bins() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_cq.getK();
    }
    int fft_length() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_cq.getFFTLength();
    }
    int hop() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_cq.getHop();
    }
    double q() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_cq.getQ();
    }

private:
    static CQConfig make_config(double sample_rate, double min_frequency, double max_frequency,
                                int bins_per_octave, double threshold) {
        check_finite_positive(sample_rate, "sample_rate");
        check_finite_positive(min_frequency, "min_frequency");
        check_finite_positive(max_frequency, "max_frequency");
        if (max_frequency <= min_frequency) {
            throw std::invalid_argument("max_frequency must be greater than min_frequency");
        }
        check_positive(bins_per_octave, "bins_per_octave");
        if (!std::isfinite(threshold) || threshold < 0.0) {
            throw std::invalid_argument("threshold must be finite and non-negative");
        }
        return {sample_rate, min_frequency, max_frequency, bins_per_octave, threshold};
    }

    CQConfig m_config;
    ConstantQ m_cq;
    std::mutex m_mutex;
};

class SpectralChromagram {
public:
    SpectralChromagram(double sample_rate, double min_frequency, double max_frequency,
                       int bins_per_octave, double threshold, const std::string &normalise)
        : m_config(make_config(sample_rate, min_frequency, max_frequency, bins_per_octave,
                               threshold, normalise)),
          m_chroma(m_config) {}

    nb::ndarray<nb::numpy, double> process_time(const ArrayD &frame) {
        std::vector<double> in = vector_input_named(frame, "frame");
        check_length(in, size_t(m_chroma.getFrameSize()), "frame");
        std::vector<double> out(size_t(m_config.BPO));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            double *ptr = m_chroma.process(in.data());
            std::copy(ptr, ptr + m_config.BPO, out.begin());
        }
        return array_output(std::move(out));
    }

    nb::ndarray<nb::numpy, double> process_frequency(const ArrayD &real, const ArrayD &imag) {
        std::vector<double> real_in = vector_input_named(real, "real");
        std::vector<double> imag_in = vector_input_named(imag, "imag");
        check_length(real_in, size_t(m_chroma.getFrameSize()), "real");
        check_length(imag_in, size_t(m_chroma.getFrameSize()), "imag");
        std::vector<double> out(size_t(m_config.BPO));
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            double *ptr = m_chroma.process(real_in.data(), imag_in.data());
            std::copy(ptr, ptr + m_config.BPO, out.begin());
        }
        return array_output(std::move(out));
    }

    nb::ndarray<nb::numpy, double> unity_normalise(const ArrayD &values) {
        std::vector<double> out = vector_input_named(values, "values");
        check_length(out, size_t(m_config.BPO), "values");
        double max_value = *std::max_element(out.begin(), out.end());
        if (max_value == 0.0) {
            throw std::invalid_argument("values must have a non-zero maximum");
        }
        {
            std::lock_guard<std::mutex> guard(m_mutex);
            m_chroma.unityNormalise(out.data());
        }
        return array_output(std::move(out));
    }

    double kabs(double real, double imag) {
        if (!std::isfinite(real) || !std::isfinite(imag)) {
            throw std::invalid_argument("real and imag must be finite");
        }
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_chroma.kabs(real, imag);
    }

    int bins() const { return m_config.BPO; }
    int cq_bins() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_chroma.getK();
    }
    int frame_size() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_chroma.getFrameSize();
    }
    int hop_size() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_chroma.getHopSize();
    }

private:
    static ChromaConfig make_config(double sample_rate, double min_frequency, double max_frequency,
                                    int bins_per_octave, double threshold,
                                    const std::string &normalise) {
        check_finite_positive(sample_rate, "sample_rate");
        check_finite_positive(min_frequency, "min_frequency");
        check_finite_positive(max_frequency, "max_frequency");
        if (max_frequency <= min_frequency) {
            throw std::invalid_argument("max_frequency must be greater than min_frequency");
        }
        check_positive(bins_per_octave, "bins_per_octave");
        if (!std::isfinite(threshold) || threshold < 0.0) {
            throw std::invalid_argument("threshold must be finite and non-negative");
        }
        return {sample_rate, min_frequency, max_frequency, bins_per_octave, threshold,
                parse_normalise(normalise)};
    }

    ChromaConfig m_config;
    Chromagram m_chroma;
    std::mutex m_mutex;
};

class SpectralMFCC {
public:
    SpectralMFCC(int sample_rate, int fft_size, int n_coefficients, double log_power,
                 bool want_c0, const std::string &window)
        : m_config(sample_rate), m_mfcc(make_config(sample_rate, fft_size, n_coefficients,
                                                    log_power, want_c0, window)) {
        m_config = make_config(sample_rate, fft_size, n_coefficients, log_power, want_c0, window);
    }

    nb::ndarray<nb::numpy, double> process_time(const ArrayD &frame) {
        std::vector<double> in = vector_input_named(frame, "frame");
        check_length(in, size_t(m_config.fftsize), "frame");
        std::vector<double> out(output_size());
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_mfcc.process(in.data(), out.data());
        }
        return array_output(std::move(out));
    }

    nb::ndarray<nb::numpy, double> process_frequency(const ArrayD &real, const ArrayD &imag) {
        std::vector<double> real_in = vector_input_named(real, "real");
        std::vector<double> imag_in = vector_input_named(imag, "imag");
        size_t bins = size_t(m_config.fftsize / 2 + 1);
        check_length(real_in, bins, "real");
        check_length(imag_in, bins, "imag");
        std::vector<double> out(output_size());
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            m_mfcc.process(real_in.data(), imag_in.data(), out.data());
        }
        return array_output(std::move(out));
    }

    int fft_length() const { return m_mfcc.getfftlength(); }
    int output_size() const { return m_config.nceps + (m_config.want_c0 ? 1 : 0); }

private:
    static MFCCConfig make_config(int sample_rate, int fft_size, int n_coefficients,
                                  double log_power, bool want_c0, const std::string &window) {
        check_positive(sample_rate, "sample_rate");
        check_positive(fft_size, "fft_size");
        if (fft_size % 2 != 0) throw std::invalid_argument("fft_size must be even");
        check_positive(n_coefficients, "n_coefficients");
        if (!std::isfinite(log_power) || log_power <= 0.0) {
            throw std::invalid_argument("log_power must be finite and positive");
        }
        MFCCConfig config(sample_rate);
        config.fftsize = fft_size;
        config.nceps = n_coefficients;
        config.logpower = log_power;
        config.want_c0 = want_c0;
        config.window = parse_window(window);
        return config;
    }

    MFCCConfig m_config;
    MFCC m_mfcc;
    std::mutex m_mutex;
};

class SpectralPhaseVocoder {
public:
    SpectralPhaseVocoder(int size, int hop)
        : m_size(positive_even_int(size, "size")), m_hop(positive_int(hop, "hop")),
          m_pv(m_size, m_hop) {}

    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>,
               nb::ndarray<nb::numpy, double>>
    process_time(const ArrayD &frame) {
        std::vector<double> in = vector_input_named(frame, "frame");
        check_length(in, size_t(m_size), "frame");
        return process([&](double *mag, double *phase, double *unwrapped) {
            m_pv.processTimeDomain(in.data(), mag, phase, unwrapped);
        });
    }

    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>,
               nb::ndarray<nb::numpy, double>>
    process_frequency(const ArrayD &real, const ArrayD &imag) {
        std::vector<double> real_in = vector_input_named(real, "real");
        std::vector<double> imag_in = vector_input_named(imag, "imag");
        size_t bins = size_t(m_size / 2 + 1);
        check_length(real_in, bins, "real");
        check_length(imag_in, bins, "imag");
        return process([&](double *mag, double *phase, double *unwrapped) {
            m_pv.processFrequencyDomain(real_in.data(), imag_in.data(), mag, phase, unwrapped);
        });
    }

    void reset() {
        nb::gil_scoped_release release;
        std::lock_guard<std::mutex> guard(m_mutex);
        m_pv.reset();
    }

    int size() const { return m_size; }
    int hop() const { return m_hop; }
    int bins() const { return m_size / 2 + 1; }

private:
    template <typename Fn>
    std::tuple<nb::ndarray<nb::numpy, double>, nb::ndarray<nb::numpy, double>,
               nb::ndarray<nb::numpy, double>>
    process(Fn fn) {
        size_t bins = size_t(m_size / 2 + 1);
        std::vector<double> mag(bins);
        std::vector<double> phase(bins);
        std::vector<double> unwrapped(bins);
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            fn(mag.data(), phase.data(), unwrapped.data());
        }
        return {array_output(std::move(mag)), array_output(std::move(phase)),
                array_output(std::move(unwrapped))};
    }

    int m_size;
    int m_hop;
    PhaseVocoder m_pv;
    std::mutex m_mutex;
};

class SpectralKeyMode {
public:
    SpectralKeyMode(double sample_rate, float tuning_frequency, double hpcp_average,
                    double median_average, int frame_overlap_factor, int decimation_factor)
        : m_config(make_config(sample_rate, tuning_frequency, hpcp_average, median_average,
                               frame_overlap_factor, decimation_factor)),
          m_key(m_config) {}

    int process(const ArrayD &frame) {
        std::vector<double> in = vector_input_named(frame, "frame");
        check_length(in, size_t(m_key.getBlockSize()), "frame");
        int result = 0;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            result = m_key.process(in.data());
            m_processed = true;
        }
        return result;
    }

    nb::ndarray<nb::numpy, double> key_strengths() {
        std::vector<double> out(24);
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            if (!m_processed) {
                throw std::invalid_argument("key strengths are unavailable before process()");
            }
            double *ptr = m_key.getKeyStrengths();
            std::copy(ptr, ptr + 24, out.begin());
        }
        return array_output(std::move(out));
    }

    int block_size() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_key.getBlockSize();
    }
    int hop_size() {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_key.getHopSize();
    }

private:
    static GetKeyMode::Config make_config(double sample_rate, float tuning_frequency,
                                          double hpcp_average, double median_average,
                                          int frame_overlap_factor, int decimation_factor) {
        check_finite_positive(sample_rate, "sample_rate");
        if (!std::isfinite(tuning_frequency) || tuning_frequency <= 0.0f) {
            throw std::invalid_argument("tuning_frequency must be finite and positive");
        }
        check_finite_positive(hpcp_average, "hpcp_average");
        check_finite_positive(median_average, "median_average");
        check_positive(frame_overlap_factor, "frame_overlap_factor");
        check_positive(decimation_factor, "decimation_factor");
        GetKeyMode::Config config(sample_rate, tuning_frequency);
        config.hpcpAverage = hpcp_average;
        config.medianAverage = median_average;
        config.frameOverlapFactor = frame_overlap_factor;
        config.decimationFactor = decimation_factor;
        return config;
    }

    GetKeyMode::Config m_config;
    GetKeyMode m_key;
    bool m_processed = false;
    std::mutex m_mutex;
};

class SpectralTonalEstimator {
public:
    nb::ndarray<nb::numpy, double> transform_chroma(const ArrayD &chroma) {
        std::vector<double> in = vector_input_named(chroma, "chroma");
        check_length(in, 12, "chroma");
        ChromaVector chroma_vector(12);
        for (size_t i = 0; i < 12; ++i) chroma_vector[i] = in[i];
        TCSVector tcs;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            tcs = m_estimator.transform2TCS(chroma_vector);
        }
        std::vector<double> out(6);
        for (size_t i = 0; i < 6; ++i) out[i] = tcs[i];
        return array_output(std::move(out));
    }

private:
    TonalEstimator m_estimator;
    std::mutex m_mutex;
};

class SpectralTCSGram {
public:
    SpectralTCSGram(double frame_duration_ms, size_t reserve) : m_frame_duration_ms(frame_duration_ms) {
        check_finite_positive(frame_duration_ms, "frame_duration_ms");
        m_gram.setFrameDuration(frame_duration_ms);
        m_gram.setNumBins(6);
        m_gram.reserve(reserve);
    }

    void add(const ArrayD &tcs) {
        std::vector<double> in = vector_input_named(tcs, "tcs");
        check_length(in, 6, "tcs");
        TCSVector vector;
        for (size_t i = 0; i < 6; ++i) vector[i] = in[i];
        std::lock_guard<std::mutex> guard(m_mutex);
        m_gram.addTCSVector(vector);
    }

    nb::ndarray<nb::numpy, double> vector_at(int index) const {
        std::lock_guard<std::mutex> guard(m_mutex);
        TCSVector vector;
        m_gram.getTCSVector(index, vector);
        std::vector<double> out(6);
        for (size_t i = 0; i < 6; ++i) out[i] = vector[i];
        return array_output(std::move(out));
    }

    nb::ndarray<nb::numpy, double> matrix() const {
        std::lock_guard<std::mutex> guard(m_mutex);
        std::vector<double> out(size_t(m_gram.getSize()) * 6);
        for (int row = 0; row < m_gram.getSize(); ++row) {
            TCSVector vector;
            m_gram.getTCSVector(row, vector);
            for (size_t col = 0; col < 6; ++col) out[size_t(row) * 6 + col] = vector[col];
        }
        return array2d_output(std::move(out), size_t(m_gram.getSize()), 6);
    }

    void clear() {
        std::lock_guard<std::mutex> guard(m_mutex);
        m_gram.clear();
    }

    void set_frame_duration_ms(double frame_duration_ms) {
        check_finite_positive(frame_duration_ms, "frame_duration_ms");
        std::lock_guard<std::mutex> guard(m_mutex);
        m_frame_duration_ms = frame_duration_ms;
        m_gram.setFrameDuration(frame_duration_ms);
    }

    int size() const {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_gram.getSize();
    }
    long time_at(size_t index) const {
        std::lock_guard<std::mutex> guard(m_mutex);
        if (index >= size_t(m_gram.getSize())) throw std::invalid_argument("index out of range");
        return m_gram.getTime(index);
    }
    long duration() const {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_gram.getDuration();
    }
    double frame_duration_ms() const { return m_frame_duration_ms; }
    TCSGram snapshot() const {
        std::lock_guard<std::mutex> guard(m_mutex);
        return m_gram;
    }

private:
    double m_frame_duration_ms;
    TCSGram m_gram;
    mutable std::mutex m_mutex;
};

class SpectralChangeDetection {
public:
    explicit SpectralChangeDetection(int smoothing_width)
        : m_config{non_negative_int(smoothing_width, "smoothing_width")}, m_change(m_config) {}

    nb::ndarray<nb::numpy, double> process_matrix(const ArrayD &matrix, double frame_duration_ms) {
        TCSGram gram = tcsgram_from_matrix(matrix, frame_duration_ms);
        return process_gram(gram);
    }

    nb::ndarray<nb::numpy, double> process_tcsgram(const SpectralTCSGram &gram) {
        TCSGram snapshot = gram.snapshot();
        return process_gram(snapshot, false);
    }

private:
    nb::ndarray<nb::numpy, double> process_gram(const TCSGram &gram, bool release_gil = true) {
        ChangeDistance distance;
        if (release_gil) {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> guard(m_mutex);
            distance = m_change.process(gram);
        } else {
            std::lock_guard<std::mutex> guard(m_mutex);
            distance = m_change.process(gram);
        }
        std::vector<double> out(distance.size());
        for (size_t i = 0; i < distance.size(); ++i) out[i] = distance[i];
        return array_output(std::move(out));
    }

    ChangeDFConfig m_config;
    ChangeDetectionFunction m_change;
    std::mutex m_mutex;
};

nb::ndarray<nb::numpy, double> spectral_tonal_transform(const ArrayD &chroma) {
    SpectralTonalEstimator estimator;
    return estimator.transform_chroma(chroma);
}

nb::ndarray<nb::numpy, double> spectral_normalize_chroma(const ArrayD &chroma) {
    std::vector<double> in = vector_input_named(chroma, "chroma");
    check_length(in, 12, "chroma");
    ChromaVector vector(12);
    for (size_t i = 0; i < 12; ++i) vector[i] = in[i];
    {
        nb::gil_scoped_release release;
        vector.normalizeL1();
    }
    std::vector<double> out(12);
    for (size_t i = 0; i < 12; ++i) out[i] = vector[i];
    return array_output(std::move(out));
}

double spectral_tonal_magnitude(const ArrayD &tcs) {
    std::vector<double> in = vector_input_named(tcs, "tcs");
    check_length(in, 6, "tcs");
    TCSVector vector;
    for (size_t i = 0; i < 6; ++i) vector[i] = in[i];
    nb::gil_scoped_release release;
    return vector.magnitude();
}

nb::tuple spectral_wavelet_filters(nb::object wavelet) {
    Wavelet::Type type;
    if (nb::isinstance<nb::int_>(wavelet)) {
        type = parse_wavelet(nb::cast<int>(wavelet));
    } else {
        type = parse_wavelet_name(nb::cast<std::string>(wavelet));
    }
    std::vector<double> low;
    std::vector<double> high;
    {
        nb::gil_scoped_release release;
        Wavelet::createDecompositionFilters(type, low, high);
    }
    return nb::make_tuple(array_output(std::move(low)), array_output(std::move(high)));
}

nb::list spectral_wavelet_names() {
    nb::list names;
    for (int i = int(Wavelet::Haar); i <= int(Wavelet::LastType); ++i) {
        names.append(Wavelet::getWaveletName(static_cast<Wavelet::Type>(i)));
    }
    return names;
}

}  // namespace

void bind_spectral(nb::module_ &m) {
    nb::class_<SpectralFFT>(m, "SpectralFFT")
        .def(nb::init<int>(), "size"_a)
        .def_prop_ro("size", &SpectralFFT::size)
        .def("process", &SpectralFFT::process, "real"_a, "imag"_a = std::nullopt,
             "inverse"_a = false);

    nb::class_<SpectralFFTReal>(m, "SpectralFFTReal")
        .def(nb::init<int>(), "size"_a)
        .def_prop_ro("size", &SpectralFFTReal::size)
        .def_prop_ro("bins", &SpectralFFTReal::bins)
        .def("forward", &SpectralFFTReal::forward, "samples"_a)
        .def("forward_magnitude", &SpectralFFTReal::forward_magnitude, "samples"_a)
        .def("inverse", &SpectralFFTReal::inverse, "real"_a, "imag"_a);

    nb::class_<SpectralDCT>(m, "SpectralDCT")
        .def(nb::init<int>(), "size"_a)
        .def_prop_ro("size", &SpectralDCT::size)
        .def("forward", &SpectralDCT::forward, "samples"_a, "unitary"_a = false)
        .def("inverse", &SpectralDCT::inverse, "coefficients"_a, "unitary"_a = false);

    nb::class_<SpectralConstantQ>(m, "SpectralConstantQ")
        .def(nb::init<double, double, double, int, double>(), "sample_rate"_a,
             "min_frequency"_a, "max_frequency"_a, "bins_per_octave"_a = 12,
             "threshold"_a = 0.0054)
        .def_prop_ro("bins", &SpectralConstantQ::bins)
        .def_prop_ro("fft_length", &SpectralConstantQ::fft_length)
        .def_prop_ro("hop", &SpectralConstantQ::hop)
        .def_prop_ro("q", &SpectralConstantQ::q)
        .def("process_frequency", &SpectralConstantQ::process_frequency, "real"_a, "imag"_a);

    nb::class_<SpectralChromagram>(m, "SpectralChromagram")
        .def(nb::init<double, double, double, int, double, const std::string &>(),
             "sample_rate"_a, "min_frequency"_a, "max_frequency"_a,
             "bins_per_octave"_a = 12, "threshold"_a = 0.0054,
             "normalise"_a = "unit_max")
        .def_prop_ro("bins", &SpectralChromagram::bins)
        .def_prop_ro("cq_bins", &SpectralChromagram::cq_bins)
        .def_prop_ro("frame_size", &SpectralChromagram::frame_size)
        .def_prop_ro("hop_size", &SpectralChromagram::hop_size)
        .def("process_time", &SpectralChromagram::process_time, "frame"_a)
        .def("process_frequency", &SpectralChromagram::process_frequency, "real"_a, "imag"_a)
        .def("unity_normalise", &SpectralChromagram::unity_normalise, "values"_a)
        .def("kabs", &SpectralChromagram::kabs, "real"_a, "imag"_a);

    nb::class_<SpectralMFCC>(m, "SpectralMFCC")
        .def(nb::init<int, int, int, double, bool, const std::string &>(), "sample_rate"_a,
             "fft_size"_a = 2048, "n_coefficients"_a = 19, "log_power"_a = 1.0,
             "want_c0"_a = true, "window"_a = "hamming")
        .def_prop_ro("fft_length", &SpectralMFCC::fft_length)
        .def_prop_ro("output_size", &SpectralMFCC::output_size)
        .def("process_time", &SpectralMFCC::process_time, "frame"_a)
        .def("process_frequency", &SpectralMFCC::process_frequency, "real"_a, "imag"_a);

    nb::class_<SpectralPhaseVocoder>(m, "SpectralPhaseVocoder")
        .def(nb::init<int, int>(), "size"_a, "hop"_a)
        .def_prop_ro("size", &SpectralPhaseVocoder::size)
        .def_prop_ro("hop", &SpectralPhaseVocoder::hop)
        .def_prop_ro("bins", &SpectralPhaseVocoder::bins)
        .def("process_time", &SpectralPhaseVocoder::process_time, "frame"_a)
        .def("process_frequency", &SpectralPhaseVocoder::process_frequency, "real"_a, "imag"_a)
        .def("reset", &SpectralPhaseVocoder::reset);

    nb::class_<SpectralKeyMode>(m, "SpectralKeyMode")
        .def(nb::init<double, float, double, double, int, int>(), "sample_rate"_a,
             "tuning_frequency"_a = 440.0f, "hpcp_average"_a = 10.0,
             "median_average"_a = 10.0, "frame_overlap_factor"_a = 1,
             "decimation_factor"_a = 8)
        .def_prop_ro("block_size", &SpectralKeyMode::block_size)
        .def_prop_ro("hop_size", &SpectralKeyMode::hop_size)
        .def("process", &SpectralKeyMode::process, "frame"_a)
        .def("key_strengths", &SpectralKeyMode::key_strengths);

    nb::class_<SpectralTonalEstimator>(m, "SpectralTonalEstimator")
        .def(nb::init<>())
        .def("transform_chroma", &SpectralTonalEstimator::transform_chroma, "chroma"_a);

    nb::class_<SpectralTCSGram>(m, "SpectralTCSGram")
        .def(nb::init<double, size_t>(), "frame_duration_ms"_a, "reserve"_a = 0)
        .def_prop_ro("frame_duration_ms", &SpectralTCSGram::frame_duration_ms)
        .def_prop_ro("size", &SpectralTCSGram::size)
        .def_prop_ro("duration", &SpectralTCSGram::duration)
        .def("add", &SpectralTCSGram::add, "tcs"_a)
        .def("vector_at", &SpectralTCSGram::vector_at, "index"_a)
        .def("matrix", &SpectralTCSGram::matrix)
        .def("time_at", &SpectralTCSGram::time_at, "index"_a)
        .def("set_frame_duration_ms", &SpectralTCSGram::set_frame_duration_ms,
             "frame_duration_ms"_a)
        .def("clear", &SpectralTCSGram::clear);

    nb::class_<SpectralChangeDetection>(m, "SpectralChangeDetection")
        .def(nb::init<int>(), "smoothing_width"_a)
        .def("process_matrix", &SpectralChangeDetection::process_matrix, "matrix"_a,
             "frame_duration_ms"_a = 10.0)
        .def("process_tcsgram", &SpectralChangeDetection::process_tcsgram, "gram"_a);

    m.def("spectral_tonal_transform", &spectral_tonal_transform, "chroma"_a);
    m.def("spectral_normalize_chroma", &spectral_normalize_chroma, "chroma"_a);
    m.def("spectral_tonal_magnitude", &spectral_tonal_magnitude, "tcs"_a);
    m.def("spectral_wavelet_filters", &spectral_wavelet_filters, "wavelet"_a);
    m.def("spectral_wavelet_names", &spectral_wavelet_names);
}
