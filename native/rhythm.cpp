// Rhythm, onset, and tempo bindings over qm-dsp public classes.
// Reads: native/common.h, extern/qm-dsp dsp/onsets, dsp/tempotracking, dsp/rhythm.
#include "common.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <mutex>
#include <numeric>

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

void require_positive_int(int value, const char *name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
}

void require_finite(double value, const char *name) {
    if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " must be finite");
}

void require_coefficients(const std::vector<double> &a, const std::vector<double> &b, int order) {
    if (order < 0) throw std::invalid_argument("lp_order must be non-negative");
    if (a.size() != static_cast<std::size_t>(order + 1) ||
        b.size() != static_cast<std::size_t>(order + 1)) {
        throw std::invalid_argument("filter coefficient arrays must have lp_order + 1 values");
    }
    for (double x : a) require_finite(x, "LPACoeffs");
    for (double x : b) require_finite(x, "LPBCoeffs");
    if (a.empty() || a[0] == 0.0) {
        throw std::invalid_argument("LPACoeffs[0] must be non-zero");
    }
}

std::vector<double> vector_input_exact(const ArrayD &a, std::size_t length, const char *name) {
    std::vector<double> v = vector_input(a);
    if (v.size() != length) {
        throw std::invalid_argument(std::string(name) + " has the wrong length");
    }
    return v;
}

std::vector<std::vector<double>> matrix_input(const ArrayD &a) {
    if (a.ndim() != 2) throw std::invalid_argument("expected a two-dimensional array");
    if (a.shape(0) < 2 || a.shape(1) < 1) {
        throw std::invalid_argument("matrix must have at least two rows and one column");
    }
    std::vector<std::vector<double>> rows(static_cast<std::size_t>(a.shape(0)));
    const double *data = a.data();
    for (std::size_t r = 0; r < static_cast<std::size_t>(a.shape(0)); ++r) {
        rows[r].resize(static_cast<std::size_t>(a.shape(1)));
        for (std::size_t c = 0; c < static_cast<std::size_t>(a.shape(1)); ++c) {
            double value = data[r * static_cast<std::size_t>(a.shape(1)) + c];
            if (!std::isfinite(value)) throw std::invalid_argument("matrix must contain finite values");
            rows[r][c] = value;
        }
    }
    return rows;
}

nb::ndarray<nb::numpy, int> int_array_output(std::vector<int> values) {
    auto *v = new std::vector<int>(std::move(values));
    nb::capsule owner(v, [](void *p) noexcept { delete static_cast<std::vector<int> *>(p); });
    return nb::ndarray<nb::numpy, int>(v->data(), {v->size()}, owner);
}

DFConfig make_df_config(int step_size, int frame_length, int df_type, double db_rise,
                        bool adaptive_whitening, double whitening_relax_coeff,
                        double whitening_floor) {
    require_positive_int(step_size, "step_size");
    require_positive_int(frame_length, "frame_length");
    if (frame_length % 2 != 0) throw std::invalid_argument("frame_length must be even");
    if (df_type < DF_HFC || df_type > DF_BROADBAND) {
        throw std::invalid_argument("df_type must be one of 1..5");
    }
    require_finite(db_rise, "db_rise");
    require_finite(whitening_relax_coeff, "whitening_relax_coeff");
    require_finite(whitening_floor, "whitening_floor");

    DFConfig config;
    config.stepSize = step_size;
    config.frameLength = frame_length;
    config.DFType = df_type;
    config.dbRise = db_rise;
    config.adaptiveWhitening = adaptive_whitening;
    config.whiteningRelaxCoeff = whitening_relax_coeff;
    config.whiteningFloor = whitening_floor;
    return config;
}

class RhythmDetectionFunction {
public:
    RhythmDetectionFunction(int step_size, int frame_length, int df_type, double db_rise,
                            bool adaptive_whitening, double whitening_relax_coeff,
                            double whitening_floor)
        : config_(make_df_config(step_size, frame_length, df_type, db_rise,
                                 adaptive_whitening, whitening_relax_coeff, whitening_floor)),
          impl_(config_) {}

    double process_time_domain(const ArrayD &samples) {
        std::vector<double> frame =
            vector_input_exact(samples, static_cast<std::size_t>(config_.frameLength), "samples");
        double value = 0.0;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            value = impl_.processTimeDomain(frame.data());
            has_processed_ = true;
        }
        return value;
    }

    double process_frequency_domain(const ArrayD &reals, const ArrayD &imags) {
        std::size_t half = static_cast<std::size_t>(config_.frameLength / 2 + 1);
        std::vector<double> real = vector_input_exact(reals, half, "reals");
        std::vector<double> imag = vector_input_exact(imags, half, "imags");
        double value = 0.0;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            value = impl_.processFrequencyDomain(real.data(), imag.data());
            has_processed_ = true;
        }
        return value;
    }

    nb::ndarray<nb::numpy, double> spectrum_magnitude() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!has_processed_) {
            throw std::logic_error("spectrum magnitude is available after processing one frame");
        }
        std::size_t half = static_cast<std::size_t>(config_.frameLength / 2 + 1);
        double *src = impl_.getSpectrumMagnitude();
        return array_output(std::vector<double>(src, src + half));
    }

    nb::dict params() const {
        nb::dict d;
        d["step_size"] = config_.stepSize;
        d["frame_length"] = config_.frameLength;
        d["df_type"] = config_.DFType;
        d["db_rise"] = config_.dbRise;
        d["adaptive_whitening"] = config_.adaptiveWhitening;
        d["whitening_relax_coeff"] = config_.whiteningRelaxCoeff;
        d["whitening_floor"] = config_.whiteningFloor;
        return d;
    }

private:
    DFConfig config_;
    DetectionFunction impl_;
    std::mutex mutex_;
    bool has_processed_ = false;
};

class RhythmPeakPicking {
public:
    RhythmPeakPicking(int length, double tau, int alpha, double cutoff, int lp_order,
                      std::vector<double> lp_a, std::vector<double> lp_b, int win_pre,
                      int win_post, double quad_a, double quad_b, double quad_c,
                      double delta)
        : length_(length),
          tau_(tau),
          alpha_(alpha),
          cutoff_(cutoff),
          lp_order_(lp_order),
          lp_a_(std::move(lp_a)),
          lp_b_(std::move(lp_b)),
          win_pre_(win_pre),
          win_post_(win_post),
          quad_a_(quad_a),
          quad_b_(quad_b),
          quad_c_(quad_c),
          delta_(delta) {
        require_positive_int(length_, "length");
        require_finite(tau_, "tau");
        require_finite(cutoff_, "cutoff");
        require_positive_int(alpha_, "alpha");
        if (win_pre_ < 0 || win_post_ < 0) {
            throw std::invalid_argument("window sizes must be non-negative");
        }
        require_finite(quad_a_, "quad_a");
        require_finite(quad_b_, "quad_b");
        require_finite(quad_c_, "quad_c");
        require_finite(delta_, "delta");
        require_coefficients(lp_a_, lp_b_, lp_order_);

        PPickParams config;
        config.length = length_;
        config.tau = tau_;
        config.alpha = alpha_;
        config.cutoff = cutoff_;
        config.LPOrd = lp_order_;
        config.LPACoeffs = lp_a_.data();
        config.LPBCoeffs = lp_b_.data();
        config.WinT = PPWinThresh(win_pre_, win_post_);
        config.QuadThresh = QFitThresh(quad_a_, quad_b_, quad_c_);
        config.delta = static_cast<float>(delta_);
        impl_ = std::make_unique<PeakPicking>(config);
    }

    nb::tuple process(const ArrayD &df) {
        std::vector<double> work =
            vector_input_exact(df, static_cast<std::size_t>(length_), "df");
        std::vector<int> onsets;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            impl_->process(work.data(), length_, onsets);
        }
        return nb::make_tuple(int_array_output(std::move(onsets)), array_output(std::move(work)));
    }

    nb::dict params() const {
        nb::dict d;
        d["length"] = length_;
        d["tau"] = tau_;
        d["alpha"] = alpha_;
        d["cutoff"] = cutoff_;
        d["lp_order"] = lp_order_;
        d["lp_a"] = array_output(lp_a_);
        d["lp_b"] = array_output(lp_b_);
        d["win_pre"] = win_pre_;
        d["win_post"] = win_post_;
        d["quad_a"] = quad_a_;
        d["quad_b"] = quad_b_;
        d["quad_c"] = quad_c_;
        d["delta"] = delta_;
        return d;
    }

private:
    int length_;
    double tau_;
    int alpha_;
    double cutoff_;
    int lp_order_;
    std::vector<double> lp_a_;
    std::vector<double> lp_b_;
    int win_pre_;
    int win_post_;
    double quad_a_;
    double quad_b_;
    double quad_c_;
    double delta_;
    std::unique_ptr<PeakPicking> impl_;
    std::mutex mutex_;
};

class RhythmTempoTrack {
public:
    RhythmTempoTrack(int win_length, int lag_length, int alpha, int lp_order,
                     std::vector<double> lp_a, std::vector<double> lp_b, int win_pre,
                     int win_post)
        : win_length_(win_length),
          lag_length_(lag_length),
          alpha_(alpha),
          lp_order_(lp_order),
          lp_a_(std::move(lp_a)),
          lp_b_(std::move(lp_b)),
          win_pre_(win_pre),
          win_post_(win_post) {
        require_positive_int(win_length_, "win_length");
        require_positive_int(lag_length_, "lag_length");
        require_positive_int(alpha_, "alpha");
        if (win_length_ < 4 * lag_length_) {
            throw std::invalid_argument("win_length must be at least 4 * lag_length");
        }
        if (win_pre_ < 0 || win_post_ < 0) {
            throw std::invalid_argument("window sizes must be non-negative");
        }
        require_coefficients(lp_a_, lp_b_, lp_order_);

        WinThresh win;
        win.pre = win_pre_;
        win.post = win_post_;
        TTParams params;
        params.winLength = win_length_;
        params.lagLength = lag_length_;
        params.alpha = alpha_;
        params.LPOrd = lp_order_;
        params.LPACoeffs = lp_a_.data();
        params.LPBCoeffs = lp_b_.data();
        params.WinT = win;
        impl_ = std::make_unique<TempoTrack>(params);
    }

    nb::tuple process(const ArrayD &df) {
        std::vector<double> values = vector_input(df);
        if (values.size() < static_cast<std::size_t>(win_length_)) {
            throw std::invalid_argument("df must contain at least win_length samples");
        }
        std::vector<double> tempo;
        std::vector<int> beats;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            beats = impl_->process(values, &tempo);
        }
        return nb::make_tuple(int_array_output(std::move(beats)), array_output(std::move(tempo)));
    }

    nb::dict params() const {
        nb::dict d;
        d["win_length"] = win_length_;
        d["lag_length"] = lag_length_;
        d["alpha"] = alpha_;
        d["lp_order"] = lp_order_;
        d["lp_a"] = array_output(lp_a_);
        d["lp_b"] = array_output(lp_b_);
        d["win_pre"] = win_pre_;
        d["win_post"] = win_post_;
        return d;
    }

private:
    int win_length_;
    int lag_length_;
    int alpha_;
    int lp_order_;
    std::vector<double> lp_a_;
    std::vector<double> lp_b_;
    int win_pre_;
    int win_post_;
    std::unique_ptr<TempoTrack> impl_;
    std::mutex mutex_;
};

class RhythmTempoTrackV2 {
public:
    RhythmTempoTrackV2(double sample_rate, int df_increment)
        : sample_rate_(sample_rate), df_increment_(df_increment),
          impl_(static_cast<float>(sample_rate), df_increment) {
        if (sample_rate_ <= 0 || !std::isfinite(sample_rate_)) {
            throw std::invalid_argument("sample_rate must be a finite positive value");
        }
        require_positive_int(df_increment_, "df_increment");
    }

    nb::tuple calculate_beat_period(const ArrayD &df, double input_tempo, bool constrain_tempo) {
        std::vector<double> values = vector_input(df);
        if (values.size() <= 640) {
            throw std::invalid_argument("df must contain more than 640 values");
        }
        if (input_tempo <= 0 || !std::isfinite(input_tempo)) {
            throw std::invalid_argument("input_tempo must be a finite positive value");
        }
        std::vector<double> beat_period(values.size(), 0.0);
        std::vector<double> tempi;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            impl_.calculateBeatPeriod(values, beat_period, tempi, input_tempo, constrain_tempo);
        }
        return nb::make_tuple(array_output(std::move(beat_period)), array_output(std::move(tempi)));
    }

    nb::ndarray<nb::numpy, double> calculate_beats(const ArrayD &df, const ArrayD &beat_period,
                                                   double alpha, double tightness) {
        std::vector<double> values = vector_input(df);
        std::vector<double> periods = vector_input_exact(beat_period, values.size(), "beat_period");
        if (values.empty()) throw std::invalid_argument("df must not be empty");
        if (alpha < 0.0 || alpha > 1.0 || !std::isfinite(alpha)) {
            throw std::invalid_argument("alpha must be a finite value in [0, 1]");
        }
        if (tightness <= 0.0 || !std::isfinite(tightness)) {
            throw std::invalid_argument("tightness must be a finite positive value");
        }
        for (double p : periods) {
            if (p <= 0.0) throw std::invalid_argument("beat_period values must be positive");
        }
        std::vector<double> beats;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            impl_.calculateBeats(values, periods, beats, alpha, tightness);
        }
        return array_output(std::move(beats));
    }

    nb::dict params() const {
        nb::dict d;
        d["sample_rate"] = sample_rate_;
        d["df_increment"] = df_increment_;
        return d;
    }

private:
    double sample_rate_;
    int df_increment_;
    TempoTrackV2 impl_;
    std::mutex mutex_;
};

bool is_power_of_two(std::size_t value) { return value != 0 && (value & (value - 1)) == 0; }

double checked_downbeat_rate(double original_sample_rate) {
    if (original_sample_rate <= 0 || !std::isfinite(original_sample_rate)) {
        throw std::invalid_argument("original_sample_rate must be a finite positive value");
    }
    return original_sample_rate;
}

std::size_t checked_decimation_factor(std::size_t decimation_factor) {
    if (!is_power_of_two(decimation_factor) || decimation_factor > 64) {
        throw std::invalid_argument("decimation_factor must be a power of two no greater than 64");
    }
    return decimation_factor;
}

std::size_t checked_df_increment(std::size_t df_increment, std::size_t decimation_factor) {
    if (df_increment == 0 || df_increment % decimation_factor != 0) {
        throw std::invalid_argument("df_increment must be a positive multiple of decimation_factor");
    }
    return df_increment;
}

class RhythmDownBeat {
public:
    RhythmDownBeat(double original_sample_rate, std::size_t decimation_factor,
                   std::size_t df_increment)
        : original_sample_rate_(checked_downbeat_rate(original_sample_rate)),
          decimation_factor_(checked_decimation_factor(decimation_factor)),
          df_increment_(checked_df_increment(df_increment, decimation_factor_)),
          impl_(static_cast<float>(original_sample_rate_), decimation_factor_, df_increment_) {}

    void set_beats_per_bar(int beats_per_bar) {
        if (beats_per_bar <= 0) throw std::invalid_argument("beats_per_bar must be positive");
        std::lock_guard<std::mutex> lock(mutex_);
        impl_.setBeatsPerBar(beats_per_bar);
        beats_per_bar_ = beats_per_bar;
    }

    nb::ndarray<nb::numpy, int> find_down_beats(const ArrayD &audio, const ArrayD &beats) {
        std::vector<double> audio_d = vector_input(audio);
        std::vector<float> audio_f(audio_d.begin(), audio_d.end());
        std::vector<double> beat_values = vector_input(beats);
        for (std::size_t i = 1; i < beat_values.size(); ++i) {
            if (beat_values[i] < beat_values[i - 1]) {
                throw std::invalid_argument("beats must be sorted");
            }
        }
        std::vector<int> downbeats;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            impl_.findDownBeats(audio_f.data(), audio_f.size(), beat_values, downbeats);
        }
        return int_array_output(std::move(downbeats));
    }

    void push_audio_block(const ArrayD &audio) {
        std::vector<double> audio_d =
            vector_input_exact(audio, df_increment_, "audio");
        std::vector<float> audio_f(audio_d.begin(), audio_d.end());
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            impl_.pushAudioBlock(audio_f.data());
        }
    }

    nb::ndarray<nb::numpy, double> buffered_audio() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::size_t length = 0;
        const float *src = impl_.getBufferedAudio(length);
        std::vector<double> values(length);
        for (std::size_t i = 0; i < length; ++i) values[i] = src[i];
        return array_output(std::move(values));
    }

    void reset_audio_buffer() {
        std::lock_guard<std::mutex> lock(mutex_);
        impl_.resetAudioBuffer();
    }

    nb::ndarray<nb::numpy, double> beat_sd() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<double> values;
        impl_.getBeatSD(values);
        return array_output(std::move(values));
    }

    nb::dict params() const {
        nb::dict d;
        d["original_sample_rate"] = original_sample_rate_;
        d["decimation_factor"] = decimation_factor_;
        d["df_increment"] = df_increment_;
        d["beats_per_bar"] = beats_per_bar_;
        return d;
    }

private:
    double original_sample_rate_;
    std::size_t decimation_factor_;
    std::size_t df_increment_;
    int beats_per_bar_ = 0;
    DownBeat impl_;
    mutable std::mutex mutex_;
};

class RhythmBeatSpectrum {
public:
    nb::ndarray<nb::numpy, double> process(const ArrayD &matrix) {
        std::vector<std::vector<double>> values = matrix_input(matrix);
        std::vector<double> result;
        {
            nb::gil_scoped_release release;
            std::lock_guard<std::mutex> lock(mutex_);
            result = impl_.process(values);
        }
        return array_output(std::move(result));
    }

private:
    BeatSpectrum impl_;
    std::mutex mutex_;
};

nb::tuple rhythm_beats_native(const ArrayD &audio, int sample_rate) {
    std::vector<double> mono = vector_input(audio);
    if (mono.empty()) throw std::invalid_argument("audio must not be empty");
    require_positive_int(sample_rate, "sample_rate");

    int step = static_cast<int>(sample_rate * kStepSecs + 0.0001);
    require_positive_int(step, "derived step size");
    int frame_length = step * 2;
    DFConfig config = make_df_config(step, frame_length, DF_COMPLEXSD, 3.0, false, -1.0, -1.0);
    DetectionFunction df(config);
    std::vector<double> frame(static_cast<std::size_t>(frame_length), 0.0);
    std::vector<double> df_output;

    {
        nb::gil_scoped_release release;
        for (std::size_t k = 0;; ++k) {
            std::size_t start = k * static_cast<std::size_t>(step);
            if (start >= mono.size()) break;
            std::fill(frame.begin(), frame.end(), 0.0);
            std::size_t available = std::min(frame.size(), mono.size() - start);
            std::copy_n(mono.begin() + static_cast<std::ptrdiff_t>(start), available,
                        frame.begin());
            df_output.push_back(df.processTimeDomain(frame.data()));
        }
    }

    std::size_t non_zero_count = df_output.size();
    while (non_zero_count > 0 && df_output[non_zero_count - 1] <= 0.0) --non_zero_count;

    std::vector<double> trimmed_df;
    if (non_zero_count > 2) {
        trimmed_df.assign(df_output.begin() + 2, df_output.begin() + static_cast<long>(non_zero_count));
    }

    std::vector<double> beat_times;
    std::vector<double> tempo_curve;
    double bpm = 0.0;
    if (!trimmed_df.empty()) {
        TempoTrackV2 tracker(static_cast<float>(sample_rate), step);
        std::vector<double> beat_period(trimmed_df.size(), 0.0);
        std::vector<double> tempi;
        std::vector<double> beat_frames;
        {
            nb::gil_scoped_release release;
            if (trimmed_df.size() > 640) {
                tracker.calculateBeatPeriod(trimmed_df, beat_period, tempi, 120.0, false);
                bool usable_periods =
                    std::all_of(beat_period.begin(), beat_period.end(), [](double x) { return x > 0.0; });
                if (usable_periods) {
                    tracker.calculateBeats(trimmed_df, beat_period, beat_frames, 0.9, 4.0);
                }
            }
        }
        beat_times.reserve(beat_frames.size());
        for (double beat : beat_frames) {
            beat_times.push_back((beat * static_cast<double>(step)) / static_cast<double>(sample_rate));
        }
        for (std::size_t i = 0; i + 1 < beat_times.size(); ++i) {
            double dt = beat_times[i + 1] - beat_times[i];
            if (dt > 0.0) tempo_curve.push_back(60.0 / dt);
        }
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
    }

    return nb::make_tuple(array_output(std::move(beat_times)), bpm,
                          array_output(std::move(tempo_curve)));
}

}  // namespace

void bind_rhythm(nb::module_ &m) {
    m.attr("DF_HFC") = DF_HFC;
    m.attr("DF_SPECDIFF") = DF_SPECDIFF;
    m.attr("DF_PHASEDEV") = DF_PHASEDEV;
    m.attr("DF_COMPLEXSD") = DF_COMPLEXSD;
    m.attr("DF_BROADBAND") = DF_BROADBAND;

    m.def("rhythm_default_filter_a", []() { return array_output(default_lpa()); });
    m.def("rhythm_default_filter_b", []() { return array_output(default_lpb()); });
    m.def("rhythm_beats", &rhythm_beats_native, "audio"_a, "sample_rate"_a);

    nb::class_<RhythmDetectionFunction>(m, "RhythmDetectionFunction")
        .def(nb::init<int, int, int, double, bool, double, double>(),
             "step_size"_a, "frame_length"_a, "df_type"_a = DF_COMPLEXSD, "db_rise"_a = 3.0,
             "adaptive_whitening"_a = false, "whitening_relax_coeff"_a = -1.0,
             "whitening_floor"_a = -1.0)
        .def("process_time_domain", &RhythmDetectionFunction::process_time_domain)
        .def("process_frequency_domain", &RhythmDetectionFunction::process_frequency_domain)
        .def("spectrum_magnitude", &RhythmDetectionFunction::spectrum_magnitude)
        .def_prop_ro("parameters", &RhythmDetectionFunction::params);

    nb::class_<RhythmPeakPicking>(m, "RhythmPeakPicking")
        .def(nb::init<int, double, int, double, int, std::vector<double>, std::vector<double>,
                      int, int, double, double, double, double>(),
             "length"_a, "tau"_a, "alpha"_a = 3, "cutoff"_a = 0.4, "lp_order"_a = 2,
             "lp_a"_a = default_lpa(), "lp_b"_a = default_lpb(), "win_pre"_a = 3,
             "win_post"_a = 3, "quad_a"_a = 0.0, "quad_b"_a = 0.0, "quad_c"_a = 0.0,
             "delta"_a = 0.0)
        .def("process", &RhythmPeakPicking::process)
        .def_prop_ro("parameters", &RhythmPeakPicking::params);

    nb::class_<RhythmTempoTrack>(m, "RhythmTempoTrack")
        .def(nb::init<int, int, int, int, std::vector<double>, std::vector<double>, int, int>(),
             "win_length"_a = 512, "lag_length"_a = 128, "alpha"_a = 3, "lp_order"_a = 2,
             "lp_a"_a = default_lpa(), "lp_b"_a = default_lpb(), "win_pre"_a = 3,
             "win_post"_a = 3)
        .def("process", &RhythmTempoTrack::process)
        .def_prop_ro("parameters", &RhythmTempoTrack::params);

    nb::class_<RhythmTempoTrackV2>(m, "RhythmTempoTrackV2")
        .def(nb::init<double, int>(), "sample_rate"_a, "df_increment"_a)
        .def("calculate_beat_period", &RhythmTempoTrackV2::calculate_beat_period,
             "df"_a, "input_tempo"_a = 120.0, "constrain_tempo"_a = false)
        .def("calculate_beats", &RhythmTempoTrackV2::calculate_beats, "df"_a, "beat_period"_a,
             "alpha"_a = 0.9, "tightness"_a = 4.0)
        .def_prop_ro("parameters", &RhythmTempoTrackV2::params);

    nb::class_<RhythmDownBeat>(m, "RhythmDownBeat")
        .def(nb::init<double, std::size_t, std::size_t>(), "original_sample_rate"_a,
             "decimation_factor"_a, "df_increment"_a)
        .def("set_beats_per_bar", &RhythmDownBeat::set_beats_per_bar)
        .def("find_down_beats", &RhythmDownBeat::find_down_beats)
        .def("push_audio_block", &RhythmDownBeat::push_audio_block)
        .def("buffered_audio", &RhythmDownBeat::buffered_audio)
        .def("reset_audio_buffer", &RhythmDownBeat::reset_audio_buffer)
        .def("beat_sd", &RhythmDownBeat::beat_sd)
        .def_prop_ro("parameters", &RhythmDownBeat::params);

    nb::class_<RhythmBeatSpectrum>(m, "RhythmBeatSpectrum")
        .def(nb::init<>())
        .def("process", &RhythmBeatSpectrum::process);
}
