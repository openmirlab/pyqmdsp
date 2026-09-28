// Utility, conditioning, segmentation, and maths bindings.
// Reads: qm-dsp base, rateconversion, signalconditioning, segmentation, maths, hmm.
#include "common.h"

#include "base/KaiserWindow.h"
#include "base/Pitch.h"
#include "base/SincWindow.h"
#include "base/Window.h"
#include "dsp/rateconversion/Decimator.h"
#include "dsp/rateconversion/DecimatorB.h"
#include "dsp/rateconversion/Resampler.h"
#include "dsp/chromagram/ConstantQ.h"
#include "dsp/segmentation/ClusterMeltSegmenter.h"
#include "dsp/segmentation/cluster_melt.h"
#include "dsp/segmentation/cluster_segmenter.h"
#include "dsp/signalconditioning/DFProcess.h"
#include "dsp/signalconditioning/FiltFilt.h"
#include "dsp/signalconditioning/Filter.h"
#include "dsp/signalconditioning/Framer.h"
#include "hmm/hmm.h"
#include "maths/Correlation.h"
#include "maths/CosineDistance.h"
#include "maths/KLDivergence.h"
#include "maths/MathUtilities.h"
#include "maths/MedianFilter.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

#include "polyfit_decl.h"

using ArrayI64 = nb::ndarray<const int64_t, nb::c_contig, nb::device::cpu>;

namespace {

std::mutex hmm_mutex;

void require_positive(int value, const char *name) {
    if (value <= 0) throw std::invalid_argument(std::string(name) + " must be positive");
}

void require_finite(double value, const char *name) {
    if (!std::isfinite(value)) throw std::invalid_argument(std::string(name) + " must be finite");
}

void require_window_type(const std::string &name, WindowType *type) {
    if (name == "rectangular") *type = RectangularWindow;
    else if (name == "bartlett") *type = BartlettWindow;
    else if (name == "hamming") *type = HammingWindow;
    else if (name == "hanning" || name == "hann") *type = HanningWindow;
    else if (name == "blackman") *type = BlackmanWindow;
    else if (name == "blackman_harris") *type = BlackmanHarrisWindow;
    else throw std::invalid_argument("unknown window type");
}

std::vector<double> vector_input_allow_empty(const ArrayD &a) {
    if (a.ndim() != 1) throw std::invalid_argument("expected a one-dimensional array");
    std::vector<double> v(a.data(), a.data() + a.size());
    for (double x : v) if (!std::isfinite(x)) throw std::invalid_argument("array must contain finite values");
    return v;
}

std::vector<double> nonempty_vector_input(const ArrayD &a, const char *name) {
    auto v = vector_input(a);
    if (v.empty()) throw std::invalid_argument(std::string(name) + " must not be empty");
    return v;
}

void validate_filter_coefficients(const std::vector<double> &a, const std::vector<double> &b) {
    if (b.empty()) throw std::invalid_argument("b coefficients must not be empty");
    if (!a.empty() && a.size() != b.size()) {
        throw std::invalid_argument("a and b coefficients must have the same length");
    }
}

void validate_histogram_params(size_t rows, int histogram_length, int clusters, int neighbour_limit) {
    if (rows < 2) throw std::invalid_argument("at least two feature frames are required");
    require_positive(histogram_length, "histogram_length");
    if (histogram_length % 2 == 0) throw std::invalid_argument("histogram_length must be odd");
    if (histogram_length > int(rows)) throw std::invalid_argument("histogram_length must not exceed frame count");
    require_positive(clusters, "clusters");
    if (clusters > int(rows)) throw std::invalid_argument("clusters must not exceed frame count");
    if (neighbour_limit < 0) throw std::invalid_argument("neighbour_limit must be non-negative");
}

void validate_constq_bins(size_t cols, int bins) {
    require_positive(bins, "bins");
    if (int(cols) < bins || int(cols) % bins != 0) {
        throw std::invalid_argument("column count must be a positive multiple of bins");
    }
}

int constantq_bin_count(int samplerate, int fmin, int fmax, int bins) {
    CQConfig config;
    config.FS = samplerate;
    config.min = fmin;
    config.max = fmax;
    config.BPO = bins;
    config.CQThresh = 0.0054;
    ConstantQ cq(config);
    return cq.getK();
}

Filter::Parameters filter_parameters(const ArrayD &a, const ArrayD &b) {
    Filter::Parameters params;
    params.a = vector_input_allow_empty(a);
    params.b = nonempty_vector_input(b, "b");
    validate_filter_coefficients(params.a, params.b);
    return params;
}

std::vector<double *> row_pointers(std::vector<double> &data, size_t rows, size_t cols) {
    std::vector<double *> ptrs(rows);
    for (size_t i = 0; i < rows; ++i) ptrs[i] = data.data() + i * cols;
    return ptrs;
}

std::vector<double *> const_row_copy(const ArrayD &a, size_t *rows, size_t *cols) {
    if (a.ndim() != 2) throw std::invalid_argument("expected a two-dimensional array");
    *rows = a.shape(0);
    *cols = a.shape(1);
    if (*rows == 0 || *cols == 0) throw std::invalid_argument("matrix dimensions must be positive");
    return {};
}

std::vector<double> matrix_input(const ArrayD &a, size_t *rows, size_t *cols) {
    const_row_copy(a, rows, cols);
    std::vector<double> data(a.data(), a.data() + a.size());
    for (double x : data) if (!std::isfinite(x)) throw std::invalid_argument("matrix must contain finite values");
    return data;
}

nb::ndarray<nb::numpy, double> matrix_output(std::vector<double> values, size_t rows, size_t cols) {
    auto *v = new std::vector<double>(std::move(values));
    nb::capsule owner(v, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    return nb::ndarray<nb::numpy, double>(v->data(), {rows, cols}, owner);
}

nb::ndarray<nb::numpy, int64_t> int_array_output(std::vector<int64_t> values) {
    auto *v = new std::vector<int64_t>(std::move(values));
    nb::capsule owner(v, [](void *p) noexcept { delete static_cast<std::vector<int64_t> *>(p); });
    return nb::ndarray<nb::numpy, int64_t>(v->data(), {v->size()}, owner);
}

nb::dict segmentation_to_dict(const Segmentation &s) {
    nb::list segments;
    for (const auto &seg : s.segments) {
        nb::dict item;
        item["start"] = seg.start;
        item["end"] = seg.end;
        item["type"] = seg.type;
        segments.append(item);
    }
    nb::dict out;
    out["nsegtypes"] = s.nsegtypes;
    out["samplerate"] = s.samplerate;
    out["segments"] = segments;
    return out;
}

class UtilitiesDecimator {
public:
    UtilitiesDecimator(int input_length, int factor) : input_length(input_length), factor(factor) {
        require_positive(input_length, "input_length");
        require_positive(factor, "factor");
        if (input_length % factor != 0) throw std::invalid_argument("input_length must be divisible by factor");
        if (!MathUtilities::isPowerOfTwo(factor)) throw std::invalid_argument("factor must be a power of two");
        if (factor > Decimator::getHighestSupportedFactor()) {
            throw std::invalid_argument("factor is higher than Decimator supports");
        }
        decimator = std::make_unique<Decimator>(input_length, factor);
    }

    nb::ndarray<nb::numpy, double> process(const ArrayD &input) {
        auto data = vector_input(input);
        if (int(data.size()) != input_length) throw std::invalid_argument("input length does not match constructor");
        std::vector<double> out(input_length / factor);
        {
            nb::gil_scoped_release release;
            decimator->process(data.data(), out.data());
        }
        return array_output(std::move(out));
    }

    void reset_filter() { decimator->resetFilter(); }

    int input_length;
    int factor;
    std::unique_ptr<Decimator> decimator;
};

class UtilitiesDecimatorB {
public:
    UtilitiesDecimatorB(int input_length, int factor) : input_length(input_length), factor(factor) {
        require_positive(input_length, "input_length");
        require_positive(factor, "factor");
        if (input_length % factor != 0) throw std::invalid_argument("input_length must be divisible by factor");
        if (!MathUtilities::isPowerOfTwo(factor)) throw std::invalid_argument("factor must be a power of two");
        decimator = std::make_unique<DecimatorB>(input_length, factor);
    }

    nb::ndarray<nb::numpy, double> process(const ArrayD &input) {
        auto data = vector_input(input);
        if (int(data.size()) != input_length) throw std::invalid_argument("input length does not match constructor");
        std::vector<double> out(input_length / factor);
        {
            nb::gil_scoped_release release;
            decimator->process(data.data(), out.data());
        }
        return array_output(std::move(out));
    }

    int input_length;
    int factor;
    std::unique_ptr<DecimatorB> decimator;
};

class UtilitiesResampler {
public:
    UtilitiesResampler(int source_rate, int target_rate, double snr, double bandwidth)
        : source_rate(source_rate), target_rate(target_rate) {
        require_positive(source_rate, "source_rate");
        require_positive(target_rate, "target_rate");
        require_finite(snr, "snr");
        require_finite(bandwidth, "bandwidth");
        if (snr <= 0) throw std::invalid_argument("snr must be positive");
        if (bandwidth <= 0 || bandwidth >= source_rate / 2.0) {
            throw std::invalid_argument("bandwidth must be positive and below Nyquist");
        }
        resampler = std::make_unique<Resampler>(source_rate, target_rate, snr, bandwidth);
    }

    nb::ndarray<nb::numpy, double> process(const ArrayD &input) {
        auto data = vector_input(input);
        std::vector<double> out;
        {
            nb::gil_scoped_release release;
            out = resampler->process(data.data(), int(data.size()));
        }
        return array_output(std::move(out));
    }

    int latency() const { return resampler->getLatency(); }

    int source_rate;
    int target_rate;
    std::unique_ptr<Resampler> resampler;
};

class UtilitiesFilter {
public:
    UtilitiesFilter(const ArrayD &a, const ArrayD &b) {
        filter = std::make_unique<Filter>(filter_parameters(a, b));
    }

    nb::ndarray<nb::numpy, double> process(const ArrayD &input) {
        auto data = vector_input(input);
        std::vector<double> out(data.size());
        {
            nb::gil_scoped_release release;
            filter->process(data.data(), out.data(), int(data.size()));
        }
        return array_output(std::move(out));
    }

    void reset() { filter->reset(); }
    int order() const { return filter->getOrder(); }

    std::unique_ptr<Filter> filter;
};

class UtilitiesMedianFilter {
public:
    UtilitiesMedianFilter(int size, double percentile) {
        require_positive(size, "size");
        require_finite(percentile, "percentile");
        if (percentile < 0 || percentile > 100) throw std::invalid_argument("percentile must be in [0, 100]");
        filter = std::make_unique<MedianFilter<double>>(size, float(percentile));
    }

    void push(double value) {
        require_finite(value, "value");
        filter->push(value);
    }
    double get() const { return filter->get(); }
    double get_at(double percentile) {
        require_finite(percentile, "percentile");
        if (percentile < 0 || percentile > 100) throw std::invalid_argument("percentile must be in [0, 100]");
        return filter->getAt(float(percentile));
    }
    void reset() { filter->reset(); }
    int size() const { return filter->getSize(); }

    std::unique_ptr<MedianFilter<double>> filter;
};

class UtilitiesClusterMeltSegmenter {
public:
    UtilitiesClusterMeltSegmenter(int feature_type, double hop_size, double window_size,
                                  int fmin, int fmax, int nbins, int ncomponents,
                                  int hmm_states, int clusters, int histogram_length,
                                  int neighbourhood_limit) {
        if (feature_type != FEATURE_TYPE_CONSTQ && feature_type != FEATURE_TYPE_CHROMA &&
            feature_type != FEATURE_TYPE_MFCC) {
            throw std::invalid_argument("unsupported feature type");
        }
        require_finite(hop_size, "hop_size");
        require_finite(window_size, "window_size");
        if (hop_size <= 0 || window_size <= 0) throw std::invalid_argument("window and hop sizes must be positive");
        require_positive(fmin, "fmin");
        require_positive(fmax, "fmax");
        if (fmax <= fmin) throw std::invalid_argument("fmax must exceed fmin");
        require_positive(nbins, "nbins");
        require_positive(ncomponents, "ncomponents");
        require_positive(hmm_states, "hmm_states");
        require_positive(clusters, "clusters");
        require_positive(histogram_length, "histogram_length");
        if (histogram_length % 2 == 0) throw std::invalid_argument("histogram_length must be odd");
        if (neighbourhood_limit < 0) throw std::invalid_argument("neighbourhood_limit must be non-negative");

        ClusterMeltSegmenterParams params;
        configured_feature_type = feature_type;
        params.featureType = static_cast<feature_types>(configured_feature_type);
        params.hopSize = hop_size;
        params.windowSize = window_size;
        params.fmin = fmin;
        params.fmax = fmax;
        params.nbins = nbins;
        this->fmin = fmin;
        this->fmax = fmax;
        this->nbins = nbins;
        params.ncomponents = ncomponents;
        params.nHMMStates = hmm_states;
        params.nclusters = clusters;
        params.histogramLength = histogram_length;
        params.neighbourhoodLimit = neighbourhood_limit;
        this->histogram_length = histogram_length;
        segmenter = std::make_unique<ClusterMeltSegmenter>(params);
    }

    void initialise(int samplerate) {
        require_positive(samplerate, "samplerate");
        if (initialized) throw std::invalid_argument("ClusterMeltSegmenter can only be initialised once");
        if (configured_feature_type == FEATURE_TYPE_CONSTQ || configured_feature_type == FEATURE_TYPE_CHROMA) {
            int internal_rate = 11025;
            int decimation_factor = samplerate / internal_rate;
            if (decimation_factor < 1) decimation_factor = 1;
            while (decimation_factor & (decimation_factor - 1)) ++decimation_factor;
            if (decimation_factor > Decimator::getHighestSupportedFactor()) {
                decimation_factor = Decimator::getHighestSupportedFactor();
            }
            int ncoeff = constantq_bin_count(samplerate / decimation_factor, fmin, fmax, nbins);
            if (configured_feature_type == FEATURE_TYPE_CONSTQ && ncoeff < 21) {
                throw std::invalid_argument("constq segmentation requires at least 21 ConstantQ bins");
            }
        }
        std::lock_guard<std::mutex> lock(hmm_mutex);
        segmenter->initialise(samplerate);
        initialized = true;
    }

    int get_windowsize() {
        require_initialized();
        std::lock_guard<std::mutex> lock(hmm_mutex);
        return segmenter->getWindowsize();
    }

    int get_hopsize() {
        require_initialized();
        std::lock_guard<std::mutex> lock(hmm_mutex);
        return segmenter->getHopsize();
    }

    void extract_features(const ArrayD &samples) {
        require_initialized();
        if (feature_source == 2) throw std::invalid_argument("cannot extract audio features after set_features");
        auto data = vector_input(samples);
        std::lock_guard<std::mutex> lock(hmm_mutex);
        int expected = segmenter->getWindowsize();
        if (int(data.size()) != expected) throw std::invalid_argument("samples length must equal get_windowsize()");
        segmenter->extractFeatures(data.data(), int(data.size()));
        ++feature_count;
        feature_source = 1;
        segmented = false;
    }

    void set_features(const ArrayD &features) {
        require_initialized();
        if (feature_source == 1) throw std::invalid_argument("cannot set feature matrix after extract_features");
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(features, &rows, &cols);
        validate_histogram_params(rows, histogram_length, 1, 0);
        std::vector<std::vector<double>> f(rows, std::vector<double>(cols));
        for (size_t r = 0; r < rows; ++r) {
            std::copy_n(data.data() + r * cols, cols, f[r].data());
        }
        std::lock_guard<std::mutex> lock(hmm_mutex);
        segmenter->setFeatures(f);
        feature_count = rows;
        feature_source = 2;
        segmented = false;
    }

    void segment_default() {
        require_ready_to_segment(segmenter->getNSegmentTypes());
        std::lock_guard<std::mutex> lock(hmm_mutex);
        segmenter->segment();
        feature_count = 0;
        segmented = true;
    }

    void segment_types(int segment_types) {
        require_ready_to_segment(segment_types);
        std::lock_guard<std::mutex> lock(hmm_mutex);
        segmenter->segment(segment_types);
        feature_count = 0;
        segmented = true;
    }

    void clear() {
        require_initialized();
        std::lock_guard<std::mutex> lock(hmm_mutex);
        segmenter->clear();
        feature_count = 0;
        segmented = false;
    }

    nb::dict get_segmentation() {
        if (!segmented) throw std::invalid_argument("segment must be called before get_segmentation");
        std::lock_guard<std::mutex> lock(hmm_mutex);
        return segmentation_to_dict(segmenter->getSegmentation());
    }

    int get_n_segment_types() {
        std::lock_guard<std::mutex> lock(hmm_mutex);
        return segmenter->getNSegmentTypes();
    }

    std::unique_ptr<ClusterMeltSegmenter> segmenter;
    int histogram_length = 0;
    int configured_feature_type = FEATURE_TYPE_UNKNOWN;
    int fmin = 0;
    int fmax = 0;
    int nbins = 0;
    size_t feature_count = 0;
    bool initialized = false;
    int feature_source = 0;
    bool segmented = false;

private:
    void require_initialized() const {
        if (!initialized) throw std::invalid_argument("ClusterMeltSegmenter must be initialised first");
    }

    void require_ready_to_segment(int segment_types) const {
        require_initialized();
        validate_histogram_params(feature_count, histogram_length, segment_types, 0);
    }
};

} // namespace

void bind_utilities(nb::module_ &m) {
    m.def("pitch_frequency", [](int midi_pitch, double cents_offset, double concert_a) {
        require_finite(cents_offset, "cents_offset");
        require_finite(concert_a, "concert_a");
        if (concert_a <= 0) throw std::invalid_argument("concert_a must be positive");
        return double(Pitch::getFrequencyForPitch(midi_pitch, float(cents_offset), float(concert_a)));
    }, "midi_pitch"_a, "cents_offset"_a = 0.0, "concert_a"_a = 440.0);

    m.def("pitch_for_frequency", [](double frequency, double concert_a) {
        require_finite(frequency, "frequency");
        require_finite(concert_a, "concert_a");
        if (frequency <= 0) throw std::invalid_argument("frequency must be positive");
        if (concert_a <= 0) throw std::invalid_argument("concert_a must be positive");
        float cents = 0.0f;
        int pitch = Pitch::getPitchForFrequency(float(frequency), &cents, float(concert_a));
        return nb::make_tuple(pitch, double(cents));
    }, "frequency"_a, "concert_a"_a = 440.0);

    m.def("window_data", [](const std::string &type_name, int size) {
        require_positive(size, "size");
        WindowType type;
        require_window_type(type_name, &type);
        Window<double> window(type, size);
        return array_output(window.getWindowData());
    }, "type"_a, "size"_a);

    m.def("window_cut", [](const std::string &type_name, const ArrayD &input) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        WindowType type;
        require_window_type(type_name, &type);
        std::vector<double> out(data.size());
        {
            nb::gil_scoped_release release;
            Window<double> window(type, int(data.size()));
            window.cut(data.data(), out.data());
        }
        return array_output(std::move(out));
    }, "type"_a, "input"_a);

    m.def("kaiser_parameters_transition_width", [](double attenuation, double transition) {
        require_finite(attenuation, "attenuation");
        require_finite(transition, "transition");
        if (transition <= 0) throw std::invalid_argument("transition must be positive");
        auto p = KaiserWindow::parametersForTransitionWidth(attenuation, transition);
        return nb::make_tuple(p.length, p.beta);
    }, "attenuation"_a, "transition"_a);

    m.def("kaiser_parameters_bandwidth", [](double attenuation, double bandwidth, double samplerate) {
        require_finite(bandwidth, "bandwidth");
        require_finite(samplerate, "samplerate");
        if (bandwidth <= 0) throw std::invalid_argument("bandwidth must be positive");
        if (samplerate <= 0) throw std::invalid_argument("samplerate must be positive");
        auto p = KaiserWindow::parametersForBandwidth(attenuation, bandwidth, samplerate);
        return nb::make_tuple(p.length, p.beta);
    }, "attenuation"_a, "bandwidth"_a, "samplerate"_a);

    m.def("kaiser_window", [](int length, double beta) {
        require_positive(length, "length");
        require_finite(beta, "beta");
        KaiserWindow window({length, beta});
        return array_output(std::vector<double>(window.getWindow(), window.getWindow() + window.getLength()));
    }, "length"_a, "beta"_a);

    m.def("kaiser_window_transition_width", [](double attenuation, double transition) {
        KaiserWindow window = KaiserWindow::byTransitionWidth(attenuation, transition);
        return array_output(std::vector<double>(window.getWindow(), window.getWindow() + window.getLength()));
    }, "attenuation"_a, "transition"_a);

    m.def("kaiser_window_bandwidth", [](double attenuation, double bandwidth, double samplerate) {
        KaiserWindow window = KaiserWindow::byBandwidth(attenuation, bandwidth, samplerate);
        return array_output(std::vector<double>(window.getWindow(), window.getWindow() + window.getLength()));
    }, "attenuation"_a, "bandwidth"_a, "samplerate"_a);

    m.def("kaiser_cut", [](double beta, const ArrayD &input) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        KaiserWindow window({int(data.size()), beta});
        std::vector<double> out(data.size());
        window.cut(data.data(), out.data());
        return array_output(std::move(out));
    }, "beta"_a, "input"_a);

    m.def("sinc_window", [](int length, double p) {
        require_positive(length, "length");
        require_finite(p, "p");
        if (p <= 0) throw std::invalid_argument("p must be positive");
        SincWindow window(length, p);
        return array_output(std::vector<double>(window.getWindow(), window.getWindow() + window.getLength()));
    }, "length"_a, "p"_a);

    m.def("sinc_cut", [](double p, const ArrayD &input) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        SincWindow window(int(data.size()), p);
        std::vector<double> out(data.size());
        window.cut(data.data(), out.data());
        return array_output(std::move(out));
    }, "p"_a, "input"_a);

    nb::class_<UtilitiesDecimator>(m, "UtilitiesDecimator")
        .def(nb::init<int, int>(), "input_length"_a, "factor"_a)
        .def("process", &UtilitiesDecimator::process)
        .def("reset_filter", &UtilitiesDecimator::reset_filter)
        .def_prop_ro("factor", [](const UtilitiesDecimator &d) { return d.factor; })
        .def_prop_ro("input_length", [](const UtilitiesDecimator &d) { return d.input_length; });

    nb::class_<UtilitiesDecimatorB>(m, "UtilitiesDecimatorB")
        .def(nb::init<int, int>(), "input_length"_a, "factor"_a)
        .def("process", &UtilitiesDecimatorB::process)
        .def_prop_ro("factor", [](const UtilitiesDecimatorB &d) { return d.factor; })
        .def_prop_ro("input_length", [](const UtilitiesDecimatorB &d) { return d.input_length; });

    nb::class_<UtilitiesResampler>(m, "UtilitiesResampler")
        .def(nb::init<int, int, double, double>(), "source_rate"_a, "target_rate"_a,
             "snr"_a = 100.0, "bandwidth"_a = 15000.0)
        .def("process", &UtilitiesResampler::process)
        .def_prop_ro("latency", &UtilitiesResampler::latency)
        .def_prop_ro("source_rate", [](const UtilitiesResampler &r) { return r.source_rate; })
        .def_prop_ro("target_rate", [](const UtilitiesResampler &r) { return r.target_rate; });

    m.def("resample", [](int source_rate, int target_rate, const ArrayD &input) {
        require_positive(source_rate, "source_rate");
        require_positive(target_rate, "target_rate");
        auto data = vector_input(input);
        std::vector<double> out;
        {
            nb::gil_scoped_release release;
            out = Resampler::resample(source_rate, target_rate, data.data(), int(data.size()));
        }
        return array_output(std::move(out));
    }, "source_rate"_a, "target_rate"_a, "input"_a);

    nb::class_<UtilitiesFilter>(m, "UtilitiesFilter")
        .def(nb::init<const ArrayD &, const ArrayD &>(), "a"_a, "b"_a)
        .def("process", &UtilitiesFilter::process)
        .def("reset", &UtilitiesFilter::reset)
        .def_prop_ro("order", &UtilitiesFilter::order);

    m.def("filtfilt", [](const ArrayD &a, const ArrayD &b, const ArrayD &input) {
        auto params = filter_parameters(a, b);
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        std::vector<double> out(data.size());
        {
            nb::gil_scoped_release release;
            FiltFilt ff(params);
            ff.process(data.data(), out.data(), int(data.size()));
        }
        return array_output(std::move(out));
    }, "a"_a, "b"_a, "input"_a);

    m.def("df_process", [](const ArrayD &input, const ArrayD &a, const ArrayD &b,
                           int win_pre, int win_post, int alpha_norm_param,
                           bool is_median_positive, double delta) {
        auto params = filter_parameters(a, b);
        if (params.a.empty()) throw std::invalid_argument("DFProcess requires IIR a coefficients");
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        if (win_pre < 0 || win_post < 0) throw std::invalid_argument("median windows must be non-negative");
        require_positive(alpha_norm_param, "alpha_norm_param");
        std::vector<double> out(data.size());
        DFProcConfig config;
        config.length = int(data.size());
        config.LPOrd = int(params.a.size()) - 1;
        config.LPACoeffs = params.a.data();
        config.LPBCoeffs = params.b.data();
        config.winPre = win_pre;
        config.winPost = win_post;
        config.AlphaNormParam = alpha_norm_param;
        config.isMedianPositive = is_median_positive;
        config.delta = float(delta);
        {
            nb::gil_scoped_release release;
            DFProcess process(config);
            process.process(data.data(), out.data());
        }
        return array_output(std::move(out));
    }, "input"_a, "a"_a, "b"_a, "win_pre"_a, "win_post"_a,
       "alpha_norm_param"_a = 2, "is_median_positive"_a = true, "delta"_a = 0.0);

    m.def("frames", [](const ArrayD &input, int frame_length, int hop) {
        auto data = vector_input(input);
        require_positive(frame_length, "frame_length");
        require_positive(hop, "hop");
        std::vector<double> out;
        int frames = 0;
        {
            nb::gil_scoped_release release;
            Framer framer;
            framer.configure(frame_length, hop);
            framer.setSource(data.data(), int64_t(data.size()));
            frames = framer.getMaxNoFrames();
            out.resize(size_t(frames) * size_t(frame_length));
            for (int i = 0; i < frames; ++i) framer.getFrame(out.data() + size_t(i) * frame_length);
        }
        return matrix_output(std::move(out), size_t(frames), size_t(frame_length));
    }, "input"_a, "frame_length"_a, "hop"_a);

    m.def("autocorrelation_unbiased", [](const ArrayD &input) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        std::vector<double> out(data.size());
        {
            nb::gil_scoped_release release;
            Correlation c;
            c.doAutoUnBiased(data.data(), out.data(), int(data.size()));
        }
        return array_output(std::move(out));
    }, "input"_a);

    m.def("cosine_distance", [](const ArrayD &a, const ArrayD &b) {
        auto va = vector_input(a);
        auto vb = vector_input(b);
        if (va.size() != vb.size()) throw std::invalid_argument("vectors must have the same length");
        if (va.empty()) throw std::invalid_argument("vectors must not be empty");
        return CosineDistance().distance(va, vb);
    }, "a"_a, "b"_a);

    m.def("kl_gaussian", [](const ArrayD &mean_a, const ArrayD &var_a,
                            const ArrayD &mean_b, const ArrayD &var_b) {
        auto ma = vector_input(mean_a);
        auto va = vector_input(var_a);
        auto mb = vector_input(mean_b);
        auto vb = vector_input(var_b);
        if (ma.empty() || ma.size() != va.size() || ma.size() != mb.size() || ma.size() != vb.size()) {
            throw std::invalid_argument("all vectors must be non-empty and have the same length");
        }
        for (double x : va) if (x < 0) throw std::invalid_argument("variances must be non-negative");
        for (double x : vb) if (x < 0) throw std::invalid_argument("variances must be non-negative");
        return KLDivergence().distanceGaussian(ma, va, mb, vb);
    }, "mean_a"_a, "var_a"_a, "mean_b"_a, "var_b"_a);

    m.def("kl_distribution", [](const ArrayD &a, const ArrayD &b, bool symmetrised) {
        auto va = vector_input(a);
        auto vb = vector_input(b);
        if (va.empty() || va.size() != vb.size()) throw std::invalid_argument("vectors must be non-empty and same length");
        for (double x : va) if (x < 0) throw std::invalid_argument("distributions must be non-negative");
        for (double x : vb) if (x < 0) throw std::invalid_argument("distributions must be non-negative");
        return KLDivergence().distanceDistribution(va, vb, symmetrised);
    }, "a"_a, "b"_a, "symmetrised"_a = false);

    m.def("math_summary", [](const ArrayD &input, int alpha) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        require_positive(alpha, "alpha");
        double min_value = 0.0;
        double max_value = 0.0;
        MathUtilities::getFrameMinMax(data.data(), int(data.size()), &min_value, &max_value);
        nb::dict out;
        out["min"] = min_value;
        out["max"] = max_value;
        out["mean"] = MathUtilities::mean(data.data(), int(data.size()));
        out["sum"] = MathUtilities::sum(data.data(), int(data.size()));
        out["median"] = MathUtilities::median(data.data(), int(data.size()));
        out["alpha_norm"] = MathUtilities::getAlphaNorm(data, alpha);
        return out;
    }, "input"_a, "alpha"_a = 2);

    m.def("normalise", [](const ArrayD &input, const std::string &mode) {
        auto data = vector_input(input);
        MathUtilities::NormaliseType type = MathUtilities::NormaliseUnitMax;
        if (mode == "none") type = MathUtilities::NormaliseNone;
        else if (mode == "unit_sum") type = MathUtilities::NormaliseUnitSum;
        else if (mode == "unit_max") type = MathUtilities::NormaliseUnitMax;
        else throw std::invalid_argument("unknown normalise mode");
        MathUtilities::normalise(data, type);
        return array_output(std::move(data));
    }, "input"_a, "mode"_a = "unit_max");

    m.def("lp_norm", [](const ArrayD &input, int p) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        if (p < 0) throw std::invalid_argument("p must be non-negative");
        return MathUtilities::getLpNorm(data, p);
    }, "input"_a, "p"_a);

    m.def("normalise_lp", [](const ArrayD &input, int p, double threshold) {
        auto data = vector_input(input);
        if (p < 0) throw std::invalid_argument("p must be non-negative");
        require_finite(threshold, "threshold");
        if (threshold < 0) throw std::invalid_argument("threshold must be non-negative");
        return array_output(MathUtilities::normaliseLp(data, p, threshold));
    }, "input"_a, "p"_a, "threshold"_a = 1e-6);

    m.def("adaptive_threshold", [](const ArrayD &input) {
        auto data = vector_input(input);
        MathUtilities::adaptiveThreshold(data);
        return array_output(std::move(data));
    }, "input"_a);

    m.def("circ_shift", [](const ArrayD &input, int shift) {
        auto data = vector_input(input);
        if (!data.empty()) MathUtilities::circShift(data.data(), int(data.size()), shift);
        return array_output(std::move(data));
    }, "input"_a, "shift"_a);

    m.def("argmax", [](const ArrayD &input) {
        auto data = vector_input(input);
        if (data.empty()) throw std::invalid_argument("input must not be empty");
        double max_value = 0.0;
        int index = MathUtilities::getMax(data, &max_value);
        return nb::make_tuple(index, max_value);
    }, "input"_a);

    m.def("scalar_math", [](double x, double y) {
        require_finite(x, "x");
        require_finite(y, "y");
        nb::dict out;
        out["round"] = MathUtilities::round(x);
        out["princarg"] = MathUtilities::princarg(x);
        out["mod"] = MathUtilities::mod(x, y);
        return out;
    }, "x"_a, "y"_a);

    m.def("integer_math", [](int x, int y) {
        nb::dict out;
        out["is_power_of_two"] = MathUtilities::isPowerOfTwo(x);
        out["next_power_of_two"] = MathUtilities::nextPowerOfTwo(x);
        out["previous_power_of_two"] = MathUtilities::previousPowerOfTwo(x);
        out["nearest_power_of_two"] = MathUtilities::nearestPowerOfTwo(x);
        out["factorial"] = MathUtilities::factorial(x);
        out["gcd"] = MathUtilities::gcd(x, y);
        return out;
    }, "x"_a, "y"_a);

    m.def("median_filter", [](int size, const ArrayD &input) {
        require_positive(size, "size");
        auto data = vector_input(input);
        return array_output(MedianFilter<double>::filter(size, data));
    }, "size"_a, "input"_a);

    nb::class_<UtilitiesMedianFilter>(m, "UtilitiesMedianFilter")
        .def(nb::init<int, double>(), "size"_a, "percentile"_a = 50.0)
        .def("push", &UtilitiesMedianFilter::push)
        .def("get", &UtilitiesMedianFilter::get)
        .def("get_at", &UtilitiesMedianFilter::get_at)
        .def("reset", &UtilitiesMedianFilter::reset)
        .def_prop_ro("size", &UtilitiesMedianFilter::size);

    m.def("polyfit", [](const ArrayD &x, const ArrayD &y, int terms) {
        auto vx = vector_input(x);
        auto vy = vector_input(y);
        require_positive(terms, "terms");
        if (vx.size() != vy.size()) throw std::invalid_argument("x and y must have the same length");
        if (vx.size() < 2) throw std::invalid_argument("at least two points are required");
        std::vector<double> coefs(size_t(terms), 0.0);
        double corr = TPolyFit::PolyFit2(vx, vy, coefs);
        return nb::make_tuple(array_output(std::move(coefs)), corr);
    }, "x"_a, "y"_a, "terms"_a);

    m.def("mpeg7_constq", [](const ArrayD &input) {
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(input, &rows, &cols);
        data.resize(rows * (cols + 1), 0.0);
        for (size_t r = rows; r-- > 0;) {
            std::copy_backward(data.begin() + ptrdiff_t(r * cols),
                               data.begin() + ptrdiff_t(r * cols + cols),
                               data.begin() + ptrdiff_t(r * (cols + 1) + cols));
        }
        auto ptrs = row_pointers(data, rows, cols + 1);
        mpeg7_constq(ptrs.data(), int(rows), int(cols));
        return matrix_output(std::move(data), rows, cols + 1);
    }, "input"_a);

    m.def("cq_to_chroma", [](const ArrayD &input, int bins) {
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(input, &rows, &cols);
        require_positive(bins, "bins");
        if (int(cols) < bins || int(cols) % bins != 0) throw std::invalid_argument("column count must be a positive multiple of bins");
        auto in_ptrs = row_pointers(data, rows, cols);
        std::vector<double> out(rows * size_t(bins), 0.0);
        auto out_ptrs = row_pointers(out, rows, size_t(bins));
        cq2chroma(in_ptrs.data(), int(rows), int(cols), bins, out_ptrs.data());
        return matrix_output(std::move(out), rows, size_t(bins));
    }, "input"_a, "bins"_a);

    m.def("create_histograms", [](const ArrayI64 &labels, int bins, int length) {
        if (labels.ndim() != 1) throw std::invalid_argument("labels must be one-dimensional");
        require_positive(bins, "bins");
        require_positive(length, "length");
        if (length % 2 == 0) throw std::invalid_argument("length must be odd");
        if (labels.size() == 0) throw std::invalid_argument("labels must not be empty");
        if (length > int(labels.size())) throw std::invalid_argument("length must not exceed labels size");
        std::vector<int> x(labels.size());
        for (size_t i = 0; i < labels.size(); ++i) {
            if (labels.data()[i] < 0 || labels.data()[i] >= bins) throw std::invalid_argument("label outside bin range");
            x[i] = int(labels.data()[i]);
        }
        std::vector<double> out(labels.size() * size_t(bins));
        create_histograms(x.data(), int(x.size()), bins, length, out.data());
        return matrix_output(std::move(out), labels.size(), size_t(bins));
    }, "labels"_a, "bins"_a, "length"_a);

    m.def("cluster_melt", [](const ArrayD &histograms, const ArrayD &schedule,
                             int clusters, int neighbour_limit) {
        size_t rows = 0;
        size_t cols = 0;
        auto h = matrix_input(histograms, &rows, &cols);
        auto bs = nonempty_vector_input(schedule, "schedule");
        require_positive(clusters, "clusters");
        if (clusters > int(rows)) throw std::invalid_argument("clusters must not exceed row count");
        if (neighbour_limit < 0) throw std::invalid_argument("neighbour_limit must be non-negative");
        std::vector<int> c(rows);
        cluster_melt(h.data(), int(cols), int(rows), bs.data(), int(bs.size()), clusters, neighbour_limit, c.data());
        std::vector<int64_t> out(c.begin(), c.end());
        return int_array_output(std::move(out));
    }, "histograms"_a, "schedule"_a, "clusters"_a, "neighbour_limit"_a = 0);

    m.def("cluster_segment", [](const ArrayD &features, int hmm_states,
                                int histogram_length, int clusters, int neighbour_limit) {
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(features, &rows, &cols);
        require_positive(hmm_states, "hmm_states");
        validate_histogram_params(rows, histogram_length, clusters, neighbour_limit);
        auto ptrs = row_pointers(data, rows, cols);
        std::vector<int> q(rows);
        {
            std::lock_guard<std::mutex> lock(hmm_mutex);
            cluster_segment(q.data(), ptrs.data(), int(rows), int(cols), hmm_states,
                            histogram_length, clusters, neighbour_limit);
        }
        std::vector<int64_t> out(q.begin(), q.end());
        return int_array_output(std::move(out));
    }, "features"_a, "hmm_states"_a, "histogram_length"_a, "clusters"_a, "neighbour_limit"_a = 0);

    m.def("constq_segment", [](const ArrayD &features, int bins, int feature_type,
                               int hmm_states, int histogram_length, int clusters,
                               int neighbour_limit) {
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(features, &rows, &cols);
        require_positive(hmm_states, "hmm_states");
        validate_histogram_params(rows, histogram_length, clusters, neighbour_limit);
        validate_constq_bins(cols, bins);
        if (feature_type != FEATURE_TYPE_CONSTQ && feature_type != FEATURE_TYPE_CHROMA) {
            throw std::invalid_argument("feature_type must be constq or chroma");
        }
        if (feature_type == FEATURE_TYPE_CONSTQ && cols < 21) {
            throw std::invalid_argument("constq segmentation requires at least 21 coefficient columns");
        }
        std::vector<double> padded(rows * (cols + 1), 0.0);
        for (size_t r = 0; r < rows; ++r) {
            std::copy_n(data.data() + r * cols, cols, padded.data() + r * (cols + 1));
        }
        auto ptrs = row_pointers(padded, rows, cols + 1);
        std::vector<int> q(rows);
        {
            std::lock_guard<std::mutex> lock(hmm_mutex);
            constq_segment(q.data(), ptrs.data(), int(rows), bins, int(cols), feature_type,
                           hmm_states, histogram_length, clusters, neighbour_limit);
        }
        std::vector<int64_t> out(q.begin(), q.end());
        return int_array_output(std::move(out));
    }, "features"_a, "bins"_a, "feature_type"_a, "hmm_states"_a,
       "histogram_length"_a, "clusters"_a, "neighbour_limit"_a = 0);

    nb::class_<UtilitiesClusterMeltSegmenter>(m, "UtilitiesClusterMeltSegmenter")
        .def(nb::init<int, double, double, int, int, int, int, int, int, int, int>(),
             "feature_type"_a, "hop_size"_a, "window_size"_a, "fmin"_a, "fmax"_a,
             "nbins"_a, "ncomponents"_a, "hmm_states"_a, "clusters"_a,
             "histogram_length"_a, "neighbourhood_limit"_a)
        .def("initialise", &UtilitiesClusterMeltSegmenter::initialise)
        .def("get_windowsize", &UtilitiesClusterMeltSegmenter::get_windowsize)
        .def("get_hopsize", &UtilitiesClusterMeltSegmenter::get_hopsize)
        .def("extract_features", &UtilitiesClusterMeltSegmenter::extract_features)
        .def("set_features", &UtilitiesClusterMeltSegmenter::set_features)
        .def("segment", &UtilitiesClusterMeltSegmenter::segment_default)
        .def("segment_types", &UtilitiesClusterMeltSegmenter::segment_types)
        .def("clear", &UtilitiesClusterMeltSegmenter::clear)
        .def("get_segmentation", &UtilitiesClusterMeltSegmenter::get_segmentation)
        .def("get_n_segment_types", &UtilitiesClusterMeltSegmenter::get_n_segment_types);

    m.def("cluster_melt_segmenter", [](const ArrayD &features, int samplerate, int segment_types) {
        size_t rows = 0;
        size_t cols = 0;
        auto data = matrix_input(features, &rows, &cols);
        require_positive(samplerate, "samplerate");
        require_positive(segment_types, "segment_types");
        int histogram_length = rows >= 15 ? 15 : int(rows | 1);
        if (histogram_length > int(rows)) histogram_length -= 2;
        validate_histogram_params(rows, histogram_length, segment_types, 0);
        std::vector<std::vector<double>> f(rows, std::vector<double>(cols));
        for (size_t r = 0; r < rows; ++r) {
            std::copy_n(data.data() + r * cols, cols, f[r].data());
        }
        ClusterMeltSegmenterParams params;
        params.featureType = FEATURE_TYPE_CHROMA;
        params.nclusters = segment_types;
        params.histogramLength = histogram_length;
        ClusterMeltSegmenter segmenter(params);
        {
            std::lock_guard<std::mutex> lock(hmm_mutex);
            segmenter.initialise(samplerate);
            segmenter.setFeatures(f);
            segmenter.segment(segment_types);
        }
        return segmentation_to_dict(segmenter.getSegmentation());
    }, "features"_a, "samplerate"_a, "segment_types"_a);
}
