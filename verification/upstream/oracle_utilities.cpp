// Pristine qm-dsp oracle for pyqmdsp.utilities parity.
// Reads: original qm-dsp headers and oracle JSON helper.
#include "oracle.h"

#include "base/KaiserWindow.h"
#include "base/Pitch.h"
#include "base/SincWindow.h"
#include "base/Window.h"
#include "dsp/rateconversion/Decimator.h"
#include "dsp/rateconversion/DecimatorB.h"
#include "dsp/rateconversion/Resampler.h"
#include "dsp/segmentation/ClusterMeltSegmenter.h"
#include "dsp/segmentation/cluster_melt.h"
#include "dsp/segmentation/cluster_segmenter.h"
#include "dsp/signalconditioning/DFProcess.h"
#include "dsp/signalconditioning/FiltFilt.h"
#include "dsp/signalconditioning/Filter.h"
#include "dsp/signalconditioning/Framer.h"
#include "maths/Correlation.h"
#include "maths/CosineDistance.h"
#include "maths/KLDivergence.h"
#include "maths/MathUtilities.h"
#include "maths/MedianFilter.h"
#include "maths/Polyfit.h"

#include <algorithm>
#include <cstdlib>
#include <numeric>

namespace {

std::vector<double> cut_window(WindowType type, const std::vector<double> &input) {
    std::vector<double> out(input.size());
    Window<double> window(type, int(input.size()));
    window.cut(input.data(), out.data());
    return out;
}

std::vector<double> window_data(WindowType type, int size) {
    return Window<double>(type, size).getWindowData();
}

std::vector<double> kaiser_values(const KaiserWindow &window) {
    return std::vector<double>(window.getWindow(), window.getWindow() + window.getLength());
}

std::vector<double> sinc_values(const SincWindow &window) {
    return std::vector<double>(window.getWindow(), window.getWindow() + window.getLength());
}

std::vector<double> filter_process(Filter &filter, const std::vector<double> &input) {
    std::vector<double> out(input.size());
    filter.process(input.data(), out.data(), int(input.size()));
    return out;
}

std::vector<double> filtfilt_process(const Filter::Parameters &params, const std::vector<double> &input) {
    std::vector<double> out(input.size());
    FiltFilt ff(params);
    ff.process(input.data(), out.data(), int(input.size()));
    return out;
}

std::vector<double> framer_output(const std::vector<double> &input, int frame_length, int hop, int *frames) {
    std::vector<double> data = input;
    Framer framer;
    framer.configure(frame_length, hop);
    framer.setSource(data.data(), int64_t(data.size()));
    *frames = framer.getMaxNoFrames();
    std::vector<double> out(size_t(*frames) * size_t(frame_length));
    for (int i = 0; i < *frames; ++i) {
        framer.getFrame(out.data() + size_t(i) * size_t(frame_length));
    }
    return out;
}

std::vector<std::vector<double>> reshape(const std::vector<double> &flat, int rows, int cols) {
    std::vector<std::vector<double>> out(rows, std::vector<double>(cols));
    for (int r = 0; r < rows; ++r) {
        std::copy_n(flat.data() + r * cols, cols, out[r].data());
    }
    return out;
}

std::vector<double *> row_ptrs(std::vector<double> &flat, int rows, int cols) {
    std::vector<double *> ptrs(rows);
    for (int r = 0; r < rows; ++r) ptrs[r] = flat.data() + r * cols;
    return ptrs;
}

Json segmentation_json(const Segmentation &s) {
    Json out;
    std::vector<int> starts;
    std::vector<int> ends;
    std::vector<int> types;
    for (const auto &seg : s.segments) {
        starts.push_back(seg.start);
        ends.push_back(seg.end);
        types.push_back(seg.type);
    }
    out.set("nsegtypes", s.nsegtypes);
    out.set("samplerate", s.samplerate);
    out.set("starts", starts);
    out.set("ends", ends);
    out.set("types", types);
    return out;
}

std::vector<double> cqt_features(int rows, int cols) {
    std::vector<double> out(size_t(rows) * size_t(cols));
    for (int r = 0; r < rows; ++r) {
        bool second = r >= rows / 2;
        for (int c = 0; c < cols; ++c) {
            double harmonic = std::sin(0.17 * r + 0.31 * c) + 0.5 * std::cos(0.11 * r * (c + 1));
            double ridge_a = 1.0 / (1.0 + std::abs(c - (4 + (r % 5))));
            double ridge_b = 1.0 / (1.0 + std::abs(c - (15 + (r % 4))));
            double block = second ? 1.6 * ridge_b + 0.2 * ridge_a : 1.4 * ridge_a + 0.25 * ridge_b;
            out[size_t(r) * size_t(cols) + size_t(c)] =
                0.35 + block + 0.07 * harmonic + 0.01 * ((r * 7 + c * 3) % 11);
        }
    }
    return out;
}

std::vector<double> chroma_features(int rows, int cols) {
    std::vector<double> out(size_t(rows) * size_t(cols));
    for (int r = 0; r < rows; ++r) {
        int block = (4 * r) / rows;
        int center_class = (block * 3) % 12;
        for (int c = 0; c < cols; ++c) {
            int pitch_class = c % 12;
            int wrapped = std::abs(pitch_class - center_class);
            int distance = std::min(wrapped, 12 - wrapped);
            double octave_bias = c >= 12 ? 0.17 : 0.0;
            out[size_t(r) * size_t(cols) + size_t(c)] =
                0.01 + octave_bias + 8.0 / (1.0 + 3.0 * distance) +
                0.02 * std::sin(0.37 * r + 0.19 * c);
        }
    }
    return out;
}

void add_base(Json &inputs, Json &outputs) {
    Json in;
    in.set("midi_pitch", 69);
    in.set("frequency", 445.0);
    in.set("concert_a", 440.0);
    inputs.set("base", in);

    Json out;
    out.set("frequency_for_pitch", Pitch::getFrequencyForPitch(69, 12.5f, 440.0f));
    float cents = 0.0f;
    int pitch = Pitch::getPitchForFrequency(445.0f, &cents, 440.0f);
    out.set("pitch_for_frequency", pitch);
    out.set("pitch_cents", cents);

    const int n = 8;
    std::vector<double> ones(n, 1.0);
    Json windows;
    windows.set("rectangular", window_data(RectangularWindow, n));
    windows.set("bartlett", window_data(BartlettWindow, n));
    windows.set("hamming", window_data(HammingWindow, n));
    windows.set("hanning", window_data(HanningWindow, n));
    windows.set("blackman", window_data(BlackmanWindow, n));
    windows.set("blackman_harris", window_data(BlackmanHarrisWindow, n));
    windows.set("hamming_cut", cut_window(HammingWindow, ones));
    out.set("windows", windows);

    auto kp = KaiserWindow::parametersForTransitionWidth(60.0, 0.2);
    KaiserWindow kw(kp);
    std::vector<double> kcut(n, 1.0);
    KaiserWindow({n, kp.beta}).cut(kcut.data());
    Json kaiser;
    kaiser.set("transition_length", kp.length);
    kaiser.set("transition_beta", kp.beta);
    auto bp = KaiserWindow::parametersForBandwidth(60.0, 100.0, 1000.0);
    kaiser.set("bandwidth_length", bp.length);
    kaiser.set("bandwidth_beta", bp.beta);
    kaiser.set("window", kaiser_values(kw));
    kaiser.set("transition_alias_window", kaiser_values(KaiserWindow::byTransitionWidth(60.0, 0.2)));
    kaiser.set("bandwidth_alias_window", kaiser_values(KaiserWindow::byBandwidth(60.0, 100.0, 1000.0)));
    kaiser.set("cut", kcut);
    out.set("kaiser", kaiser);

    SincWindow sw(9, 4.0);
    std::vector<double> scut(9, 1.0);
    sw.cut(scut.data());
    Json sinc;
    sinc.set("window", sinc_values(sw));
    sinc.set("cut", scut);
    out.set("sinc", sinc);
    outputs.set("base", out);
}

void add_rate(Json &inputs, Json &outputs) {
    Json in;
    auto decimator_input = signal(16, 0.2);
    auto decimator_input_b = signal(16, 0.9);
    auto res_in = signal(2304, 0.4);
    in.set("decimator_input", decimator_input);
    in.set("decimator_input_b", decimator_input_b);
    in.set("resampler_input", res_in);
    inputs.set("rate", in);

    Json out;
    Decimator dec(16, 2);
    std::vector<double> dec_out(8), dec_second(8), dec_again(8);
    dec.process(decimator_input.data(), dec_out.data());
    dec.process(decimator_input_b.data(), dec_second.data());
    dec.resetFilter();
    dec.process(decimator_input.data(), dec_again.data());
    out.set("decimator_factor", dec.getFactor());
    out.set("decimator_first", dec_out);
    out.set("decimator_second_history", dec_second);
    out.set("decimator_after_reset", dec_again);

    DecimatorB decb(16, 4);
    std::vector<double> decb_out(4), decb_second(4);
    decb.process(decimator_input.data(), decb_out.data());
    decb.process(decimator_input_b.data(), decb_second.data());
    out.set("decimator_b_factor", decb.getFactor());
    out.set("decimator_b_first", decb_out);
    out.set("decimator_b_second_history", decb_second);

    auto one_off = Resampler::resample(8000, 11025, res_in.data(), int(res_in.size()));
    Resampler up(8000, 16000, 100.0, 3600.0);
    auto chunk_a = up.process(res_in.data(), 777);
    auto chunk_b = up.process(res_in.data() + 777, 777);
    auto chunk_c = up.process(res_in.data() + 1554, int(res_in.size()) - 1554);
    Resampler down(16000, 8000, 90.0, 3000.0);
    auto down_out = down.process(res_in.data(), int(res_in.size()));
    out.set("resample_one_off", one_off);
    out.set("resampler_up_latency", up.getLatency());
    out.set("resampler_up_chunk_a", chunk_a);
    out.set("resampler_up_chunk_b", chunk_b);
    out.set("resampler_up_chunk_c", chunk_c);
    out.set("resampler_down_latency", down.getLatency());
    out.set("resampler_down", down_out);
    outputs.set("rate", out);
}

void add_conditioning(Json &inputs, Json &outputs) {
    auto input = signal(10, 0.6);
    inputs.set("conditioning", input);
    Json out;

    Filter::Parameters fir;
    fir.b = {0.25, 0.5, 0.25};
    Filter filter(fir);
    auto fir_a = filter_process(filter, input);
    auto fir_b = filter_process(filter, input);
    filter.reset();
    auto fir_reset = filter_process(filter, input);
    out.set("filter_order", filter.getOrder());
    out.set("filter_first", fir_a);
    out.set("filter_second_history", fir_b);
    out.set("filter_after_reset", fir_reset);

    Filter::Parameters iir;
    iir.a = {1.0, -0.25};
    iir.b = {0.5, 0.25};
    Filter iir_filter(iir);
    out.set("iir_filter", filter_process(iir_filter, input));
    out.set("filtfilt", filtfilt_process(fir, input));

    DFProcConfig cfg;
    std::vector<double> a = {1.0};
    std::vector<double> b = {1.0};
    cfg.length = int(input.size());
    cfg.LPOrd = 0;
    cfg.LPACoeffs = a.data();
    cfg.LPBCoeffs = b.data();
    cfg.winPre = 1;
    cfg.winPost = 1;
    cfg.AlphaNormParam = 2;
    cfg.isMedianPositive = true;
    cfg.delta = 0.01f;
    std::vector<double> df = input;
    std::vector<double> df_out(input.size());
    DFProcess process(cfg);
    process.process(df.data(), df_out.data());
    out.set("df_process", df_out);

    int frames = 0;
    auto framed = framer_output(input, 4, 3, &frames);
    Json fr;
    fr.set("frames", frames);
    fr.set("data", reshape(framed, frames, 4));
    out.set("framer", fr);
    outputs.set("conditioning", out);
}

void add_maths(Json &inputs, Json &outputs) {
    auto data = signal(9, 0.8);
    inputs.set("maths", data);
    Json out;
    std::vector<double> autocorr(data.size());
    Correlation corr;
    corr.doAutoUnBiased(data.data(), autocorr.data(), int(data.size()));
    out.set("autocorrelation_unbiased", autocorr);
    out.set("cosine_distance", CosineDistance().distance({1.0, 2.0, 3.0}, {3.0, 2.0, 1.0}));
    KLDivergence kl;
    out.set("kl_gaussian", kl.distanceGaussian({0.0, 1.0}, {1.0, 2.0}, {0.5, 0.5}, {1.5, 2.5}));
    out.set("kl_distribution", kl.distanceDistribution({0.2, 0.3, 0.5}, {0.3, 0.3, 0.4}, false));
    out.set("kl_distribution_sym", kl.distanceDistribution({0.2, 0.3, 0.5}, {0.3, 0.3, 0.4}, true));

    double min_value = 0.0, max_value = 0.0;
    MathUtilities::getFrameMinMax(data.data(), int(data.size()), &min_value, &max_value);
    Json summary;
    summary.set("min", min_value);
    summary.set("max", max_value);
    summary.set("mean", MathUtilities::mean(data.data(), int(data.size())));
    summary.set("sum", MathUtilities::sum(data.data(), int(data.size())));
    summary.set("median", MathUtilities::median(data.data(), int(data.size())));
    summary.set("alpha_norm", MathUtilities::getAlphaNorm(data, 2));
    out.set("summary", summary);

    auto normalise_sum = data;
    auto normalise_max = data;
    MathUtilities::normalise(normalise_sum, MathUtilities::NormaliseUnitSum);
    MathUtilities::normalise(normalise_max, MathUtilities::NormaliseUnitMax);
    out.set("normalise_sum", normalise_sum);
    out.set("normalise_max", normalise_max);
    out.set("lp_norm", MathUtilities::getLpNorm(data, 3));
    out.set("normalise_lp", MathUtilities::normaliseLp(data, 2, 1e-6));
    auto adaptive = data;
    MathUtilities::adaptiveThreshold(adaptive);
    out.set("adaptive_threshold", adaptive);
    auto shifted = data;
    MathUtilities::circShift(shifted.data(), int(shifted.size()), 2);
    out.set("circ_shift", shifted);
    double argmax_value = 0.0;
    int argmax_index = MathUtilities::getMax(data, &argmax_value);
    Json argmax;
    argmax.set("index", argmax_index);
    argmax.set("value", argmax_value);
    out.set("argmax", argmax);

    Json scalar;
    scalar.set("round", MathUtilities::round(-1.6));
    scalar.set("princarg", MathUtilities::princarg(-1.6));
    scalar.set("mod", MathUtilities::mod(-1.6, 2.0));
    out.set("scalar", scalar);
    Json integer;
    integer.set("is_power_of_two", MathUtilities::isPowerOfTwo(16));
    integer.set("next_power_of_two", MathUtilities::nextPowerOfTwo(16));
    integer.set("previous_power_of_two", MathUtilities::previousPowerOfTwo(16));
    integer.set("nearest_power_of_two", MathUtilities::nearestPowerOfTwo(16));
    integer.set("factorial", MathUtilities::factorial(16));
    integer.set("gcd", MathUtilities::gcd(16, 30));
    out.set("integer", integer);

    out.set("median_filter", MedianFilter<double>::filter(3, data));
    MedianFilter<double> mf(3, 75.0f);
    std::vector<double> mf_outputs;
    for (double value : data) {
        mf.push(value);
        mf_outputs.push_back(mf.get());
    }
    Json mf_state;
    mf_state.set("outputs", mf_outputs);
    mf_state.set("at_25", mf.getAt(25.0f));
    mf.reset();
    mf_state.set("after_reset", mf.get());
    out.set("median_stateful", mf_state);

    std::vector<double> x = {-2.0, -1.0, 0.0, 1.0, 2.0};
    std::vector<double> y;
    for (double v : x) y.push_back(1.5 - 0.25 * v + 0.75 * v * v);
    std::vector<double> coefs(3);
    double poly_corr = TPolyFit::PolyFit2(x, y, coefs);
    Json poly;
    poly.set("coefficients", coefs);
    poly.set("correlation", poly_corr);
    out.set("polyfit", poly);
    outputs.set("maths", out);
}

void add_segmentation_deterministic(Json &inputs, Json &outputs) {
    Json in;
    std::vector<double> constq = {
        1.0, 2.0, 4.0, 8.0,
        2.0, 4.0, 8.0, 16.0,
        3.0, 6.0, 9.0, 18.0
    };
    in.set("constq_matrix", reshape(constq, 3, 4));
    inputs.set("segmentation_deterministic", in);

    Json out;
    std::vector<double> mpeg_storage(3 * 5, 0.0);
    for (int r = 0; r < 3; ++r) std::copy_n(constq.data() + r * 4, 4, mpeg_storage.data() + r * 5);
    auto mpeg_ptrs = row_ptrs(mpeg_storage, 3, 5);
    mpeg7_constq(mpeg_ptrs.data(), 3, 4);
    out.set("mpeg7_constq", reshape(mpeg_storage, 3, 5));

    std::vector<double> chroma(3 * 2);
    auto constq_ptrs = row_ptrs(constq, 3, 4);
    auto chroma_ptrs = row_ptrs(chroma, 3, 2);
    cq2chroma(constq_ptrs.data(), 3, 4, 2, chroma_ptrs.data());
    out.set("cq_to_chroma", reshape(chroma, 3, 2));

    std::vector<int> labels = {0, 1, 1, 0, 1};
    std::vector<double> hist(labels.size() * 2);
    create_histograms(labels.data(), int(labels.size()), 2, 3, hist.data());
    out.set("create_histograms", reshape(hist, int(labels.size()), 2));
    outputs.set("segmentation_deterministic", out);
}

void add_segmentation_stochastic(Json &inputs, Json &outputs) {
    Json in;
    std::vector<double> hist = {
        0.2, 0.8,
        0.25, 0.75,
        0.8, 0.2,
        0.75, 0.25,
        0.7, 0.3
    };
    std::vector<double> features = {
        0.0, 0.1,
        0.2, 0.0,
        0.1, 0.2,
        5.0, 5.1,
        5.2, 5.0,
        5.1, 5.2
    };
    std::vector<double> constq = cqt_features(60, 24);
    std::vector<double> chroma_features_input = chroma_features(60, 24);
    in.set("histograms", reshape(hist, 5, 2));
    in.set("features", reshape(features, 6, 2));
    in.set("chroma_features", reshape(chroma_features_input, 60, 24));
    in.set("constq_features", reshape(constq, 60, 24));
    std::vector<std::vector<double>> audio_windows;
    int audio_window_size = int(0.05 * 11025 + 0.001);
    for (int i = 0; i < 5; ++i) {
        audio_windows.push_back(signal(size_t(audio_window_size), 0.15 * i));
    }
    in.set("stateful_audio_windows", audio_windows);
    inputs.set("segmentation_stochastic", in);

    Json out;
    std::vector<int> assignments(5);
    std::vector<double> schedule = {100.0, 50.0};
    std::srand(20260928);
    cluster_melt(hist.data(), 2, 5, schedule.data(), int(schedule.size()), 2, 0, assignments.data());
    out.set("cluster_melt", assignments);

    auto feature_ptrs = row_ptrs(features, 6, 2);
    std::vector<int> clustered(6);
    cluster_segment(clustered.data(), feature_ptrs.data(), 6, 2, 2, 3, 2, 0);
    out.set("cluster_segment", clustered);

    std::vector<int> chroma_segmented(60);
    auto chroma_feature_ptrs = row_ptrs(chroma_features_input, 60, 24);
    constq_segment(chroma_segmented.data(), chroma_feature_ptrs.data(), 60, 12, 24, FEATURE_TYPE_CHROMA, 6, 11, 3, 0);
    out.set("constq_segment_chroma", chroma_segmented);

    std::vector<double> constq_storage(60 * 25, 0.0);
    for (int r = 0; r < 60; ++r) std::copy_n(constq.data() + r * 24, 24, constq_storage.data() + r * 25);
    auto constq_ptrs = row_ptrs(constq_storage, 60, 25);
    std::vector<int> constq_segmented(60);
    constq_segment(constq_segmented.data(), constq_ptrs.data(), 60, 12, 24, FEATURE_TYPE_CONSTQ, 6, 11, 3, 0);
    out.set("constq_segment_constq", constq_segmented);

    ClusterMeltSegmenterParams params;
    params.featureType = FEATURE_TYPE_CHROMA;
    params.hopSize = 0.05;
    params.windowSize = 0.1;
    params.nHMMStates = 2;
    params.nclusters = 2;
    params.histogramLength = 3;
    params.neighbourhoodLimit = 2;
    ClusterMeltSegmenter feature_segmenter(params);
    feature_segmenter.initialise(11025);
    feature_segmenter.setFeatures(reshape(features, 6, 2));
    feature_segmenter.segment();
    Json stateful_feature;
    stateful_feature.set("windowsize", feature_segmenter.getWindowsize());
    stateful_feature.set("hopsize", feature_segmenter.getHopsize());
    stateful_feature.set("n_segment_types", feature_segmenter.getNSegmentTypes());
    stateful_feature.set("default_segmentation", segmentation_json(feature_segmenter.getSegmentation()));
    feature_segmenter.clear();
    feature_segmenter.setFeatures(reshape(features, 6, 2));
    feature_segmenter.segment(2);
    stateful_feature.set("after_clear_reuse_segmentation", segmentation_json(feature_segmenter.getSegmentation()));
    out.set("stateful_feature", stateful_feature);

    ClusterMeltSegmenterParams factory_params;
    factory_params.featureType = FEATURE_TYPE_CHROMA;
    factory_params.nclusters = 2;
    factory_params.histogramLength = 5;
    ClusterMeltSegmenter factory_segmenter(factory_params);
    factory_segmenter.initialise(11025);
    factory_segmenter.setFeatures(reshape(features, 6, 2));
    factory_segmenter.segment(2);
    out.set("factory_cluster_melt_segmenter", segmentation_json(factory_segmenter.getSegmentation()));

    ClusterMeltSegmenterParams audio_params;
    audio_params.featureType = FEATURE_TYPE_MFCC;
    audio_params.hopSize = 0.05;
    audio_params.windowSize = 0.05;
    audio_params.fmax = 5000;
    audio_params.nHMMStates = 2;
    audio_params.nclusters = 2;
    audio_params.histogramLength = 3;
    audio_params.neighbourhoodLimit = 2;
    ClusterMeltSegmenter audio_segmenter(audio_params);
    audio_segmenter.initialise(11025);
    int win = audio_segmenter.getWindowsize();
    for (int i = 0; i < 5; ++i) {
        auto audio = audio_windows[size_t(i)];
        audio_segmenter.extractFeatures(audio.data(), int(audio.size()));
    }
    audio_segmenter.segment(2);
    Json stateful_audio;
    stateful_audio.set("windowsize", win);
    stateful_audio.set("hopsize", audio_segmenter.getHopsize());
    stateful_audio.set("n_segment_types", audio_segmenter.getNSegmentTypes());
    stateful_audio.set("segmentation", segmentation_json(audio_segmenter.getSegmentation()));
    out.set("stateful_audio", stateful_audio);
    outputs.set("segmentation_stochastic", out);
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    Json inputs;
    Json outputs;
    add_base(inputs, outputs);
    add_rate(inputs, outputs);
    add_conditioning(inputs, outputs);
    add_maths(inputs, outputs);
    add_segmentation_deterministic(inputs, outputs);
    add_segmentation_stochastic(inputs, outputs);
    Json root;
    root.set("inputs", inputs);
    root.set("outputs", outputs);
    root.save(argv[1]);
    return 0;
}
