# Rhythm, Onset, and Tempo Coverage

Source of truth: `extern/qm-dsp` at `e34a3cc188332ed7c33cd9257ef164de5b587191`.

## `dsp/onsets/DetectionFunction`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `DetectionFunction(DFConfig)` | `rhythm.DetectionFunction(...)` | Constructor exposes every `DFConfig` field. |
| `~DetectionFunction()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `processTimeDomain(const double*)` | `DetectionFunction.process_time_domain(samples)` | Requires one-dimensional float64 input of exactly `frame_length`. |
| `processFrequencyDomain(const double*, const double*)` | `DetectionFunction.process_frequency_domain(reals, imags)` | Requires `frame_length / 2 + 1` real and imaginary bins. |
| `getSpectrumMagnitude()` | `DetectionFunction.spectrum_magnitude()` | Returns a copied NumPy array, avoiding raw pointer ownership exposure. |

Constants `DF_HFC`, `DF_SPECDIFF`, `DF_PHASEDEV`, `DF_COMPLEXSD`, and `DF_BROADBAND`
are exported from `pyqmdsp.rhythm`.

## `dsp/onsets/PeakPicking`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `PeakPicking(PPickParams)` | `rhythm.PeakPicking(...)` | Constructor exposes length, tau, alpha, cutoff, low-pass order/coefs, threshold windows, quadratic threshold coefficients, and delta. |
| `~PeakPicking()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `process(double*, int, std::vector<int>&)` | `PeakPicking.process(detection_function)` | Returns `{"onsets", "processed"}` because upstream mutates the detection-function buffer while returning onset indices. |
| `PPWinThresh`, `QFitThresh`, `PPickParams` fields | Constructor args plus `.parameters` | Structs are infrastructure for construction, not separate Python object types. |

## `dsp/tempotracking/TempoTrack`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `TempoTrack(TTParams)` | `rhythm.TempoTrack(...)` | Constructor exposes all `TTParams` fields. |
| `~TempoTrack()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `process(std::vector<double>, std::vector<double>*)` | `TempoTrack.process(detection_function)` | Returns `{"beats", "tempo"}`; always requests `tempoReturn` so the optional upstream output is reachable. |
| `WinThresh`, `TTParams` fields | Constructor args plus `.parameters` | Structs are infrastructure for construction, not separate Python object types. |

## `dsp/tempotracking/TempoTrackV2`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `TempoTrackV2(float, int)` | `rhythm.TempoTrackV2(sample_rate, df_increment)` | Preserves stateful object construction. |
| `~TempoTrackV2()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `calculateBeatPeriod(df, beatPeriod, tempi)` | `TempoTrackV2.calculate_beat_period(detection_function)` | Uses upstream defaults `input_tempo=120`, `constrain_tempo=False`. |
| `calculateBeatPeriod(df, beatPeriod, tempi, inputtempo, constraintempo)` | `TempoTrackV2.calculate_beat_period(..., input_tempo=..., constrain_tempo=...)` | Full overload exposed. |
| `calculateBeats(df, beatPeriod, beats)` | `TempoTrackV2.calculate_beats(detection_function, beat_period)` | Uses upstream defaults `alpha=0.9`, `tightness=4.0`. |
| `calculateBeats(df, beatPeriod, beats, alpha, tightness)` | `TempoTrackV2.calculate_beats(..., alpha=..., tightness=...)` | Full overload exposed. |

## `dsp/tempotracking/DownBeat`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `DownBeat(float, size_t, size_t)` | `rhythm.DownBeat(original_sample_rate, decimation_factor, df_increment)` | Constructor validates documented factor and increment constraints before native construction. |
| `~DownBeat()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `setBeatsPerBar(int)` | `DownBeat.set_beats_per_bar(beats_per_bar)` | Exposed. |
| `findDownBeats(const float*, size_t, const vector<double>&, vector<int>&)` | `DownBeat.find_down_beats(audio, beats_frames)` | Exposed. Audio is accepted as float64 and converted to native float. |
| `getBeatSD(vector<double>&)` | `DownBeat.beat_sd()` | Exposed as a copied NumPy array. |
| `pushAudioBlock(const float*)` | `DownBeat.push_audio_block(audio)` | Exposed; block length must match `df_increment`. |
| `getBufferedAudio(size_t&)` | `DownBeat.buffered_audio()` | Exposed as float64 NumPy copy to avoid raw pointer ownership. |
| `resetAudioBuffer()` | `DownBeat.reset_audio_buffer()` | Exposed. |

## `dsp/rhythm/BeatSpectrum`

| Upstream public API | Python binding | Notes |
|---|---|---|
| `BeatSpectrum()` | `rhythm.BeatSpectrum()` | Exposed. |
| `~BeatSpectrum()` | Python object lifetime | Native destructor runs when the wrapped object is freed. |
| `process(const vector<vector<double>>& inmatrix)` | `BeatSpectrum.process(matrix)` | Exposed for two-dimensional float64 NumPy arrays. |

## Convenience API

| API | Binding | Notes |
|---|---|---|
| Tactus beat-tracking prototype | `rhythm.beats(audio, sample_rate)` | Uses qm-dsp `DetectionFunction` with `DF_COMPLEXSD` and `TempoTrackV2`, matching the prototype's uncentered framing, trailing-zero trim, first-two-DF-frame discard, and beat-time conversion. Returns `{"beats_s", "bpm", "tempo_curve"}` as NumPy arrays plus a float BPM. `TempoTrackV2` requires more than 640 detection-function frames after that trim/discard stage so its 512-frame window with 128-hop produces at least two tempo-analysis frames; shorter material returns empty beat arrays and `0.0` BPM rather than calling upstream with unsafe zero periods. |

## Infrastructure intentionally not exposed

Private helper methods in these classes, raw pointer buffers, destructors as callable methods,
and construction structs as mutable Python objects are not separate public bindings. Their public
effects are reachable through constructors, methods, return dictionaries, `.parameters`, and NumPy
copies. Low-level threading, allocation, FFT internals, filters, and math helpers are covered by
their owning binding domains rather than this rhythm map.
