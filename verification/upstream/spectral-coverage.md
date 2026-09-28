# Spectral Upstream Parity Coverage

This verification set compares the public `pyqmdsp.spectral` surface against a
standalone executable linked to pristine `extern/qm-dsp` sources. The executable
does not import Python bindings and does not link package build patches.

Golden fixture: `verification/upstream/fixtures/spectral.json`

Python replay: `tests/upstream/test_spectral.py`

## Covered APIs

| Python API | Upstream source | Fixture keys |
|---|---|---|
| `FFT.process(real, imag, inverse=False)` | `FFT::process(false, real, imag, ...)` | `outputs.fft.forward` |
| `FFT.process(..., inverse=True)` | `FFT::process(true, ...)` | `outputs.fft.inverse` |
| `FFT.process(real)` / `fft(real)` path | `FFT::process(false, real, nullptr, ...)` | `outputs.fft.real_only_forward` |
| `fft(real, imag)` convenience | `FFT::process(false, real, imag, ...)` | `outputs.convenience.fft` |
| `RealFFT.forward` | `FFTReal::forward` | `outputs.real_fft.forward` |
| `real_fft` convenience | `FFTReal::forward` | `outputs.convenience.real_fft` |
| `RealFFT.forward_magnitude` | `FFTReal::forwardMagnitude` | `outputs.real_fft.magnitude` |
| `RealFFT.inverse` | `FFTReal::inverse` | `outputs.real_fft.inverse` |
| `DCT.forward(unitary=False)` | `DCT::forward` | `outputs.dct.forward` |
| `DCT.forward(unitary=True)` | `DCT::forwardUnitary` | `outputs.dct.forward_unitary` |
| `dct(unitary=True)` convenience | `DCT::forwardUnitary` | `outputs.convenience.dct` |
| `DCT.inverse(unitary=False)` | `DCT::inverse` | `outputs.dct.inverse` |
| `DCT.inverse(unitary=True)` | `DCT::inverseUnitary` | `outputs.dct.inverse_unitary` |
| `idct(unitary=True)` convenience | `DCT::inverseUnitary` | `outputs.convenience.idct` |
| `ConstantQ` metadata | `ConstantQ::getK/getFFTLength/getHop/getQ` | `outputs.constant_q.{bins,fft_length,hop,q}` |
| `ConstantQ.process_frequency` | `ConstantQ::process(real, imag, ...)` | `outputs.constant_q.process_frequency` |
| `Chromagram` metadata | `Chromagram::getK/getFrameSize/getHopSize` | `outputs.chromagram.{bins,cq_bins,frame_size,hop_size}` |
| `Chromagram.process_time` | `Chromagram::process(data)` | `outputs.chromagram.time_none`, `time_unit_sum`, `time_unit_max` |
| `Chromagram.process_frequency` | `Chromagram::process(real, imag)` | `outputs.chromagram.frequency_none` |
| `Chromagram.unity_normalise` | `Chromagram::unityNormalise` | `outputs.chromagram.unity_normalise` |
| `Chromagram.kabs` | `Chromagram::kabs` | `outputs.chromagram.kabs` |
| `MFCC.process_time` | `MFCC::process(frame, out)` | `outputs.mfcc.time` |
| `MFCC.process_frequency` | `MFCC::process(real, imag, out)` | `outputs.mfcc.frequency` |
| `MFCC` metadata | `MFCC::getfftlength` and configured output size | `outputs.mfcc.{fft_length,output_size}` |
| default `MFCC` configuration | `MFCCConfig(16000)` defaults: `want_c0=true`, `logpower=1`, `HammingWindow` | `outputs.mfcc_default` |
| `PhaseVocoder.process_time` | `PhaseVocoder::processTimeDomain` | `outputs.phase_vocoder.time.*` |
| `PhaseVocoder.process_frequency` | `PhaseVocoder::processFrequencyDomain` | `outputs.phase_vocoder.frequency` |
| `PhaseVocoder.reset` | `PhaseVocoder::reset` before the third time-domain frame | `outputs.phase_vocoder.time.after_reset` |
| `PhaseVocoder` metadata | constructor size/hop and derived bins | `outputs.phase_vocoder.{size,hop,bins}` |
| `KeyMode.process` | `GetKeyMode::process` over a multi-hop stream | `outputs.key.keys` |
| `KeyMode.key_strengths` | `GetKeyMode::getKeyStrengths` after streaming | `outputs.key.strengths` |
| `KeyMode` metadata | `GetKeyMode::getBlockSize/getHopSize` | `outputs.key.{block_size,hop_size}` |
| `TonalEstimator.transform_chroma` / `tonal_transform` | `TonalEstimator::transform2TCS` | `outputs.tonal.transform` |
| `tonal_transform` convenience | `TonalEstimator::transform2TCS` | `outputs.convenience.tonal_transform` |
| `normalize_chroma` | `ChromaVector::normalizeL1` | `outputs.tonal.normalize_chroma` |
| `tonal_magnitude` | `TCSVector::magnitude` | `outputs.tonal.magnitude` |
| `TCSGram.add/vector_at/matrix` | `TCSGram::addTCSVector/getTCSVector` | `outputs.tcsgram.before_clear` |
| `TCSGram.size/time_at/duration` | `TCSGram::getSize/getTime/getDuration` | `outputs.tcsgram.before_clear`, `after_setter`, `after_clear` |
| `TCSGram.set_frame_duration_ms` | `TCSGram::setFrameDuration` | `outputs.tcsgram.after_setter` |
| `TCSGram.clear` | `TCSGram::clear` | `outputs.tcsgram.after_clear` |
| `ChangeDetection.process_tcsgram` | `ChangeDetectionFunction::process(TCSGram)` | `outputs.change_detection.width_0_tcsgram`, `width_2_tcsgram` |
| `ChangeDetection.process_matrix` | same upstream process after constructing a `TCSGram` from matrix rows | `outputs.change_detection.width_0_matrix`, `width_2_matrix` |
| `wavelet_names` | `Wavelet::getWaveletName` for every enum value | `outputs.wavelet_names` |
| `wavelet_filters(int)` | `Wavelet::createDecompositionFilters` for even enum values | `outputs.wavelet_filters` |
| `wavelet_filters(str)` | `Wavelet::createDecompositionFilters` selected by exact upstream name for odd enum values | `outputs.wavelet_filters` |

## Fixture Shape

The fixture root has two keys:

- `inputs`: original double inputs emitted by `oracle_spectral`, including all
  signal frames and FFT-domain frames needed by Python replay. These are replay
  material, not assertion targets.
- `outputs`: pristine qm-dsp outputs for every replayed public numerical method.

`tests/upstream/test_spectral.py` builds an `actual_outputs` tree and calls
shared `assert_matches(actual_outputs, expected["outputs"])`. This keeps the
report focused on algorithm outputs while still requiring every expected output
key to be consumed.

## Explicit Gaps

- `ConstantQ::process(const double *FFTData)` remains excluded from the Python
  API and this parity fixture. The pinned upstream implementation clears
  `m_CQdata` with `row < 2 * m_uK` while writing `m_CQdata[row + 1]`, producing
  an out-of-bounds write at the final row. The safe separate real/imag overload
  is covered.
- `TCSGram::normalize` is declared in the upstream header but has no definition
  in the pinned sources, so no executable parity target can link it.
- `TCSGram::setNumBins` is not exposed by the Python API because upstream
  storage uses fixed six-dimensional `TCSVector` values in the covered path.
- `printDebug` methods are omitted because they are diagnostic stdout helpers,
  not numerical outputs.
