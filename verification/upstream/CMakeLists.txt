# Standalone original-qm-dsp oracle. No binding sources or package patches.
# Reads: pinned upstream Makefile source inventory, oracle drivers, f2c guard.
cmake_minimum_required(VERSION 3.24)
project(qmdsp_original LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(QM "${CMAKE_CURRENT_SOURCE_DIR}/../../extern/qm-dsp" ABSOLUTE)
find_package(Threads REQUIRED)
add_library(qmdsp_original STATIC
  ${QM}/ext/clapack/src/dgetrf.c
  ${QM}/ext/clapack/src/dgetri.c
  ${QM}/ext/clapack/src/dgetf2.c
  ${QM}/ext/clapack/src/xerbla.c
  ${QM}/ext/clapack/src/dlaswp.c
  ${QM}/ext/clapack/src/dtrtri.c
  ${QM}/ext/clapack/src/ilaenv.c
  ${QM}/ext/clapack/src/iparmq.c
  ${QM}/ext/clapack/src/s_cat.c
  ${QM}/ext/clapack/src/s_copy.c
  ${QM}/ext/clapack/src/s_cmp.c
  ${QM}/ext/clapack/src/pow_di.c
  ${QM}/ext/clapack/src/ieeeck.c
  ${QM}/ext/clapack/src/i_nint.c
  ${QM}/ext/clapack/src/dtrti2.c
  ${QM}/ext/clapack/src/f77_aloc.c
  ${QM}/ext/clapack/src/exit_.c
  ${QM}/ext/cblas/src/dgemm.c
  ${QM}/ext/cblas/src/ddot.c
  ${QM}/ext/cblas/src/dgemv.c
  ${QM}/ext/cblas/src/dswap.c
  ${QM}/ext/cblas/src/dtrsm.c
  ${QM}/ext/cblas/src/dger.c
  ${QM}/ext/cblas/src/idamax.c
  ${QM}/ext/cblas/src/dscal.c
  ${QM}/ext/cblas/src/dtrmm.c
  ${QM}/ext/cblas/src/lsame.c
  ${QM}/ext/cblas/src/dlamch.c
  ${QM}/ext/cblas/src/dtrmv.c
  ${QM}/ext/cblas/src/cblas_globals.c
  ${QM}/ext/cblas/src/cblas_dgemm.c
  ${QM}/ext/cblas/src/cblas_ddot.c
  ${QM}/ext/cblas/src/cblas_xerbla.c
  ${QM}/base/Pitch.cpp
  ${QM}/base/KaiserWindow.cpp
  ${QM}/base/SincWindow.cpp
  ${QM}/dsp/chromagram/Chromagram.cpp
  ${QM}/dsp/chromagram/ConstantQ.cpp
  ${QM}/dsp/keydetection/GetKeyMode.cpp
  ${QM}/dsp/mfcc/MFCC.cpp
  ${QM}/dsp/onsets/DetectionFunction.cpp
  ${QM}/dsp/onsets/PeakPicking.cpp
  ${QM}/dsp/phasevocoder/PhaseVocoder.cpp
  ${QM}/dsp/rateconversion/Decimator.cpp
  ${QM}/dsp/rateconversion/DecimatorB.cpp
  ${QM}/dsp/rateconversion/Resampler.cpp
  ${QM}/dsp/rhythm/BeatSpectrum.cpp
  ${QM}/dsp/segmentation/cluster_melt.c
  ${QM}/dsp/segmentation/ClusterMeltSegmenter.cpp
  ${QM}/dsp/segmentation/cluster_segmenter.c
  ${QM}/dsp/segmentation/Segmenter.cpp
  ${QM}/dsp/signalconditioning/DFProcess.cpp
  ${QM}/dsp/signalconditioning/Filter.cpp
  ${QM}/dsp/signalconditioning/FiltFilt.cpp
  ${QM}/dsp/signalconditioning/Framer.cpp
  ${QM}/dsp/tempotracking/DownBeat.cpp
  ${QM}/dsp/tempotracking/TempoTrack.cpp
  ${QM}/dsp/tempotracking/TempoTrackV2.cpp
  ${QM}/dsp/tonal/ChangeDetectionFunction.cpp
  ${QM}/dsp/tonal/TCSgram.cpp
  ${QM}/dsp/tonal/TonalEstimator.cpp
  ${QM}/dsp/transforms/DCT.cpp
  ${QM}/dsp/transforms/FFT.cpp
  ${QM}/dsp/wavelet/Wavelet.cpp
  ${QM}/hmm/hmm.c
  ${QM}/maths/Correlation.cpp
  ${QM}/maths/CosineDistance.cpp
  ${QM}/maths/KLDivergence.cpp
  ${QM}/maths/MathUtilities.cpp
  ${QM}/maths/pca/pca.c
  ${QM}/thread/Thread.cpp
  ${QM}/ext/kissfft/kiss_fft.c
  ${QM}/ext/kissfft/tools/kiss_fftr.c
)
target_include_directories(qmdsp_original PUBLIC ${QM} PRIVATE ${QM}/ext/kissfft ${QM}/ext/kissfft/tools ${QM}/ext/clapack/include ${QM}/ext/cblas/include)
target_compile_definitions(qmdsp_original PRIVATE kiss_fft_scalar=double NO_BLAS_WRAP ADD_)
if(WIN32)
  target_compile_definitions(qmdsp_original PUBLIC _USE_MATH_DEFINES NOMINMAX)
else()
  target_compile_definitions(qmdsp_original PRIVATE USE_PTHREADS)
endif()
target_link_libraries(qmdsp_original PUBLIC Threads::Threads)
foreach(domain rhythm spectral utilities statistics)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/oracle_${domain}.cpp")
    add_executable(oracle_${domain} oracle_${domain}.cpp f2c_guard.c)
    target_link_libraries(oracle_${domain} PRIVATE qmdsp_original)
    target_include_directories(oracle_${domain} PRIVATE ${QM}/ext/clapack/include)
  endif()
endforeach()
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_library(fixed_time SHARED fixed_time.c)
endif()
