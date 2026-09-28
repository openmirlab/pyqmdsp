# Direct upstream build; sources mirror qm-dsp Makefile.inc at the pinned revision.
cmake_minimum_required(VERSION 3.24)
project(pyqmdsp LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)
find_package(Threads REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/apply_source_patch.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/extern/qm-dsp/hmm/hmm.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/patches/hmm-free-allocations.patch"
  "${CMAKE_CURRENT_BINARY_DIR}/patched/hmm.c" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/apply_source_patch.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/extern/qm-dsp/ext/cblas/src/dlamch.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/patches/dlamch-native-warning.patch"
  "${CMAKE_CURRENT_BINARY_DIR}/patched/dlamch.c" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/apply_source_patch.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/extern/qm-dsp/dsp/segmentation/ClusterMeltSegmenter.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/patches/segmenter-free-mfcc.patch"
  "${CMAKE_CURRENT_BINARY_DIR}/patched/ClusterMeltSegmenter.cpp" COMMAND_ERROR_IS_FATAL ANY)
add_library(qmdsp STATIC
  extern/qm-dsp/base/Pitch.cpp
  extern/qm-dsp/base/KaiserWindow.cpp
  extern/qm-dsp/base/SincWindow.cpp
  extern/qm-dsp/dsp/chromagram/Chromagram.cpp
  extern/qm-dsp/dsp/chromagram/ConstantQ.cpp
  extern/qm-dsp/dsp/keydetection/GetKeyMode.cpp
  extern/qm-dsp/dsp/mfcc/MFCC.cpp
  extern/qm-dsp/dsp/onsets/DetectionFunction.cpp
  extern/qm-dsp/dsp/onsets/PeakPicking.cpp
  extern/qm-dsp/dsp/phasevocoder/PhaseVocoder.cpp
  extern/qm-dsp/dsp/rateconversion/Decimator.cpp
  extern/qm-dsp/dsp/rateconversion/DecimatorB.cpp
  extern/qm-dsp/dsp/rateconversion/Resampler.cpp
  extern/qm-dsp/dsp/rhythm/BeatSpectrum.cpp
  extern/qm-dsp/dsp/segmentation/cluster_melt.c
  ${CMAKE_CURRENT_BINARY_DIR}/patched/ClusterMeltSegmenter.cpp
  extern/qm-dsp/dsp/segmentation/cluster_segmenter.c
  extern/qm-dsp/dsp/segmentation/Segmenter.cpp
  extern/qm-dsp/dsp/signalconditioning/DFProcess.cpp
  extern/qm-dsp/dsp/signalconditioning/Filter.cpp
  extern/qm-dsp/dsp/signalconditioning/FiltFilt.cpp
  extern/qm-dsp/dsp/signalconditioning/Framer.cpp
  extern/qm-dsp/dsp/tempotracking/DownBeat.cpp
  extern/qm-dsp/dsp/tempotracking/TempoTrack.cpp
  extern/qm-dsp/dsp/tempotracking/TempoTrackV2.cpp
  extern/qm-dsp/dsp/tonal/ChangeDetectionFunction.cpp
  extern/qm-dsp/dsp/tonal/TCSgram.cpp
  extern/qm-dsp/dsp/tonal/TonalEstimator.cpp
  extern/qm-dsp/dsp/transforms/DCT.cpp
  extern/qm-dsp/dsp/transforms/FFT.cpp
  extern/qm-dsp/dsp/wavelet/Wavelet.cpp
  ${CMAKE_CURRENT_BINARY_DIR}/patched/hmm.c
  extern/qm-dsp/maths/Correlation.cpp
  extern/qm-dsp/maths/CosineDistance.cpp
  extern/qm-dsp/maths/KLDivergence.cpp
  extern/qm-dsp/maths/MathUtilities.cpp
  extern/qm-dsp/maths/pca/pca.c
  extern/qm-dsp/thread/Thread.cpp
  extern/qm-dsp/ext/kissfft/kiss_fft.c
  extern/qm-dsp/ext/kissfft/tools/kiss_fftr.c
  extern/qm-dsp/ext/clapack/src/dgetrf.c
  extern/qm-dsp/ext/clapack/src/dgetri.c
  extern/qm-dsp/ext/clapack/src/dgetf2.c
  extern/qm-dsp/ext/clapack/src/xerbla.c
  extern/qm-dsp/ext/clapack/src/dlaswp.c
  extern/qm-dsp/ext/clapack/src/dtrtri.c
  extern/qm-dsp/ext/clapack/src/ilaenv.c
  extern/qm-dsp/ext/clapack/src/iparmq.c
  extern/qm-dsp/ext/clapack/src/s_cat.c
  extern/qm-dsp/ext/clapack/src/s_copy.c
  extern/qm-dsp/ext/clapack/src/s_cmp.c
  extern/qm-dsp/ext/clapack/src/pow_di.c
  extern/qm-dsp/ext/clapack/src/ieeeck.c
  extern/qm-dsp/ext/clapack/src/i_nint.c
  extern/qm-dsp/ext/clapack/src/dtrti2.c
  extern/qm-dsp/ext/clapack/src/f77_aloc.c
  extern/qm-dsp/ext/clapack/src/exit_.c
  extern/qm-dsp/ext/cblas/src/dgemm.c
  extern/qm-dsp/ext/cblas/src/ddot.c
  extern/qm-dsp/ext/cblas/src/dgemv.c
  extern/qm-dsp/ext/cblas/src/dswap.c
  extern/qm-dsp/ext/cblas/src/dtrsm.c
  extern/qm-dsp/ext/cblas/src/dger.c
  extern/qm-dsp/ext/cblas/src/idamax.c
  extern/qm-dsp/ext/cblas/src/dscal.c
  extern/qm-dsp/ext/cblas/src/dtrmm.c
  extern/qm-dsp/ext/cblas/src/lsame.c
  ${CMAKE_CURRENT_BINARY_DIR}/patched/dlamch.c
  extern/qm-dsp/ext/cblas/src/dtrmv.c
  extern/qm-dsp/ext/cblas/src/cblas_globals.c
  extern/qm-dsp/ext/cblas/src/cblas_dgemm.c
  extern/qm-dsp/ext/cblas/src/cblas_ddot.c
  extern/qm-dsp/ext/cblas/src/cblas_xerbla.c
)
target_include_directories(qmdsp PUBLIC extern/qm-dsp PRIVATE extern/qm-dsp/hmm extern/qm-dsp/dsp/segmentation extern/qm-dsp/ext/kissfft extern/qm-dsp/ext/kissfft/tools extern/qm-dsp/ext/clapack/include extern/qm-dsp/ext/cblas/include)
target_compile_definitions(qmdsp PRIVATE kiss_fft_scalar=double NO_BLAS_WRAP ADD_)
if(WIN32)
  # Public upstream headers use M_PI and are also compiled by the bindings.
  target_compile_definitions(qmdsp PUBLIC _USE_MATH_DEFINES NOMINMAX)
else()
  target_compile_definitions(qmdsp PRIVATE USE_PTHREADS)
endif()
target_link_libraries(qmdsp PUBLIC Threads::Threads)
# Retain the exact upstream class declaration without Polyfit.h's non-inline definitions.
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/extern/qm-dsp/maths/Polyfit.h" POLYFIT_HEADER)
string(FIND "${POLYFIT_HEADER}" "// some utility functions" POLYFIT_DECL_END)
if(POLYFIT_DECL_END LESS 0)
  message(FATAL_ERROR "Upstream Polyfit declaration boundary changed")
endif()
string(SUBSTRING "${POLYFIT_HEADER}" 0 ${POLYFIT_DECL_END} POLYFIT_DECL)
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/generated/polyfit_decl.h" "${POLYFIT_DECL}\n#endif\n")
nanobind_add_module(_native NB_STATIC native/module.cpp native/rhythm.cpp native/spectral.cpp native/utilities.cpp native/statistics.cpp)
target_include_directories(_native PRIVATE native "${CMAKE_CURRENT_BINARY_DIR}/generated")
target_link_libraries(_native PRIVATE qmdsp)
target_compile_definitions(_native PRIVATE PYQMDSP_COMPILER="${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
install(TARGETS _native LIBRARY DESTINATION pyqmdsp RUNTIME DESTINATION pyqmdsp)
