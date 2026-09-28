// Native registration and build provenance. Reads: per-domain bindings.
#include "common.h"
NB_MODULE(_native, m) {
    m.doc() = "Direct qm-dsp native algorithms; no Vamp dependency.";
    m.def("engine_info", []() {
        nb::dict d;
        d["engine"] = "qm-dsp";
        d["source_revision"] = "e34a3cc188332ed7c33cd9257ef164de5b587191";
        d["fft"] = "kissfft-double";
        d["fast_math"] = false;
        return d;
    });
    bind_rhythm(m); bind_spectral(m); bind_utilities(m);
}
