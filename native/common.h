// Shared NumPy buffer ownership and validation for direct qm-dsp bindings.
// Reads: nanobind ndarray and STL adapters.
#pragma once
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/tuple.h>
#include <vector>
#include <cmath>
#include <stdexcept>
namespace nb = nanobind;
using namespace nb::literals;
using ArrayD = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;
inline std::vector<double> vector_input(const ArrayD &a) {
    if (a.ndim() != 1) throw std::invalid_argument("expected a one-dimensional array");
    std::vector<double> v(a.data(), a.data() + a.size());
    for (double x : v) if (!std::isfinite(x)) throw std::invalid_argument("array must contain finite values");
    return v;
}
inline nb::ndarray<nb::numpy, double> array_output(std::vector<double> values) {
    auto *v = new std::vector<double>(std::move(values));
    nb::capsule owner(v, [](void *p) noexcept { delete static_cast<std::vector<double> *>(p); });
    return nb::ndarray<nb::numpy, double>(v->data(), {v->size()}, owner);
}
void bind_rhythm(nb::module_ &);
void bind_spectral(nb::module_ &);
void bind_utilities(nb::module_ &);
