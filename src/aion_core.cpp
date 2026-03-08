/*
 * Aqwel-Aion - C++ extension for fast numerical operations
 *
 * Optional native module: if built, aion._core uses this for fast_sum.
 * Requires pybind11 and a C++14 compiler at build time.
 *
 * Author: Aqwel AI Team
 * License: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

double fast_sum(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_sum: expected 1D array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    const size_t n = buf.shape[0];
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += ptr[i];
    }
    return s;
}

PYBIND11_MODULE(_aion_core, m) {
    m.def("fast_sum", &fast_sum,
          "Sum of a 1D float64 array (C++ implementation for speed).");
}
