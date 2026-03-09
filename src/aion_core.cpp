/*
 * Aqwel-Aion - C++ extension for fast numerical operations
 *
 * Optional native module: if built, aion._core uses these for speed.
 * Requires pybind11 and a C++14 compiler at build time.
 *
 * Author: Aqwel AI Team
 * License: Apache-2.0
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <algorithm>
#include <limits>

namespace py = pybind11;

// ---- 1D reductions (contiguous double) ----

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

double fast_dot(py::array_t<double> a, py::array_t<double> b) {
    py::buffer_info ba = a.request(), bb = b.request();
    if (ba.ndim != 1 || bb.ndim != 1) {
        throw std::runtime_error("fast_dot: expected 1D arrays");
    }
    const size_t n = ba.shape[0];
    if (bb.shape[0] != n) {
        throw std::runtime_error("fast_dot: shape mismatch");
    }
    const double* pa = static_cast<const double*>(ba.ptr);
    const double* pb = static_cast<const double*>(bb.ptr);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += pa[i] * pb[i];
    }
    return s;
}

double fast_norm2(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_norm2: expected 1D array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    const size_t n = buf.shape[0];
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += ptr[i] * ptr[i];
    }
    return std::sqrt(s);
}

double fast_mean(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_mean: expected 1D array");
    }
    const size_t n = buf.shape[0];
    if (n == 0) {
        throw std::runtime_error("fast_mean: empty array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += ptr[i];
    }
    return s / static_cast<double>(n);
}

double fast_variance(py::array_t<double> arr, int ddof) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_variance: expected 1D array");
    }
    const size_t n = buf.shape[0];
    if (n <= static_cast<size_t>(ddof)) {
        throw std::runtime_error("fast_variance: n must be > ddof");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    double sum = 0.0, sum2 = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum += ptr[i];
        sum2 += ptr[i] * ptr[i];
    }
    double mean = sum / static_cast<double>(n);
    double cnt = static_cast<double>(n - ddof);
    return (sum2 / static_cast<double>(n) - mean * mean) * static_cast<double>(n) / cnt;
}

// ---- 1D index ----

int64_t fast_argmax(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_argmax: expected 1D array");
    }
    const size_t n = buf.shape[0];
    if (n == 0) {
        throw std::runtime_error("fast_argmax: empty array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    size_t best = 0;
    double best_val = ptr[0];
    for (size_t i = 1; i < n; ++i) {
        if (ptr[i] > best_val) {
            best_val = ptr[i];
            best = i;
        }
    }
    return static_cast<int64_t>(best);
}

// ---- 1D element-wise (return new array) ----

py::array_t<double> fast_relu(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_relu: expected 1D array");
    }
    const size_t n = buf.shape[0];
    const double* in_ptr = static_cast<const double*>(buf.ptr);
    auto out = py::array_t<double>(n);
    py::buffer_info obuf = out.request();
    double* out_ptr = static_cast<double*>(obuf.ptr);
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = in_ptr[i] > 0.0 ? in_ptr[i] : 0.0;
    }
    return out;
}

py::array_t<double> fast_softmax(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_softmax: expected 1D array");
    }
    const size_t n = buf.shape[0];
    if (n == 0) {
        throw std::runtime_error("fast_softmax: empty array");
    }
    const double* in_ptr = static_cast<const double*>(buf.ptr);
    double max_val = in_ptr[0];
    for (size_t i = 1; i < n; ++i) {
        if (in_ptr[i] > max_val) max_val = in_ptr[i];
    }
    auto out = py::array_t<double>(n);
    double* out_ptr = static_cast<double*>(out.request().ptr);
    double sum = 0.0;
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] = std::exp(in_ptr[i] - max_val);
        sum += out_ptr[i];
    }
    for (size_t i = 0; i < n; ++i) {
        out_ptr[i] /= sum;
    }
    return out;
}

py::array_t<double> fast_cumsum(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_cumsum: expected 1D array");
    }
    const size_t n = buf.shape[0];
    const double* in_ptr = static_cast<const double*>(buf.ptr);
    auto out = py::array_t<double>(n);
    double* out_ptr = static_cast<double*>(out.request().ptr);
    double s = 0.0;
    for (size_t i = 0; i < n; ++i) {
        s += in_ptr[i];
        out_ptr[i] = s;
    }
    return out;
}

// ---- Matrix-vector: (M, N) @ (N,) -> (M,) ----

py::array_t<double> fast_matrix_vector_mul(py::array_t<double> mat, py::array_t<double> vec) {
    py::buffer_info bm = mat.request(), bv = vec.request();
    if (bm.ndim != 2 || bv.ndim != 1) {
        throw std::runtime_error("fast_matrix_vector_mul: expected 2D matrix and 1D vector");
    }
    const size_t M = bm.shape[0];
    const size_t N = bm.shape[1];
    if (bv.shape[0] != N) {
        throw std::runtime_error("fast_matrix_vector_mul: matrix cols != vector size");
    }
    const double* m_ptr = static_cast<const double*>(bm.ptr);
    const double* v_ptr = static_cast<const double*>(bv.ptr);
    auto out = py::array_t<double>(M);
    double* out_ptr = static_cast<double*>(out.request().ptr);
    for (size_t i = 0; i < M; ++i) {
        double s = 0.0;
        for (size_t j = 0; j < N; ++j) {
            s += m_ptr[i * N + j] * v_ptr[j];
        }
        out_ptr[i] = s;
    }
    return out;
}

// ---- Binary search: lower_bound style (first index where arr[i] >= value) ----

int64_t fast_lower_bound(py::array_t<double> arr, double value) {
    py::buffer_info buf = arr.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("fast_lower_bound: expected 1D array");
    }
    const double* ptr = static_cast<const double*>(buf.ptr);
    const size_t n = buf.shape[0];
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t mid = lo + (hi - lo) / 2;
        if (ptr[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return static_cast<int64_t>(lo);
}

PYBIND11_MODULE(_aion_core, m) {
    m.def("fast_sum", &fast_sum,
          "Sum of a 1D float64 array.");
    m.def("fast_dot", &fast_dot,
          "Dot product of two 1D float64 arrays.");
    m.def("fast_norm2", &fast_norm2,
          "L2 norm of a 1D float64 array.");
    m.def("fast_mean", &fast_mean,
          "Mean of a 1D float64 array.");
    m.def("fast_variance", &fast_variance,
          py::arg("arr"), py::arg("ddof") = 0,
          "Variance of a 1D float64 array (ddof=0 population, ddof=1 sample).");
    m.def("fast_argmax", &fast_argmax,
          "Index of maximum value in a 1D float64 array.");
    m.def("fast_relu", &fast_relu,
          "ReLU(x) = max(0, x) element-wise; returns new 1D array.");
    m.def("fast_softmax", &fast_softmax,
          "Numerically stable softmax over 1D float64 array; returns new array.");
    m.def("fast_cumsum", &fast_cumsum,
          "Cumulative sum of a 1D float64 array; returns new array.");
    m.def("fast_matrix_vector_mul", &fast_matrix_vector_mul,
          "Matrix-vector product: (M, N) @ (N,) -> (M,).");
    m.def("fast_lower_bound", &fast_lower_bound,
          "First index i where arr[i] >= value (assumes ascending order).");
}
