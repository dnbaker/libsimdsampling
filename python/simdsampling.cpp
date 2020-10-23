#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "simdsampling.h"
#include "argminmax.h"

namespace py = pybind11;


std::string size2dtype(Py_ssize_t n) {
    if(n > 0xFFFFFFFFu) return "L";
    if(n > 0xFFFFu) return "I";
    if(n > 0xFFu) return "H";
    return "B";
}

PYBIND11_MODULE(simdsampling, m) {
    m.def("sample", [](py::array data, Py_ssize_t seed, Py_ssize_t k, bool with_rep, bool with_exp_skips) -> py::object {
        auto inf = data.request();
        if(inf.format.size() != 1 || (inf.format[0] != 'd' && inf.format[0] != 'f')) throw std::invalid_argument("bad format");
        if(k <= 0) throw std::invalid_argument("k must be > 0");
        SampleFmt fmt = NEITHER;
        if(with_rep) fmt = (SampleFmt)((int)fmt | WITH_REPLACEMENT);
        if(with_exp_skips) fmt = (SampleFmt)((int)fmt | USE_EXPONENTIAL_SKIPS);
        if(k == 1) {
            Py_ssize_t ret;
            if(inf.format[0] == 'd') {
                ret = dsimd_sample((double *)inf.ptr, inf.size, seed, fmt);
            } else {
                ret = fsimd_sample((float *)inf.ptr, inf.size, seed, fmt);
            }
            return py::int_(ret);
        }
        const std::string dt = size2dtype(inf.size);
        py::array ret(py::dtype(dt), std::vector<Py_ssize_t>{{Py_ssize_t(k)}});
        auto retinf = ret.request();
        std::vector<uint64_t> i64r(k);
        if(inf.format[0] == 'd') {
            dsimd_sample_k((double *)inf.ptr, inf.size, k, i64r.data(), seed, fmt);
        } else {
            fsimd_sample_k((float *)inf.ptr, inf.size, k, i64r.data(), seed, fmt);
        }
        switch(dt[0]) {
           case 'L': std::copy(i64r.begin(), i64r.end(), (uint64_t *)retinf.ptr); break;
           case 'I': std::copy(i64r.begin(), i64r.end(), (uint32_t *)retinf.ptr); break;
           case 'H': std::copy(i64r.begin(), i64r.end(), (uint16_t *)retinf.ptr); break;
           case 'B': std::copy(i64r.begin(), i64r.end(), (uint8_t *)retinf.ptr);  break;
           default: throw std::invalid_argument("Internal dtype failure");
        }
        return ret;
    }, py::arg("data"), py::arg("seed") = 0, py::arg("k") = 1, py::arg("with_rep") = false, py::arg("exp_skips") = false)
    .def("argmin", [](py::array data, Py_ssize_t k, int multithread) -> py::object {
        auto inf = data.request();
        if(inf.format.size() != 1) throw std::invalid_argument("bad format");
        if(k == 1) {
            Py_ssize_t ret;
            switch(inf.format.front()) {
                case 'd': ret = dargmin((double *)inf.ptr, inf.size, multithread); break;
                case 'f': ret = fargmin((float *)inf.ptr, inf.size, multithread);  break;
                default: throw std::invalid_argument("bad format");
            }
            return py::int_(ret);
        }
        py::array ret(py::dtype(size2dtype(inf.size)), std::vector<Py_ssize_t>{k});
        std::vector<uint64_t> tmp(k);
        auto retinf = ret.request();
        switch(inf.format.front()) {
            case 'd': dargmin_k((double *)inf.ptr, inf.size, k, tmp.data(), multithread); break;
            case 'f': fargmin_k((float *)inf.ptr, inf.size, k, tmp.data(), multithread); break;
            default: throw std::invalid_argument("argmin only accepts d and f");
        }
        switch(retinf.format.front()) {
            case 'L': std::copy(tmp.begin(), tmp.end(), (uint64_t *)retinf.ptr); break;
            case 'I': std::copy(tmp.begin(), tmp.end(), (uint32_t *)retinf.ptr); break;
            case 'H': std::copy(tmp.begin(), tmp.end(), (uint16_t *)retinf.ptr); break;
            case 'B': std::copy(tmp.begin(), tmp.end(), (uint8_t *)retinf.ptr); break;
            default: throw std::invalid_argument("internal argmin error");
        }
        return ret;
    }, py::arg("array"), py::arg("k") = 1, py::arg("mt") = false)
    .def("argmax", [](py::array data, Py_ssize_t k, int multithread) -> py::object {
        auto inf = data.request();
        if(inf.format.size() != 1) throw std::invalid_argument("bad format");
        if(k == 1) {
            Py_ssize_t ret;
            switch(inf.format.front()) {
                case 'd': ret = dargmax((double *)inf.ptr, inf.size, multithread); break;
                case 'f': ret = fargmax((float *)inf.ptr, inf.size, multithread);  break;
                default: throw std::invalid_argument("bad format");
            }
            return py::int_(ret);
        }
        py::array ret(py::dtype(size2dtype(inf.size)), std::vector<Py_ssize_t>{k});
        std::vector<uint64_t> tmp(k);
        auto retinf = ret.request();
        switch(inf.format.front()) {
            case 'd': dargmax_k((double *)inf.ptr, inf.size, k, tmp.data(), multithread); break;
            case 'f': fargmax_k((float *)inf.ptr, inf.size, k, tmp.data(), multithread); break;
            default: throw std::invalid_argument("argmax only accepts d and f");
        }
        switch(retinf.format.front()) {
            case 'L': std::copy(tmp.begin(), tmp.end(), (uint64_t *)retinf.ptr); break;
            case 'I': std::copy(tmp.begin(), tmp.end(), (uint32_t *)retinf.ptr); break;
            case 'H': std::copy(tmp.begin(), tmp.end(), (uint16_t *)retinf.ptr); break;
            case 'B': std::copy(tmp.begin(), tmp.end(), (uint8_t *)retinf.ptr); break;
            default: throw std::invalid_argument("internal argmax error");
        }
        return ret;
    }, py::arg("array"), py::arg("k") = 1, py::arg("mt") = false)
    .def("get_version", []() {return simd_sample_get_version();})
    .def("get_major_version", []() {return simd_sample_get_major_version();})
    .def("get_minor_version", []() {return simd_sample_get_minor_version();})
    .def("get_revision_version", []() {return simd_sample_get_revision_version();});
    m.doc() = "Python bindings for libsimdsampling; sample 1 or k items from a weighted probability distribution or compute vectorized/parallelized argmin/max";
}
