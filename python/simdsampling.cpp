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
    m.def("sample", [](py::array data, Py_ssize_t seed, Py_ssize_t k) -> py::object {
        auto inf = data.request();
        if(inf.format.size() != 1 || (inf.format[0] != 'd' && inf.format[0] != 'f')) throw std::invalid_argument("bad format");
        if(k <= 0) throw std::invalid_argument("k must be > 0");
        if(k == 1) {
            Py_ssize_t ret;
            if(inf.format[0] == 'd') {
                ret = dsimd_sample((double *)inf.ptr, inf.size, seed);
            } else {
                ret = fsimd_sample((float *)inf.ptr, inf.size, seed);
            }
            return py::int_(ret);
        }
        const std::string dt = size2dtype(inf.size);
        py::array ret(py::dtype(dt), std::vector<Py_ssize_t>{{Py_ssize_t(k)}});
        auto retinf = ret.request();
        std::vector<uint64_t> i64r(k);
        if(inf.format[0] == 'd') {
            dsimd_sample_k((double *)inf.ptr, inf.size, k, i64r.data(), seed);
        } else {
            fsimd_sample_k((float *)inf.ptr, inf.size, k, i64r.data(), seed);
        }
        switch(dt[0]) {
           case 'L': std::copy(i64r.begin(), i64r.end(), (uint64_t *)retinf.ptr); break;
           case 'I': std::copy(i64r.begin(), i64r.end(), (uint32_t *)retinf.ptr); break;
           case 'H': std::copy(i64r.begin(), i64r.end(), (uint16_t *)retinf.ptr); break;
           case 'B': std::copy(i64r.begin(), i64r.end(), (uint8_t *)retinf.ptr);  break;
           default: throw std::invalid_argument("Internal dtype failure");
        }
        return ret;
    }, py::arg("data"), py::arg("seed") = 0, py::arg("k") = 1)
    .def("argmin", [](py::array data) {
        auto inf = data.request();
        if(inf.format.size() != 1) throw std::invalid_argument("bad format");
        Py_ssize_t ret;
        switch(inf.format.front()) {
            case 'd': ret = dargmin((double *)inf.ptr, inf.size); break;
            case 'f': ret = fargmin((float *)inf.ptr, inf.size);  break;
            default: throw std::invalid_argument("bad format");
        }
        return ret;
    })
    .def("argmax", [](py::array data) {
        auto inf = data.request();
        if(inf.format.size() != 1) throw std::invalid_argument("bad format");
        Py_ssize_t ret;
        switch(inf.format.front()) {
            case 'd': ret = dargmax((double *)inf.ptr, inf.size); break;
            case 'f': ret = fargmax((float *)inf.ptr, inf.size);  break;
            default: throw std::invalid_argument("bad format");
        }
        return ret;
    });
    m.doc() = "Python bindings for libsimdsampling; sample 1 or k items from a weighted probability distribution or compute vectorized/parallelized argmin/max";
}
