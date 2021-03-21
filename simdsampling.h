#ifndef SIMD_SAMPLING_H
#define SIMD_SAMPLING_H

#ifndef SIMD_SAMPLING_API
#define SIMD_SAMPLING_API
#endif

#include <x86intrin.h>
#include "macros.h"

#ifdef __cplusplus
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#else
#include <stdint.h>
#endif


#define LIB_SIMDSAMPLING_MAJOR 0
#define LIB_SIMDSAMPLING_MINOR 3
#define LIB_SIMDSAMPLING_REVISION 1
#define LIB_SIMDSAMPLING_VERSION ((LIB_SIMDSAMPLING_MAJOR << 16) | (LIB_SIMDSAMPLING_MINOR << 8)| LIB_SIMDSAMPLING_REVISION)

#ifndef SIMD_SAMPLING_HIGH_PRECISION
#define SIMD_SAMPLING_HIGH_PRECISION 1
#endif

enum SampleFmt {
    WITH_REPLACEMENT = 1,
    USE_EXPONENTIAL_SKIPS = 2,
    NEITHER = 0,
    BOTH = WITH_REPLACEMENT | USE_EXPONENTIAL_SKIPS
};

#ifdef __cplusplus
extern "C" {
#endif
SIMD_SAMPLING_API uint64_t fsimd_sample(const float *weights, size_t n, uint64_t seed, enum SampleFmt fmt);
SIMD_SAMPLING_API uint64_t dsimd_sample(const double *weights, size_t n, uint64_t seed, enum SampleFmt fmt);

SIMD_SAMPLING_API int fsimd_sample_k(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt);
SIMD_SAMPLING_API int dsimd_sample_k(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt);
// Return value: the number of selected elements; min(n, k)
// uint64_t *ret: pointer to which to write selected elements

SIMD_SAMPLING_API int simd_sample_get_version();
SIMD_SAMPLING_API int simd_sample_get_major_version();
SIMD_SAMPLING_API int simd_sample_get_minor_version();
SIMD_SAMPLING_API int simd_sample_get_revision_version();

#ifdef __cplusplus
} // extern C

#include <vector>
namespace reservoir_simd {
using std::uint64_t;

// Sample 1
template<typename FT>
inline uint64_t sample(const FT *weights, size_t n, uint64_t seed=0, enum SampleFmt fmt=NEITHER) {
    throw std::runtime_error(std::string("SIMD Sampling not implemented for type ") + __PRETTY_FUNCTION__);
}
template<> inline uint64_t sample<double>(const double *weights, size_t n, uint64_t seed, enum SampleFmt fmt) {
    return dsimd_sample(weights, n, seed, fmt);
}
template<> inline uint64_t sample<float>(const float *weights, size_t n, uint64_t seed, enum SampleFmt fmt) {
    return fsimd_sample(weights, n, seed, fmt);
}

template<typename Container, typename=typename std::enable_if<!std::is_pointer<Container>::value>::type>
static INLINE uint64_t sample(const Container &x, uint64_t seed=0, enum SampleFmt fmt=NEITHER) {
    return sample(x.data(), x.size(), seed);
}

// Sample k
template<typename FT>
inline std::vector<uint64_t> sample_k(const FT *weights, size_t n, int k, uint64_t seed=0, enum SampleFmt fmt=NEITHER) {
    throw std::runtime_error(std::string("SIMD Sampling not implemented for type ") + __PRETTY_FUNCTION__);
}
template<> inline std::vector<uint64_t> sample_k<double>(const double *weights, size_t n, int k, uint64_t seed, enum SampleFmt fmt) {
    std::vector<uint64_t> ret(k);
    int kret = dsimd_sample_k(weights, n, k, ret.data(), seed, fmt);
    if(kret != k) {
        std::fprintf(stderr, "Return %u vs %u items\n", kret, k);
        ret.resize(kret);
    }
    return ret;
}
template<> inline std::vector<uint64_t> sample_k<float>(const float *weights, size_t n, int k, uint64_t seed, enum SampleFmt fmt) {
    std::vector<uint64_t> ret(k);
    int kret = fsimd_sample_k(weights, n, k, ret.data(), seed, fmt);
    if(kret != k) ret.resize(kret);
    return ret;
}

template<typename Container, typename=typename std::enable_if<!std::is_pointer<Container>::value>::type>
static INLINE std::vector<uint64_t> sample_k(const Container &x, int k, uint64_t seed=0, enum SampleFmt fmt=NEITHER) {
    return sample_k(x.data(), x.size(), k, seed, fmt);
}

template<typename T>
INLINE int sample_k(const T *weights, size_t n, int k, uint64_t *ret, uint64_t seed=0, enum SampleFmt fmt=NEITHER) {
    throw std::runtime_error("Not Implemented");
}
template<> inline int sample_k<double>(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt) {
    return dsimd_sample_k(weights, n, k, ret, seed, fmt);
}
template<> inline int sample_k<float>(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt) {
    return fsimd_sample_k(weights, n, k, ret, seed, fmt);
}

} // namespace reservoir_simd

#endif // ifdef __cplusplus

#endif
