#ifndef SIMD_SAMPLING_H
#define SIMD_SAMPLING_H
#include <x86intrin.h>
#include <limits>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>
#include "macros.h"


using std::uint64_t;

uint64_t double_simd_sampling(const double *weights, size_t n, uint64_t seed=0);
uint64_t float_simd_sampling(const float *weights, size_t n, uint64_t seed=0);

template<typename FT>
inline uint64_t simd_sampling(const FT *weights, size_t n, uint64_t seed=0) {
    throw std::runtime_error(std::string("SIMD Sampling not implemented for type ") + __PRETTY_FUNCTION__);
}
template<> inline uint64_t simd_sampling<double>(const double *weights, size_t n, uint64_t seed) {
    return double_simd_sampling(weights, n, seed);
}
template<> inline uint64_t simd_sampling<float>(const float *weights, size_t n, uint64_t seed) {
    return float_simd_sampling(weights, n, seed);
}

template<typename Container, typename=typename std::enable_if<!std::is_pointer<Container>::value>::type>
static INLINE uint64_t simd_sampling(const Container &x, uint64_t seed=0) {
    return simd_sampling(x.data(), x.size(), seed);
}

#endif
