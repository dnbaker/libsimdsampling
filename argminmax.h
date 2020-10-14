#pragma once

#ifndef __cplusplus
#include <stdint.h>
#include <stdlib.h>
#else
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#endif

enum ArgReduction {
    ARGMIN,
    ARGMAX
};

#ifdef __cplusplus
using std::uint64_t;
using std::size_t;
extern "C" {
#endif
uint64_t fargsel(const float *weights, size_t n, enum ArgReduction ar);
uint64_t dargsel(const double *weights, size_t n, enum ArgReduction ar);
#ifdef __cplusplus
}

namespace reservoir_simd {
template<typename T>
static inline uint64_t argsel(const T *weights, size_t n, ArgReduction ar) {
    if(ar == ARGMIN) {
        return std::min_element(weights, weights + n) - weights;
    }
    return std::max_element(weights, weights + n) - weights;
}
template<> uint64_t argsel<double>(const double *weights, size_t n, ArgReduction ar) {
    return dargsel(weights, n, ar);
}
template<> uint64_t argsel<float>(const float *weights, size_t n, ArgReduction ar) {
    return fargsel(weights, n, ar);
}

template<typename T>
static inline uint64_t argmax(const T *weights, size_t n) {
    return std::max_element(weights, weights + n) - weights;
}
template<typename T>
static inline uint64_t argmin(const T *weights, size_t n) {
    return std::min_element(weights, weights + n) - weights;
}

template<> uint64_t argmin<double>(const double *weights, size_t n) {
    return dargsel(weights, n, ARGMIN);
}
template<> uint64_t argmin<float>(const float *weights, size_t n) {
    return fargsel(weights, n, ARGMIN);
}
template<> uint64_t argmax<double>(const double *weights, size_t n) {
    return dargsel(weights, n, ARGMAX);
}
template<> uint64_t argmax<float>(const float *weights, size_t n) {
    return fargsel(weights, n, ARGMAX);
}
template<typename Container>
INLINE uint64_t argmax(const Container &x) {
    return argmax(x.data(), x.size());
}

template<typename Container>
INLINE uint64_t argmin(const Container &x) {
    return argmin(x.data(), x.size());
}

}

#endif
