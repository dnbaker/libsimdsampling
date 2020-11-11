#pragma once

#ifndef SIMD_SAMPLING_API
#define SIMD_SAMPLING_API
#endif

#ifndef __cplusplus
#include <stdint.h>
#include <stdlib.h>
#else
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <queue>
#endif
#include "macros.h"

enum ArgReduction {
    ARGMIN,
    ARGMAX
};

#ifdef __cplusplus
using std::uint64_t;
using std::size_t;
using std::ptrdiff_t;
extern "C" {
#endif
SIMD_SAMPLING_API uint64_t fargsel(const float *weights, size_t n, enum ArgReduction ar, int mt);
SIMD_SAMPLING_API uint64_t fargmin(const float *weights, size_t n, int mt);
SIMD_SAMPLING_API uint64_t fargmax(const float *weights, size_t n, int mt);
SIMD_SAMPLING_API uint64_t dargsel(const double *weights, size_t n, enum ArgReduction ar, int mt);
SIMD_SAMPLING_API uint64_t dargmin(const double *weights, size_t n, int mt);
SIMD_SAMPLING_API uint64_t dargmax(const double *weights, size_t n, int mt);
SIMD_SAMPLING_API uint64_t fargmin_st(const float *weights, size_t n);
SIMD_SAMPLING_API uint64_t fargmax_st(const float *weights, size_t n);
SIMD_SAMPLING_API uint64_t dargmin_st(const double *weights, size_t n);
SIMD_SAMPLING_API uint64_t dargmax_st(const double *weights, size_t n);
SIMD_SAMPLING_API uint64_t fargmin_mt(const float *weights, size_t n);
SIMD_SAMPLING_API uint64_t fargmax_mt(const float *weights, size_t n);
SIMD_SAMPLING_API uint64_t dargmin_mt(const double *weights, size_t n);
SIMD_SAMPLING_API uint64_t dargmax_mt(const double *weights, size_t n);
SIMD_SAMPLING_API ptrdiff_t fargsel_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar, int mt);
SIMD_SAMPLING_API ptrdiff_t dargsel_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar, int mt);
SIMD_SAMPLING_API ptrdiff_t fargsel_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar);
SIMD_SAMPLING_API ptrdiff_t dargsel_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar);
SIMD_SAMPLING_API ptrdiff_t fargsel_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar);
SIMD_SAMPLING_API ptrdiff_t dargsel_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar);
SIMD_SAMPLING_API ptrdiff_t fargmin_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t dargmin_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t fargmin_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t dargmin_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t fargmax_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t dargmax_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t fargmax_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t dargmax_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret);
SIMD_SAMPLING_API ptrdiff_t dargmin_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt);
SIMD_SAMPLING_API ptrdiff_t fargmin_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt);
SIMD_SAMPLING_API ptrdiff_t dargmax_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt);
SIMD_SAMPLING_API ptrdiff_t fargmax_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt);
#ifdef __cplusplus
}
#include <vector>
#include <stdexcept>

namespace reservoir_simd {
template<typename T>
static inline uint64_t argsel(const T *weights, size_t n, ArgReduction ar, int mt=false) {
    return (ar == ARGMIN ? std::min_element(weights, weights + n)
                         : std::max_element(weights, weights + n))
           - weights;
}
template<> inline uint64_t argsel<double>(const double *weights, size_t n, ArgReduction ar, int mt) {
    return dargsel(weights, n, ar, mt);
}
template<> inline uint64_t argsel<float>(const float *weights, size_t n, ArgReduction ar, int mt) {
    return fargsel(weights, n, ar, mt);
}

template<typename T>
static inline uint64_t argmax(const T *weights, size_t n, int mt=false) {
    return std::max_element(weights, weights + n) - weights;
}
template<typename T>
static inline uint64_t argmin(const T *weights, size_t n, int mt=false) {
    return std::min_element(weights, weights + n) - weights;
}

template<> inline uint64_t argmin<double>(const double *weights, size_t n, int mt) {
    return dargsel(weights, n, ARGMIN, mt);
}
template<> inline uint64_t argmin<float>(const float *weights, size_t n, int mt) {
    return fargsel(weights, n, ARGMIN, mt);
}
template<> inline uint64_t argmax<double>(const double *weights, size_t n, int mt) {
    return dargsel(weights, n, ARGMAX, mt);
}
template<> inline uint64_t argmax<float>(const float *weights, size_t n, int mt) {
    return fargsel(weights, n, ARGMAX, mt);
}
template<typename Container>
INLINE uint64_t argmax(const Container &x, bool mt=false) {
    return argmax(x.data(), x.size(), mt);
}

template<typename Container>
INLINE uint64_t argmin(const Container &x, bool mt=false) {
    return argmin(x.data(), x.size(), mt);
}

template<typename T>
INLINE ptrdiff_t argsel(T *ptr, size_t n, ptrdiff_t k, uint64_t *ret, ArgReduction ar, bool mt=false) {
    if(mt) throw std::invalid_argument("mt only available for float/double");
    if(ar == ARGMAX) {
        std::priority_queue<std::pair<T, ptrdiff_t>, std::vector<std::pair<T, ptrdiff_t>>, std::greater<std::pair<T, ptrdiff_t>>> pq;
        for(auto p = ptr; p != ptr + n; ++p) {
            if(pq.size() < k) pq.push(*p, p - ptr);
            else if(*p > pq.top()) {
                pq.pop();
                pq.push(*p, p - ptr);
            }
        }
        while(pq.size()) {
            ret[pq.size() - 1] = pq.top().second;
            pq.pop();
        }
    } else {
        std::priority_queue<std::pair<T, ptrdiff_t>, std::vector<std::pair<T, ptrdiff_t>>, std::less<std::pair<T, ptrdiff_t>>> pq;
        for(auto p = ptr; p != ptr + n; ++p) {
            if(pq.size() < k) pq.push(*p, ptr - ptr);
            else if(*p < pq.top()) {
                pq.pop();
                pq.push(*p, p - ptr);
            }
        }
        while(pq.size()) {
            ret[pq.size() - 1] = pq.top().second;
            pq.pop();
        }
    }
    return n;
}

template<>
INLINE ptrdiff_t argsel<double>(double *ptr, size_t n, ptrdiff_t k, uint64_t *ret, ArgReduction ar, bool mt) {
    return dargsel_k(ptr, n, k, ret, ar, mt);
}
template<>
INLINE ptrdiff_t argsel<float>(float *ptr, size_t n, ptrdiff_t k, uint64_t *ret, ArgReduction ar, bool mt) {
    return fargsel_k(ptr, n, k, ret, ar, mt);
}

template<typename T>
inline std::vector<uint64_t> argsel(T *ptr, size_t n, ptrdiff_t k, ArgReduction ar, bool mt) {
    std::vector<uint64_t> ret(k);
    argsel(ptr, n, k, ret.data(), ar, mt);
    return ret;
}

} // namespace reservoir_simd

#endif
