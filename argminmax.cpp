#include "omp.h"
#include "argminmax.h"
#include "x86intrin.h"
#include "aesctr/wy.h"
#include "ctz.h"
#include "macros.h"
#include "reds.h"
#include <limits>
#include <type_traits>


using namespace reservoir_simd;
// Forward declaratios of core kernels
// Single-sample
template<LoadFormat aln, ArgReduction AR, bool MT> uint64_t double_argsel_fmt(const double *weights, size_t n);
template<LoadFormat aln, ArgReduction AR, bool MT> uint64_t float_argsel_fmt(const float * weights, size_t n);

template<LoadFormat aln, ArgReduction AR, bool MT> ptrdiff_t double_argsel_k_fmt(const double * weights, size_t n, ptrdiff_t k, uint64_t *ret);
template<LoadFormat aln, ArgReduction AR, bool MT> ptrdiff_t float_argsel_k_fmt(const float * weights, size_t n, ptrdiff_t k, uint64_t *ret);


// TODO: add top-k selection methods
extern "C" {
SIMD_SAMPLING_API uint64_t fargsel(const float *weights, size_t n, ArgReduction ar, int mt)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    uint64_t ret;
    switch((int(aligned) << 2) | (int(ar == ARGMAX) << 1) | mt) {
        case 0: ret = float_argsel_fmt<UNALIGNED, ARGMIN, false>(weights, n); break;
        case 1: ret = float_argsel_fmt<UNALIGNED, ARGMIN, true>(weights, n); break;
        case 2: ret = float_argsel_fmt<UNALIGNED, ARGMAX, false>(weights, n); break;
        case 3: ret = float_argsel_fmt<UNALIGNED, ARGMAX, true>(weights, n); break;
        case 4: ret = float_argsel_fmt<ALIGNED, ARGMIN, false>(weights, n); break;
        case 5: ret = float_argsel_fmt<ALIGNED, ARGMIN, true>(weights, n); break;
        case 6: ret = float_argsel_fmt<ALIGNED, ARGMAX, false>(weights, n); break;
        case 7: ret = float_argsel_fmt<ALIGNED, ARGMAX, true>(weights, n); break;
        default: __builtin_unreachable();
    }
    return ret;
}

SIMD_SAMPLING_API uint64_t dargmin_mt(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMIN, true>(weights, n): double_argsel_fmt<ALIGNED, ARGMIN, true>(weights, n);
}
SIMD_SAMPLING_API uint64_t dargmin_st(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMIN, false>(weights, n): double_argsel_fmt<ALIGNED, ARGMIN, false>(weights, n);
}
SIMD_SAMPLING_API uint64_t dargmin(const double *weights, size_t n, int mt) {
    return mt ? dargmin_mt(weights, n): dargmin_st(weights, n);
}
SIMD_SAMPLING_API uint64_t fargmin_mt(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMIN, true>(weights, n): float_argsel_fmt<ALIGNED, ARGMIN, true>(weights, n);
}
SIMD_SAMPLING_API uint64_t fargmin_st(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMIN, false>(weights, n): float_argsel_fmt<ALIGNED, ARGMIN, false>(weights, n);
}

SIMD_SAMPLING_API uint64_t fargmin(const float *weights, size_t n, int mt) {
    return mt ? fargmin_mt(weights, n): fargmin_st(weights, n);
}


SIMD_SAMPLING_API uint64_t dargmax_mt(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMAX, true>(weights, n): double_argsel_fmt<ALIGNED, ARGMAX, true>(weights, n);
}
SIMD_SAMPLING_API uint64_t dargmax_st(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMAX, false>(weights, n): double_argsel_fmt<ALIGNED, ARGMAX, false>(weights, n);
}
SIMD_SAMPLING_API uint64_t dargmax(const double *weights, size_t n, int  mt) {
    return mt ? dargmax_mt(weights, n): dargmax_st(weights, n);
}
SIMD_SAMPLING_API uint64_t fargmax_mt(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMAX, true>(weights, n): float_argsel_fmt<ALIGNED, ARGMAX, true>(weights, n);
}
SIMD_SAMPLING_API uint64_t fargmax_st(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMAX, false>(weights, n): float_argsel_fmt<ALIGNED, ARGMAX, false>(weights, n);
}

SIMD_SAMPLING_API uint64_t fargmax(const float *weights, size_t n, int mt) {
    return mt ? fargmax_mt(weights, n): fargmax_st(weights, n);
}


SIMD_SAMPLING_API uint64_t dargsel(const double *weights, size_t n, ArgReduction ar, int mt)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    uint64_t ret;
    switch((int(aligned) << 2) | (int(ar == ARGMAX) << 1) | mt) {
        case 0: ret = double_argsel_fmt<UNALIGNED, ARGMIN, false>(weights, n); break;
        case 1: ret = double_argsel_fmt<UNALIGNED, ARGMIN, true>(weights, n); break;
        case 2: ret = double_argsel_fmt<UNALIGNED, ARGMAX, false>(weights, n); break;
        case 3: ret = double_argsel_fmt<UNALIGNED, ARGMAX, true>(weights, n); break;
        case 4: ret = double_argsel_fmt<ALIGNED, ARGMIN, false>(weights, n); break;
        case 5: ret = double_argsel_fmt<ALIGNED, ARGMIN, true>(weights, n); break;
        case 6: ret = double_argsel_fmt<ALIGNED, ARGMAX, false>(weights, n); break;
        case 7: ret = double_argsel_fmt<ALIGNED, ARGMAX, true>(weights, n); break;
        default: __builtin_unreachable();
    }
    return ret;
}

SIMD_SAMPLING_API ptrdiff_t fargsel_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, ArgReduction ar, int mt)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 2) | (int(ar == ARGMAX) << 1) | mt) {
        case 0: rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 1: rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 2: rv = float_argsel_k_fmt<UNALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 3: rv = float_argsel_k_fmt<UNALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        case 4: rv = float_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 5: rv = float_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 6: rv = float_argsel_k_fmt<ALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 7: rv = float_argsel_k_fmt<ALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t dargsel_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, ArgReduction ar, int mt)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 2) | (int(ar == ARGMAX) << 1) | mt) {
        case 0: rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 1: rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 2: rv = double_argsel_k_fmt<UNALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 3: rv = double_argsel_k_fmt<UNALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        case 4: rv = double_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 5: rv = double_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 6: rv = double_argsel_k_fmt<ALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 7: rv = double_argsel_k_fmt<ALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t dargsel_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 1) | (int(ar == ARGMAX))) {
        case 0: rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 1: rv = double_argsel_k_fmt<UNALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        case 2: rv = double_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 3: rv = double_argsel_k_fmt<ALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t dargsel_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 1) | (int(ar == ARGMAX))) {
        case 0: rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 1: rv = double_argsel_k_fmt<UNALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 2: rv = double_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 3: rv = double_argsel_k_fmt<ALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t fargsel_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 1) | (int(ar == ARGMAX))) {
        case 0: rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 1: rv = float_argsel_k_fmt<UNALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        case 2: rv = float_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret); break;
        case 3: rv = float_argsel_k_fmt<ALIGNED, ARGMAX, true>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t fargsel_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, enum ArgReduction ar) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    switch((int(aligned) << 1) | (int(ar == ARGMAX))) {
        case 0: rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 1: rv = float_argsel_k_fmt<UNALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        case 2: rv = float_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret); break;
        case 3: rv = float_argsel_k_fmt<ALIGNED, ARGMAX, false>(weights, n, k, ret); break;
        default: __builtin_unreachable();
    }
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t dargmax_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt) {
    if(mt) return dargmax_k_mt(weights, n, k, ret);
    else   return dargmax_k_st(weights, n, k, ret);
}
SIMD_SAMPLING_API ptrdiff_t fargmax_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt) {
    if(mt) return fargmax_k_mt(weights, n, k, ret);
    else   return fargmax_k_st(weights, n, k, ret);
}
SIMD_SAMPLING_API ptrdiff_t dargmin_k(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt) {
    if(mt) return dargmin_k_mt(weights, n, k, ret);
    else   return dargmin_k_st(weights, n, k, ret);
}
SIMD_SAMPLING_API ptrdiff_t fargmin_k(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret, int mt) {
    if(mt) return fargmin_k_mt(weights, n, k, ret);
    else   return fargmin_k_st(weights, n, k, ret);
}

SIMD_SAMPLING_API ptrdiff_t fargmax_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = float_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret);
    else        rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret);
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t dargmax_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = double_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret);
    else        rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret);
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t fargmax_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = float_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret);
    else        rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret);
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t dargmax_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = double_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret);
    else        rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret);
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t fargmin_k_st(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = float_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret);
    else        rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret);
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t dargmin_k_st(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = double_argsel_k_fmt<ALIGNED, ARGMIN, false>(weights, n, k, ret);
    else        rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, false>(weights, n, k, ret);
    return rv;
}
SIMD_SAMPLING_API ptrdiff_t fargmin_k_mt(const float *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = float_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret);
    else        rv = float_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret);
    return rv;
}

SIMD_SAMPLING_API ptrdiff_t dargmin_k_mt(const double *weights, size_t n, ptrdiff_t k, uint64_t *ret) {
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT == 0;
    ptrdiff_t rv;
    if(aligned) rv = double_argsel_k_fmt<ALIGNED, ARGMIN, true>(weights, n, k, ret);
    else        rv = double_argsel_k_fmt<UNALIGNED, ARGMIN, true>(weights, n, k, ret);
    return rv;
}

} // extern "C"


template<ArgReduction AR>
struct Cmp {
#if __AVX512F__
    static INLINE __mmask8 eq(__m512d x, __m512d y) {
        return _mm512_cmp_pd_mask(x, y, _CMP_EQ_OQ);
    }
    static INLINE __mmask16 eq(__m512 x, __m512 y) {
        return _mm512_cmp_ps_mask(x, y, _CMP_EQ_OQ);
    }
    static INLINE __mmask16 cmp(__m512 x, __m512 y, std::true_type) {
        return _mm512_cmp_ps_mask(x, y, _CMP_GT_OQ);
    }
    static INLINE __mmask16 cmp(__m512 x, __m512 y, std::false_type) {
        return _mm512_cmp_ps_mask(x, y, _CMP_LT_OQ);
    }
    static INLINE __mmask8 cmp(__m512d x, __m512d y, std::true_type) {
        return _mm512_cmp_pd_mask(x, y, _CMP_GT_OQ);
    }
    static INLINE __mmask8 cmp(__m512d x, __m512d y, std::false_type) {
        return _mm512_cmp_pd_mask(x, y, _CMP_LT_OQ);
    }
    static INLINE __mmask16 cmp(__m512 x, __m512 y) {
        return cmp(x, y, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE __mmask8 cmp(__m512d x, __m512d y) {
        return cmp(x, y, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE float reduce(__m512 x, std::true_type) {
        return _mm512_reduce_max_ps(x);
    }
    static INLINE float reduce(__m512 x, std::false_type) {
        return _mm512_reduce_min_ps(x);
    }
    static INLINE double reduce(__m512d x, std::true_type) {
        return _mm512_reduce_max_pd(x);
    }
    static INLINE double reduce(__m512d x, std::false_type) {
        return _mm512_reduce_min_pd(x);
    }
    static INLINE double reduce(__m512d x) {
        return reduce(x, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE float reduce(__m512 x) {
        return reduce(x, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE __m512d max(__m512d x, __m512d y, std::true_type) {
        return _mm512_max_pd(x, y);
    }
    static INLINE __m512d max(__m512d x, __m512d y, std::false_type) {
        return _mm512_min_pd(x, y);
    }
    static INLINE __m512 max(__m512 x, __m512 y, std::true_type) {
        return _mm512_max_ps(x, y);
    }
    static INLINE __m512 max(__m512 x, __m512 y, std::false_type) {
        return _mm512_min_ps(x, y);
    }
#endif
#if __AVX2__
    static INLINE __m256d max(__m256d x, __m256d y, std::true_type) {
        return _mm256_max_pd(x, y);
    }
    static INLINE __m256d max(__m256d x, __m256d y, std::false_type) {
        return _mm256_min_pd(x, y);
    }
    static INLINE __m256 max(__m256 x, __m256 y, std::true_type) {
        return _mm256_max_ps(x, y);
    }
    static INLINE __m256 max(__m256 x, __m256 y, std::false_type) {
        return _mm256_min_ps(x, y);
    }

    static INLINE __m256 cmp(__m256 x, __m256 y, std::true_type) {
        return _mm256_cmp_ps(x, y, _CMP_GT_OQ);
    }
    static INLINE __m256 cmp(__m256 x, __m256 y, std::false_type) {
        return _mm256_cmp_ps(x, y, _CMP_LT_OQ);
    }
    static INLINE __m256d cmp(__m256d x, __m256d y, std::true_type) {
        return _mm256_cmp_pd(x, y, _CMP_GT_OQ);
    }
    static INLINE __m256d cmp(__m256d x, __m256d y, std::false_type) {
        return _mm256_cmp_pd(x, y, _CMP_LT_OQ);
    }
    static INLINE __m256 eq(__m256 x, __m256 y) {
        return _mm256_cmp_ps(x, y, _CMP_EQ_OQ);
    }
    static INLINE __m256d eq(__m256d x, __m256d y) {
        return _mm256_cmp_pd(x, y, _CMP_EQ_OQ);
    }
    static INLINE __m256d max(__m256d x, std::false_type) {
        return broadcast_min(x);
    }
    static INLINE __m256d max(__m256d x, std::true_type) {
        return broadcast_max(x);
    }
    static INLINE __m256d max(__m256d x) {
        return max(x, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE __m256 max(__m256 x) {
        return max(x, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE __m256 max(__m256 x, std::false_type) {
        return broadcast_min(x);
    }
    static INLINE __m256 max(__m256 x, std::true_type) {
        return broadcast_max(x);
    }
#endif
#ifdef __SSE2__
    static INLINE __m128 cmp(__m128 x, __m128 y, std::true_type) {
        return _mm_cmp_ps(x, y, _CMP_GT_OQ);
    }
    static INLINE __m128 cmp(__m128 x, __m128 y, std::false_type) {
        return _mm_cmp_ps(x, y, _CMP_LT_OQ);
    }
    static INLINE __m128 max(__m128 x, std::false_type) {
        return broadcast_min(x);
    }
    static INLINE __m128 max(__m128 x, std::true_type) {
        return broadcast_max(x);
    }
#endif
    template<typename T>
    static INLINE T max(T x, T y) {
        return max(x, y, std::integral_constant<bool, AR == ARGMAX>());
    }
    template<typename T>
    static INLINE T max(T x) {
        return max(x, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE float cmp(const float x, const float y, std::false_type) {
        return x < y;
    }
    static INLINE float cmp(const float x, const float y, std::true_type) {
        return x > y;
    }
    template<typename T>
    static INLINE T cmp(T x, T y) {
        return cmp(x, y, std::integral_constant<bool, AR == ARGMAX>());
    }
    static INLINE double cmp(const double x, const double y, std::false_type) {
        return x < y;
    }
    static INLINE double cmp(const double x, const double y, std::true_type) {
        return x > y;
    }
    static INLINE double cmp(const double x, const double y) {
        return cmp(x, y, std::integral_constant<bool, AR == ARGMAX>());
    }
};

template<LoadFormat aln, ArgReduction AR, bool MT>
SIMD_SAMPLING_API uint64_t double_argsel_fmt(const double *weights, size_t n)
{
    uint64_t bestind = 0;
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr double STARTVAL = IS_MAX ? -std::numeric_limits<double>::max(): std::numeric_limits<double>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    __m512d vmaxv = _mm512_set1_pd(STARTVAL);
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            __m512d ov = load<aln>((const double *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(ov, vmaxv);
            if(cmpmask) {
                auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(ov));
                if((cmpmask = Cmp<AR>::eq(ov, newmaxv))) {
                    OMP_CRITICAL
                    if(Cmp<AR>::cmp(ov, vmaxv)) {
                        vmaxv = newmaxv;
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            __m512d ov = load<aln>((const double *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(ov, vmaxv);
            if(cmpmask) {
                auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(ov));
                if((cmpmask = Cmp<AR>::eq(ov, newmaxv))) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm512_cvtsd_f64(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(Cmp<AR>::cmp(weights[p], maxv)) {
            maxv = weights[p], bestind = p;
        }
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    __m256d vmaxv = _mm256_set1_pd(STARTVAL);
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            __m256d ov = load<aln>((const double *)&weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(ov, vmaxv);
            auto cmpmask = _mm256_movemask_pd(cmp);
            if(cmpmask) {
                __m256d newmax = Cmp<AR>::max(ov);
                {
                    OMP_CRITICAL
                    if(_mm256_movemask_pd(Cmp<AR>::cmp(ov, vmaxv))) {
                        vmaxv = newmax;
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            __m256d ov = load<aln>((const double *)&weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(ov, vmaxv);
            auto cmpmask = _mm256_movemask_pd(cmp);
            if(cmpmask) {
                __m256d newmax = Cmp<AR>::max(ov);
                cmpmask = _mm256_movemask_pd(Cmp<AR>::cmp(ov, vmaxv));
                vmaxv = newmax;
                bestind = ctz(cmpmask) + o * nperel;
            }
        }
    }
    double maxv = _mm256_cvtsd_f64(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(Cmp<AR>::cmp(weights[p], maxv)) maxv = weights[p], bestind = p;
    }
#else
    double bestv = STARTVAL;
    if(MT) {
        OMP_PFOR
        for(size_t i = 0; i < n; ++i) {
            auto v = weights[i];
            if(Cmp<AR>::cmp(v,  bestv)) {
                OMP_CRITICAL
                if(Cmp<AR>::cmp(v,  bestv)) bestv = v, bestind = i;
            }
        }
    } else {
        for(size_t i = 0; i < n; ++i) {
            auto v = weights[i];
            if(Cmp<AR>::cmp(v,  bestv)) {
                bestv = v, bestind = i;
            }
        }
    }
#endif
    return bestind;
}


template<LoadFormat aln, ArgReduction AR, bool MT>
SIMD_SAMPLING_API uint64_t float_argsel_fmt(const float * weights, size_t n)
{
    uint64_t bestind = 0;
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr float STARTVAL = IS_MAX ? -std::numeric_limits<float>::max(): std::numeric_limits<float>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 vmaxv = _mm512_set1_ps(STARTVAL);
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            __m512 lv = load<aln>((const float *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
                auto newmaxv = _mm512_set1_ps(Cmp<AR>::reduce(lv));
                if((cmpmask = Cmp<AR>::eq(lv, newmaxv))) {
                    OMP_CRITICAL
                    if(Cmp<AR>::cmp(lv, vmaxv)) {
                        vmaxv = newmaxv;
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            __m512 lv = load<aln>((const float *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
                auto newmaxv = _mm512_set1_ps(Cmp<AR>::reduce(lv));
                if((cmpmask = Cmp<AR>::eq(lv, newmaxv))) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    float maxv = _mm512_cvtss_f32(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        auto v = weights[p];
        if(Cmp<AR>::cmp(v,  maxv))
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    __m256 vmaxv = _mm256_set1_ps(STARTVAL);
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            __m256 divv = load<aln>((const float *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(divv, vmaxv);
            auto cmpmask = _mm256_movemask_ps(cmp);
            if(cmpmask) {
                const __m256 m2 = Cmp<AR>::max(divv);
                OMP_CRITICAL
                {
                    cmp = Cmp<AR>::eq(m2, divv);
                    cmpmask = _mm256_movemask_ps(cmp); // set to 1 where the largest value in cmp is
                    if(_mm256_movemask_ps(Cmp<AR>::cmp(m2, vmaxv))) { // If any in m2 are > vmaxv
                        vmaxv = m2;
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            __m256 divv = load<aln>((const float *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(divv, vmaxv);
            auto cmpmask = _mm256_movemask_ps(cmp);
            if(cmpmask) {
                const __m256 m2 = Cmp<AR>::max(divv);
                {
                    cmp = Cmp<AR>::eq(m2, divv);
                    cmpmask = _mm256_movemask_ps(cmp); // set to 1 where the largest value in cmp is
                    if(_mm256_movemask_ps(Cmp<AR>::cmp(m2, vmaxv))) { // If any in m2 are > vmaxv
                        vmaxv = m2;
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    }
    float maxv = _mm256_cvtss_f32(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(Cmp<AR>::cmp(weights[p], maxv)) {
            OMP_CRITICAL
            if(Cmp<AR>::cmp(weights[p], maxv))
                maxv = weights[p], bestind = p;
        }
    }
#else
    float bestv = weights[0];
    bestind = 0;
    if(MT) {
        OMP_PFOR
        for(size_t i = 1; i < n; ++i) {
            if(Cmp<AR>::cmp(weights[i], bestv)) {
                OMP_CRITICAL
                if(Cmp<AR>::cmp(weights[i], bestv))
                    bestv = weights[i], bestind = i;
            }
        }
    } else {
        for(size_t i = 1; i < n; ++i) {
            if(Cmp<AR>::cmp(weights[i], bestv)) {
                if(Cmp<AR>::cmp(weights[i], bestv))
                    bestv = weights[i], bestind = i;
            }
        }
    }
#endif
    return bestind;
}

template<typename T, ArgReduction AR>
struct Comparator {
    using type = typename std::conditional<AR == ARGMAX, std::greater<T>, std::less<T>>::type;
    template<typename OT>
    using otype = typename std::conditional<AR == ARGMAX, std::greater<OT>, std::less<OT>>::type;
    template<typename T2>
    static INLINE bool compare(const T2 &x, const T2 &y) {
        return otype<T2>()(x, y);
    }
};

template<typename FT, ArgReduction AR>
struct argminpq_t: public std::priority_queue<std::pair<FT, uint64_t>, std::vector<std::pair<FT, uint64_t>>, typename Comparator<std::pair<FT, uint64_t>, AR>::type> {
    using value_t = std::pair<FT, uint64_t>;
    using vec_t = std::vector<std::pair<FT, uint64_t>>;
    using cmp_t = Comparator<std::pair<FT, uint64_t>, AR>;
    //using fcmp_t = typename Comparator<std::pair<FT, uint64_t>, AR>::otype<FT>;
    uint32_t k_;
    argminpq_t(int k): k_(k) {
        this->c.reserve(k);
    }
    INLINE void add(FT val, uint64_t id) {
        if(this->size() < k_) {
            this->push(value_t(val, id));
        } else if(cmp_t::compare(val, this->top().first)) {
            this->pop();
            this->push(value_t(val, id));
        }
    }
};

template<LoadFormat aln, ArgReduction AR, bool MT>
ptrdiff_t float_argsel_k_fmt(const float * weights, size_t n, ptrdiff_t k, uint64_t *ret)
{
    std::vector<argminpq_t<float, AR>> pqs;
    if(MT) {
        int nt;
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
        pqs.resize(nt, argminpq_t<float, AR>(k));
    } else {
        pqs.emplace_back(k);
    }
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr float STARTVAL = IS_MAX ? -std::numeric_limits<float>::max(): std::numeric_limits<float>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 *vmaxvs;
    if(posix_memalign((void **)&vmaxvs, LSS_ALIGNMENT, sizeof(__m512) * pqs.size()))
        throw std::bad_alloc();
    for(size_t i = 0; i < pqs.size(); ++i) {
        vmaxvs[i] = _mm512_set1_ps(STARTVAL);
    }
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            const int tid = MT ? omp_get_thread_num(): 0;
            auto &vmaxv = vmaxvs[tid];
            argminpq_t<float, AR> &pq = pqs[tid];
            __m512 lv = load<aln>((const float *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
#define DOITER {\
    auto ind = ctz(cmpmask);\
    pq.add(lv[ind], ind + o * nperel);\
    cmpmask ^= (1u << ind); }
                switch(__builtin_popcount(cmpmask)) {
                    case 16: DOITER;__attribute__((fallthrough));
                    case 15: DOITER;__attribute__((fallthrough));
                    case 14: DOITER;__attribute__((fallthrough));
                    case 13: DOITER;__attribute__((fallthrough));
                    case 12: DOITER;__attribute__((fallthrough));
                    case 11: DOITER;__attribute__((fallthrough));
                    case 10: DOITER;__attribute__((fallthrough));
                    case 9: DOITER;__attribute__((fallthrough));
                    case 8: DOITER;__attribute__((fallthrough));
                    case 7: DOITER;__attribute__((fallthrough));
                    case 6: DOITER;__attribute__((fallthrough));
                    case 5: DOITER;__attribute__((fallthrough));
                    case 4: DOITER;__attribute__((fallthrough));
                    case 3: DOITER;__attribute__((fallthrough));
                    case 2: DOITER;__attribute__((fallthrough));
                    case 1: DOITER;
                }
                vmaxv = _mm512_set1_ps(pq.top().first);
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            argminpq_t<float, AR> &pq = pqs[0];
            auto &vmaxv = vmaxvs[0];
            __m512 lv = load<aln>((const float *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                    case 16: DOITER; __attribute__((fallthrough));
                    case 15: DOITER; __attribute__((fallthrough));
                    case 14: DOITER; __attribute__((fallthrough));
                    case 13: DOITER; __attribute__((fallthrough));
                    case 12: DOITER; __attribute__((fallthrough));
                    case 11: DOITER; __attribute__((fallthrough));
                    case 10: DOITER; __attribute__((fallthrough));
                    case 9: DOITER; __attribute__((fallthrough));
                    case 8: DOITER; __attribute__((fallthrough));
                    case 7: DOITER; __attribute__((fallthrough));
                    case 6: DOITER; __attribute__((fallthrough));
                    case 5: DOITER; __attribute__((fallthrough));
                    case 4: DOITER; __attribute__((fallthrough));
                    case 3: DOITER; __attribute__((fallthrough));
                    case 2: DOITER; __attribute__((fallthrough));
                    case 1: DOITER;
#undef DOITER
                }
                vmaxv = _mm512_set1_ps(pq.top().first);
            }
        }
    }
    std::free(vmaxvs);
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    __m256 *vmaxvs;
    if(posix_memalign((void **)&vmaxvs, LSS_ALIGNMENT, sizeof(__m256) * pqs.size()))
        throw std::bad_alloc();
    for(size_t i = 0; i < pqs.size(); ++i) {
        vmaxvs[i] = _mm256_set1_ps(STARTVAL);
    }
#define DOITER {\
    auto ind = ctz(cmpmask);\
    pq.add(lv[ind], ind + o * nperel);\
    cmpmask ^= (1u << ind); }
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            const int tid = MT ? omp_get_thread_num(): 0;
            auto &vmaxv = vmaxvs[tid];
            auto &pq = pqs[tid];
            __m256 lv = load<aln>((const float *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(lv, vmaxv);
            auto cmpmask = _mm256_movemask_ps(cmp);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                case 8: DOITER __attribute__ ((fallthrough));
                case 7: DOITER __attribute__ ((fallthrough));
                case 6: DOITER __attribute__ ((fallthrough));
                case 5: DOITER __attribute__ ((fallthrough));
                case 4: DOITER __attribute__ ((fallthrough));
                case 3: DOITER __attribute__ ((fallthrough));
                case 2: DOITER __attribute__ ((fallthrough));
                case 1: DOITER
                }
                vmaxv = _mm256_set1_ps(pq.top().first);
            }
        } 
    } else {
        for(size_t o = 0; o < e; ++o) {
            auto &pq = pqs[0];
            auto &vmaxv = vmaxvs[0];
            __m256 lv = load<aln>((const float *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(lv, vmaxv);
            auto cmpmask = _mm256_movemask_ps(cmp);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                case 8: DOITER __attribute__ ((fallthrough));
                case 7: DOITER __attribute__ ((fallthrough));
                case 6: DOITER __attribute__ ((fallthrough));
                case 5: DOITER __attribute__ ((fallthrough));
                case 4: DOITER __attribute__ ((fallthrough));
                case 3: DOITER __attribute__ ((fallthrough));
                case 2: DOITER __attribute__ ((fallthrough));
                case 1: DOITER
                }
                vmaxv = _mm256_set1_ps(pq.top().first);
            }
        } 
    }
    std::free(vmaxvs);
#undef DOITER
#else
    if(MT) {
        OMP_PFOR
        for(size_t i = 0; i < n; ++i) {
            const int tid = omp_get_thread_num();
            auto &pq = pqs[tid];
            if(Cmp<AR>::cmp(weights[i], pq.top().first)) {
                pq.add(weights[i], i);
            }
        }
    } else {
        for(size_t i = 0; i < n; ++i) {
            if(Cmp<AR>::cmp(weights[i], pqs[0].top().first)) {
                pqs[0].add(weights[i], i);
                if(Cmp<AR>::cmp(weights[i], bestv))
                    bestv = weights[i], bestind = i;
            }
        }
    }
#endif
    // TODO: parallelize merging
    auto &basepq = pqs[0];
    for(ptrdiff_t i = pqs.size() - 1; i > 0; --i) {
        auto &opq = pqs[i];
        while(opq.size()) {
            basepq.add(opq.top().first, opq.top().second);
            if(basepq.size() > size_t(k)) basepq.pop();
        }
        pqs.pop_back();
    }
    for(size_t p = e * nperel; p != n; ++p) {
        basepq.add(weights[p], p);
        if(basepq.size() > size_t(k)) basepq.pop();
    }
    const ptrdiff_t rv = basepq.size();
    for(ptrdiff_t i = 0; i < rv; ++i) {
        ret[rv - i - 1] = basepq.top().second;
        basepq.pop();
    }
    return rv;
}

template<LoadFormat aln, ArgReduction AR, bool MT> ptrdiff_t
double_argsel_k_fmt(const double * weights, size_t n, ptrdiff_t k, uint64_t *ret)
{
    std::vector<argminpq_t<double, AR>> pqs;
    if(MT) {
        int nt;
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
        pqs.resize(nt, argminpq_t<double, AR>(k));
    } else {
        pqs.emplace_back(k);
    }
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr double STARTVAL = IS_MAX ? -std::numeric_limits<double>::max(): std::numeric_limits<double>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512) / sizeof(double);
    const size_t e = n / nperel;
    __m512d *vmaxvs;
    if(posix_memalign((void **)&vmaxvs, LSS_ALIGNMENT, sizeof(__m512d) * pqs.size()))
        throw std::bad_alloc();
    for(size_t i = 0; i < pqs.size(); ++i) {
        vmaxvs[i] = _mm512_set1_pd(STARTVAL);
    }
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            const int tid = MT ? omp_get_thread_num(): 0;
            auto &vmaxv = vmaxvs[tid];
            argminpq_t<double, AR> &pq = pqs[tid];
            __m512d lv = load<aln>((const double *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
#define DOITER do {\
    auto ind = ctz(cmpmask);\
    pq.add(lv[ind], ind + o * nperel);\
    cmpmask ^= (1u << ind); } while(0)
                switch(__builtin_popcount(cmpmask)) {
                    case 8: DOITER; __attribute__((fallthrough));
                    case 7: DOITER; __attribute__((fallthrough));
                    case 6: DOITER; __attribute__((fallthrough));
                    case 5: DOITER; __attribute__((fallthrough));
                    case 4: DOITER; __attribute__((fallthrough));
                    case 3: DOITER; __attribute__((fallthrough));
                    case 2: DOITER; __attribute__((fallthrough));
                    case 1: DOITER;
                }
                vmaxv = _mm512_set1_pd(pq.top().first);
            }
        }
    } else {
        for(size_t o = 0; o < e; ++o) {
            argminpq_t<double, AR> &pq = pqs[0];
            auto &vmaxv = vmaxvs[0];
            __m512d lv = load<aln>((const double *)&weights[o * nperel]);
            auto cmpmask = Cmp<AR>::cmp(lv, vmaxv);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                    case 8: DOITER;__attribute__((fallthrough));
                    case 7: DOITER;__attribute__((fallthrough));
                    case 6: DOITER;__attribute__((fallthrough));
                    case 5: DOITER;__attribute__((fallthrough));
                    case 4: DOITER;__attribute__((fallthrough));
                    case 3: DOITER;__attribute__((fallthrough));
                    case 2: DOITER;__attribute__((fallthrough));
                    case 1: DOITER;
#undef DOITER
                }
                vmaxv = _mm512_set1_pd(pq.top().first);
            }
        }
    }
    std::free(vmaxvs);
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(double);
    const size_t e = (n / nperel);
    __m256d *vmaxvs;
    if(posix_memalign((void **)&vmaxvs, LSS_ALIGNMENT, sizeof(__m256) * pqs.size()))
        throw std::bad_alloc();
    for(size_t i = 0; i < pqs.size(); ++i) {
        vmaxvs[i] = _mm256_set1_pd(STARTVAL);
    }
#define DOITER {\
    auto ind = ctz(cmpmask);\
    pq.add(lv[ind], ind + o * nperel);\
    cmpmask ^= (1u << ind); }
    if(MT) {
        OMP_PFOR
        for(size_t o = 0; o < e; ++o) {
            const int tid = MT ? omp_get_thread_num(): 0;
            auto &vmaxv = vmaxvs[tid];
            auto &pq = pqs[tid];
            __m256d lv = load<aln>((const double *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(lv, vmaxv);
            auto cmpmask = _mm256_movemask_pd(cmp);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                case 4: DOITER __attribute__ ((fallthrough));
                case 3: DOITER __attribute__ ((fallthrough));
                case 2: DOITER __attribute__ ((fallthrough));
                case 1: DOITER
                }
                vmaxv = _mm256_set1_pd(pq.top().first);
            }
        } 
    } else {
        for(size_t o = 0; o < e; ++o) {
            auto &pq = pqs[0];
            auto &vmaxv = vmaxvs[0];
            __m256d lv = load<aln>((const double *) &weights[o * nperel]);
            auto cmp = Cmp<AR>::cmp(lv, vmaxv);
            auto cmpmask = _mm256_movemask_pd(cmp);
            if(cmpmask) {
                switch(__builtin_popcount(cmpmask)) {
                case 4: DOITER __attribute__ ((fallthrough));
                case 3: DOITER __attribute__ ((fallthrough));
                case 2: DOITER __attribute__ ((fallthrough));
                case 1: DOITER
                }
                vmaxv = _mm256_set1_pd(pq.top().first);
            }
        } 
    }
    std::free(vmaxvs);
#undef DOITER
#else
    if(MT) {
        OMP_PFOR
        for(size_t i = 0; i < n; ++i) {
            const int tid = omp_get_thread_num();
            auto &pq = pqs[tid];
            if(Cmp<AR>::cmp(weights[i], pq.top().first)) {
                pq.add(weights[i], i);
            }
        }
    } else {
        for(size_t i = 0; i < n; ++i) {
            if(Cmp<AR>::cmp(weights[i], pqs[0].top().first)) {
                pqs[0].add(weights[i], i);
                if(Cmp<AR>::cmp(weights[i], bestv))
                    bestv = weights[i], bestind = i;
            }
        }
    }
#endif
    // TODO: parallelize merging
    auto &basepq = pqs[0];
    for(ptrdiff_t i = pqs.size() - 1; i > 0; --i) {
        auto &opq = pqs[i];
        while(opq.size()) {
            basepq.add(opq.top().first, opq.top().second);
            if(basepq.size() > size_t(k)) basepq.pop();
        }
        pqs.pop_back();
    }
    for(size_t p = e * nperel; p != n; ++p) {
        basepq.add(weights[p], p);
        if(basepq.size() > size_t(k)) basepq.pop();
    }
    const ptrdiff_t rv = basepq.size();
    for(ptrdiff_t i = 0; i < rv; ++i) {
        ret[rv - i - 1] = basepq.top().second;
        basepq.pop();
    }
    return rv;
}

