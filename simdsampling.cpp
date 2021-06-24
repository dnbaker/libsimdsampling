#ifdef _OPENMP
#include "omp.h"
#endif
#include "sleef.h"
#include "x86intrin.h"
#include "ctz.h"
#include "simdsampling.h"
#include "aesctr/wy.h"
#include <limits>
#include <queue>
#include <memory>
#include "reds.h"

#if __AVX512F__ || __AVX2__
#include "simdpcg32.h"
#endif
#include "reservoir.h"

#ifdef SIMD_SAMPLING_USE_APPROX_LOG
    #define Sleef_logd2_u35 _mm_ss_alog_pd
    #define Sleef_logd4_u35 _mm256_ss_alog_pd
    #define Sleef_logd8_u35 _mm512_ss_alog_pd
    #define Sleef_logf4_u35 _mm_ss_alog_ps
    #define Sleef_logf8_u35 _mm256_ss_alog_ps
    #define Sleef_logf16_u35 _mm512_ss_alog_ps
#endif


#ifndef __SLEEF_H__
#ifdef __AVX512F__
static inline __m512 Sleef_logf16_u35(__m512 x) {
    #pragma GCC unroll 16
    for(size_t i = 0; i < 16; ++i)
        x[i] = logf(x[i]);
    return x;
}
static inline __m512 Sleef_logf16_u10(__m512 x) {return Sleef_logf16_u35(x);}
static inline __m512d Sleef_logd8_u35(__m512d x) {
    #pragma GCC unroll 8
    for(size_t i = 0; i < 8; ++i) x[i] = log(x[i]);
    return x;
}
static inline __m512d Sleef_logd8_u10(__m512d x) {return Sleef_logd8_u35(x);}
static inline __m512 Sleef_sqrtf16_u35(__m512 x) {
    #pragma GCC unroll 16
    for(size_t i = 0; i < 16; ++i)
        x[i] = sqrtf(x[i]);
    return x;
}
static inline __m512 Sleef_sqrtf16_u10(__m512 x) {return Sleef_sqrtf16_u35(x);}
static inline __m512d Sleef_sqrtd8_u35(__m512d x) {
    #pragma GCC unroll 8
    for(size_t i = 0; i < 8; ++i) x[i] = sqrt(x[i]);
    return x;
}
static inline __m512d Sleef_sqrtd8_u10(__m512d x) {return Sleef_sqrtd8_u35(x);}
#endif
#ifdef __AVX2__
static inline __m256 Sleef_logf8_u35(__m256 x) {
    #pragma GCC unroll 8
    for(size_t i = 0; i < 8; ++i) x[i] = logf(x[i]);
    return x;
}
static inline __m256 Sleef_logf8_u10(__m256 x) {return Sleef_logf8_u35(x);}
static inline __m256d Sleef_logd4_u35(__m256d x) {
    x[0] = log(x[0]); x[1] = log(x[1]);
    x[2] = log(x[2]); x[3] = log(x[3]);
    return x;
}
static inline __m256d Sleef_logd4_u10(__m256d x) {return Sleef_logd4_u35(x);}
static inline __m256 Sleef_sqrtf8_u35(__m256 x) {
    #pragma GCC unroll 8
    for(size_t i = 0; i < 8; ++i)
        x[i] = sqrtf(x[i]);
    return x;
}
static inline __m256 Sleef_sqrtf8_u10(__m256 x) {return Sleef_sqrtf8_u35(x);}
static inline __m256d Sleef_sqrtd4_u35(__m256d x) {
    #pragma GCC unroll 4
    for(size_t i = 0; i < 4; ++i) x[i] = sqrt(x[i]);
    return x;
}
static inline __m256d Sleef_sqrtd4_u10(__m256d x) {return Sleef_sqrtd4_u35(x);}
#endif
#ifdef __SSE2__
static inline __m128 Sleef_logf4_u35(__m128 x) {
    x[0] = logf(x[0]); x[1] = logf(x[1]);
    x[2] = logf(x[2]); x[3] = logf(x[3]);
    return x;
}
static inline __m128 Sleef_logf4_u10(__m128 x) {return Sleef_logf4_u35(x);}
static inline __m128d Sleef_logd2_u35(__m128d x) {
    x[0] = log(x[0]); x[1] = log(x[1]);
    return x;
}
static inline __m128d Sleef_logd2_u10(__m128d x) {return Sleef_logd2_u35(x);}
static inline __m128 Sleef_sqrtf4_u35(__m128 x) {
    x[0] = sqrtf(x[0]); x[1] = sqrtf(x[1]);
    x[2] = sqrtf(x[2]); x[3] = sqrtf(x[3]);
    return x;
}
static inline __m128 Sleef_sqrtf4_u10(__m128 x) {return Sleef_sqrtf4_u35(x);}
static inline __m128d Sleef_sqrtd2_u35(__m128d x) {
    x[0] = sqrt(x[0]); x[1] = sqrt(x[1]);
    return x;
}
static inline __m128d Sleef_sqrtd2_u10(__m128d x) {return Sleef_sqrtd2_u35(x);}
#endif
#endif

#if SIMD_SAMPLING_HIGH_PRECISION
#  ifndef Sleef_logd2_u35
#    define Sleef_logd2_u35 Sleef_logd2_u10
#  endif
#  ifndef Sleef_logd4_u35
#    define Sleef_logd4_u35 Sleef_logd4_u10
#  endif
#  ifndef Sleef_logd8_u35
#    define Sleef_logd8_u35 Sleef_logd8_u10
#  endif
#  ifndef Sleef_logf4_u35
#    define Sleef_logf4_u35 Sleef_logf4_u10
#  endif
#  ifndef Sleef_logf8_u35
#    define Sleef_logf8_u35 Sleef_logf8_u10
#  endif
#  ifndef Sleef_logf16_u35
#    define Sleef_logf16_u35 Sleef_logf16_u10
#  endif
#endif

#if __cplusplus < 201703L
#define LSS_FLOAT_PSMUL static_cast<float>(1. / (1ull << 29))
#define LSS_DOUBLE_PDMUL (1. / (1ull << 52))
#else
#define LSS_FLOAT_PSMUL  0x1p-29f
#define LSS_DOUBLE_PDMUL 0x1p-52
#endif

#ifndef __FMA__
#ifdef __AVX2__
#define _mm256_fmadd_ps(a, b, c) (_mm256_add_ps(c, _mm256_mul_ps(a, b)))
#define _mm256_fmadd_pd(a, b, c) (_mm256_add_pd(c, _mm256_mul_pd(a, b)))
#endif
#ifdef __SSE2__
#define _mm_fmadd_ps(a, b, c) (_mm_add_ps(c, _mm_mul_ps(a, b)))
#define _mm_fmadd_pd(a, b, c) (_mm_add_pd(c, _mm_mul_pd(a, b)))
#endif
#endif

#if !__AVX512DQ__
#  ifndef _mm512_cvtepi64_pd
#    define _mm512_cvtepi64_pd(x) _mm512_fmadd_pd(\
        _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(_mm512_srli_epi64(x, 32))),\
        _mm512_set1_pd(0x100000000LL), _mm512_cvtepu32_pd(_mm512_cvtepi64_epi32(x)))
#  endif
#endif
#define LIBKL_ALOG_PD_MUL 1.539095918623324e-16
#define LIBKL_ALOG_PD_INC -709.0895657128241
#define LIBKL_ALOG_PS_MUL 8.2629582881927490e-8f
#define LIBKL_ALOG_PS_INC -88.02969186f


#if __AVX512F__

static inline  __attribute__((always_inline)) __m512d _mm512_ss_alog_pd(__m512d x) {
    return _mm512_fmadd_pd(_mm512_cvtepi64_pd(_mm512_castpd_si512(x)),
                           _mm512_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm512_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m512 _mm512_ss_alog_ps(__m512 x) {
    return _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_castps_si512(x)),
                           _mm512_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm512_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif

#if __AVX2__
static inline __attribute__((always_inline)) __m256 _mm256_abs_ps(__m256 a) {
    return _mm256_max_ps(a, -a);
}
static inline __attribute__((always_inline)) __m256d _mm256_abs_pd(__m256d a) {
    return _mm256_max_pd(a, -a);
}
#ifndef DEFINED_mm256_cvtepi64_pd_manual
static inline __attribute__((always_inline)) __m256d _mm256_cvtepi64_pd_manual(const __m256i v)
// From https://stackoverflow.com/questions/41144668/how-to-efficiently-perform-double-int64-conversions-with-sse-avx/41223013
{
    __m256i magic_i_lo   = _mm256_set1_epi64x(0x4330000000000000);                /* 2^52        encoded as floating-point  */
    __m256i magic_i_hi32 = _mm256_set1_epi64x(0x4530000000000000);                /* 2^84        encoded as floating-point  */
    __m256i magic_i_all  = _mm256_set1_epi64x(0x4530000000100000);                /* 2^84 + 2^52 encoded as floating-point  */
    __m256d magic_d_all  = _mm256_castsi256_pd(magic_i_all);

    __m256i v_lo         = _mm256_blend_epi32(magic_i_lo, v, 0b01010101);         /* Blend the 32 lowest significant bits of v with magic_int_lo                                                   */
    __m256i v_hi         = _mm256_srli_epi64(v, 32);                              /* Extract the 32 most significant bits of v                                                                     */
            v_hi         = _mm256_xor_si256(v_hi, magic_i_hi32);                  /* Blend v_hi with 0x45300000                                                                                    */
    __m256d v_hi_dbl     = _mm256_sub_pd(_mm256_castsi256_pd(v_hi), magic_d_all); /* Compute in double precision:                                                                                  */
    __m256d result       = _mm256_add_pd(v_hi_dbl, _mm256_castsi256_pd(v_lo));    /* (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition !!                        */
            return result;                                                        /* With gcc use -O3, then -fno-associative-math is default. Do not use -Ofast, which enables -fassociative-math! */
                                                                                  /* With icc use -fp-model precise                                                                                */
}
#define DEFINED_mm256_cvtepi64_pd_manual 1
#endif
#ifndef  _mm256_cvtepi64_pd
#define  _mm256_cvtepi64_pd(x) _mm256_cvtepi64_pd_manual(x)
#endif

static inline  __attribute__((always_inline)) __m256d _mm256_ss_alog_pd(__m256d x) {
    return _mm256_fmadd_pd(_mm256_cvtepi64_pd(_mm256_castpd_si256(x)),
                           _mm256_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm256_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m256 _mm256_ss_alog_ps(__m256 x) {
    return _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_castps_si256(x)),
                           _mm256_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm256_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif
#if __SSE2__
#ifndef DEFINED_mm_cvtepi64_pd_manual
static inline __attribute__((always_inline)) __m128d _mm_cvtepi64_pd_manual(__m128i x){
    __m128i xH = _mm_srli_epi64(x, 32);
    xH = _mm_or_si128(xH, _mm_castpd_si128(_mm_set1_pd(19342813113834066795298816.)));          //  2^84
    __m128i xL = _mm_blend_epi16(x, _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)), 0xcc);   //  2^52
    __m128d f = _mm_sub_pd(_mm_castsi128_pd(xH), _mm_set1_pd(19342813118337666422669312.));     //  2^84 + 2^52
    return _mm_add_pd(f, _mm_castsi128_pd(xL));
}
#define DEFINED_mm_cvtepi64_pd_manual
#endif

#ifndef _mm_cvtepi64_pd
#define _mm_cvtepi64_pd(x) _mm_cvtepi64_pd_manual(x)
#endif
static inline __attribute__((always_inline)) __m128 _mm_abs_ps(__m128 a) {
    return _mm_max_ps(a, -a);
}
static inline __attribute__((always_inline)) __m128d _mm_abs_pd(__m128d a) {
    return _mm_max_pd(a, -a);
}
static inline  __attribute__((always_inline)) __m128d _mm_ss_alog_pd(__m128d x) {
    return _mm_fmadd_pd(_mm_cvtepi64_pd(_mm_castpd_si128(x)),
                           _mm_set1_pd(LIBKL_ALOG_PD_MUL),
                           _mm_set1_pd(LIBKL_ALOG_PD_INC));
}
static inline  __attribute__((always_inline)) __m128 _mm_ss_alog_ps(__m128 x) {
    return _mm_fmadd_ps(_mm_cvtepi32_ps(_mm_castps_si128(x)),
                           _mm_set1_ps(LIBKL_ALOG_PS_MUL),
                           _mm_set1_ps(LIBKL_ALOG_PS_INC));
}
#endif

#ifndef FALLTHROUGH
#  ifdef __GNUC__
#    define FALLTHROUGH __attribute__((fallthrough));
#  else
#    define FALLTHROUGH ;
#  endif
#endif

#ifdef __AVX512F__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m512) / sizeof(char))
#elif __AVX2__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m256) / sizeof(char))
#elif __AVX__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m128) / sizeof(char))
#else
#define SIMD_SAMPLING_ALIGNMENT 1
#endif

#if defined(__GNUC__) && __GNUC__ < 8
#define _mm256_set_m128i(xmm1, xmm2) _mm256_permute2f128_si256(_mm256_castsi128_si256(xmm1), _mm256_castsi128_si256(xmm2), 2)
#define _mm256_set_m128f(xmm1, xmm2) _mm256_permute2f128_ps(_mm256_castps128_ps256(xmm1), _mm256_castps128_ps256(xmm2), 2)
#endif


// Do we really need to `max(0, weights)`?
// It should keep us robust to slightly off results due to precision
// but it does waste a couple instructions

#ifndef LSS_MAX_0
#define LSS_MAX_0 0
#endif

#ifndef USE_AVX256_RNG
#define USE_AVX256_RNG 1
#endif

using namespace reservoir_simd;

// Forward declaratios of core kernels
// Single-sample
template<LoadFormat aln>
SIMD_SAMPLING_API uint64_t double_simd_sampling_fmt(const double *weights, size_t n, uint64_t seed);
template<LoadFormat aln>
SIMD_SAMPLING_API uint64_t float_simd_sampling_fmt(const float *weights, size_t n, uint64_t seed);

// Multiple-sample
template<LoadFormat aln> SIMD_SAMPLING_API int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement);
template<LoadFormat aln> SIMD_SAMPLING_API int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement);
template<LoadFormat aln> SIMD_SAMPLING_API int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement);
template<LoadFormat aln> SIMD_SAMPLING_API int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement);

using ssize_t = typename std::make_signed<size_t>::type;

extern "C" {
SIMD_SAMPLING_API uint64_t dsimd_sample(const double *weights, size_t n, uint64_t seed, enum SampleFmt fmt)
{
    if(fmt & USE_EXPONENTIAL_SKIPS) {
        int nt = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
#endif
        return DOGS::CalaverasReservoirSampler<uint64_t>::parallel_sample1(weights, weights + n, nt, seed);
    }
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? double_simd_sampling_fmt<UNALIGNED>(weights, n, seed)
        : double_simd_sampling_fmt<ALIGNED>(weights, n, seed);
}

SIMD_SAMPLING_API uint64_t fsimd_sample(const float *weights, size_t n, uint64_t seed, enum SampleFmt fmt)
{
    if(fmt & USE_EXPONENTIAL_SKIPS) {
        int nt = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
#endif
        return DOGS::CalaverasReservoirSampler<uint64_t>::parallel_sample1(weights, weights + n, nt, seed);
    }
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? float_simd_sampling_fmt<UNALIGNED>(weights, n, seed)
        : float_simd_sampling_fmt<ALIGNED>(weights, n, seed);
}

SIMD_SAMPLING_API int dsimd_sample_k(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt)
{
    if(k <= 0) throw std::invalid_argument(std::string("k must be > 0 [") + std::to_string(k) + "]\n");
    if(fmt & USE_EXPONENTIAL_SKIPS) {
        if(fmt & WITH_REPLACEMENT) {
            std::fprintf(stderr, "Warning: exponential skips with replacement not implemented. Returning without replacement.\n");
        }
        int nt = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
#endif
        auto container = DOGS::CalaverasReservoirSampler<uint64_t>::parallel_sample_weights(weights, weights + n, k, nt, seed);
        if(container.size() != unsigned(k)) throw std::runtime_error(std::string("container expected ") + std::to_string(k) + ", but found " + std::to_string(container.size()));
        std::sort(container.begin(), container.end());
        auto rp = ret;
        for(const auto &pair: container)
            *rp++ = pair.second;
        std::sort(ret, rp);
        return k;
    }
    const bool with_replacement = fmt & WITH_REPLACEMENT;
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? double_simd_sample_k_fmt<UNALIGNED>(weights, n, k, ret, seed, with_replacement)
        : double_simd_sample_k_fmt<ALIGNED>(weights, n, k, ret, seed, with_replacement);
}

SIMD_SAMPLING_API int fsimd_sample_k(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, enum SampleFmt fmt)
{
    if(k <= 0) throw std::invalid_argument(std::string("k must be > 0 [") + std::to_string(k) + "]\n");
    if(fmt & USE_EXPONENTIAL_SKIPS) {
        if(fmt & WITH_REPLACEMENT) {
            std::fprintf(stderr, "Warning: exponential skips with replacement not implemented. Returning without replacement.\n");
        }
        int nt = 1;
#ifdef _OPENMP
        #pragma omp parallel
        {
            nt = omp_get_num_threads();
        }
#endif
        auto container = DOGS::CalaverasReservoirSampler<uint64_t>::parallel_sample_weights(weights, weights + n, k, nt, seed);
        if(container.size() != unsigned(k)) throw std::runtime_error(std::string("container expected ") + std::to_string(k) + ", but found " + std::to_string(container.size()));
        auto rp = ret;
        for(const auto &pair: container)
            *rp++ = pair.second;
        std::sort(ret, rp);
        return k;
    }
    const bool with_replacement = fmt & WITH_REPLACEMENT;
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? float_simd_sample_k_fmt<UNALIGNED>(weights, n, k, ret, seed, with_replacement)
        : float_simd_sample_k_fmt<ALIGNED>(weights, n, k, ret, seed, with_replacement);
}


} // extern "C" for the C-api

template<LoadFormat aln>
uint64_t double_simd_sampling_fmt(const double *weights, size_t n, uint64_t seed)
{
    uint64_t bestind = 0;
    wy::WyRand<uint64_t> baserng(seed * seed + 13);
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
#endif


#if __AVX512F__
    #if __AVX512DQ__
    using simdpcg_t = avx512bis_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        x.state[0] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
        x.inc[1] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
    };
    #else
    using simdpcg_t = avx256_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    #endif
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngstates[omp_get_thread_num()],
                             baserngstate);
        __m512i v =
#if __AVX512DQ__
                    avx512bis_pcg32_random_r(&rng);
#else
                    pack_result(avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng),avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng));
#endif

        const __m512d v2 =
#ifdef __AVX512DQ__
            _mm512_mul_pd(_mm512_cvtepi64_pd(_mm512_srli_epi64(v, 12)), _mm512_set1_pd(LSS_DOUBLE_PDMUL));
#else
            _mm512_mul_pd(_mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(_mm512_srli_epi64(v, 12), _mm512_castpd_si512(_mm512_set1_pd(0x0010000000000000)))), _mm512_set1_pd(0x0010000000000000)),  _mm512_set1_pd(LSS_DOUBLE_PDMUL));
#endif
        // Shift right by 12, convert from ints to doubles, and then multiply by 2^-52
        // resulting in uniform [0, 1] sampling

        const __m512d v3 = _mm512_ss_alog_pd(v2);
        // Log-transform the [0, 1] sampling
        __m512d ov = load<aln>((const double *)&weights[o * nperel]);
        auto divv = _mm512_div_pd(v3, ov);
        auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
        // TODO: replace with a switch-based unrolled loop on popcount(cmpmask), since
        //       there are only 8
        if(cmpmask) {
            auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(divv));
            if((cmpmask = _mm512_cmp_pd_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                OMP_CRITICAL
                cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
                if(cmpmask) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm512_cvtsd_f64(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    __m256d vmaxv = _mm256_set1_pd(-std::numeric_limits<double>::max());
    using simdpcg_t = avx256_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngstates[omp_get_thread_num()],
                             baserngstate);
        __m256i v = _mm256_set_m128i(avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng));
        auto v2 = _mm256_or_si256(_mm256_srli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
        auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(LSS_DOUBLE_PDMUL));
        __m256d v5 = _mm256_ss_alog_pd(v4);
        __m256d ov = load<aln>((const double *)&weights[o * nperel]);
        auto divv = _mm256_div_pd(v5, ov);
        auto cmp = _mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_pd(cmp);
        if(cmpmask) {
            __m256d y = _mm256_permute2f128_pd(divv, divv, 1);
            __m256d m1 = _mm256_max_pd(divv, y);
            __m256d m2 = _mm256_permute_pd(m1, 5);
            auto newmaxv = _mm256_max_pd(m1, m2);
            {
                OMP_CRITICAL
                if(_mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ))) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm256_cvtsd_f64(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(!weights[p]) continue;
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __SSE2__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(double);
    const size_t e = n / nperel;
    double maxv = -std::numeric_limits<double>::max();
#ifdef __AVX__
    __m128d vmaxv = _mm_set1_pd(maxv);
#endif
#ifdef _OPENMP
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
#endif
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(LSS_DOUBLE_PDMUL));
        auto v5 = _mm_ss_alog_pd(v4);
        __m128d ov6 = load<aln>((const double *) &weights[o * nperel]);
        auto divv = _mm_div_pd(v5, ov6);
        int cmpmask;
#if __AVX__
        __m128d cmp;
        cmp = _mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        cmpmask = _mm_movemask_pd(cmp);
#else
        cmpmask = (divv[0] > maxv) | (divv[1] > maxv);
#endif
        if(cmpmask) {
            OMP_CRITICAL
#if __AVX__
            cmpmask = _mm_movemask_pd(_mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ));
#else
            cmpmask = (divv[0] > maxv) | (divv[1] > maxv);
#endif
            if(cmpmask) {
#if __AVX__
                vmaxv = _mm_max_pd(divv, _mm_permute_pd(divv, 1));
#else
                maxv = std::max(divv[0], divv[1]);
#endif
                bestind = (divv[1] > divv[0]) + o * nperel;
            }
        }
    }
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    double bestv = std::log(std::uniform_real_distribution<double>()(rng)) / weights[0];
    for(size_t i = 1; i < n; ++i) {
        auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
        if(v > bestv) bestv = v, bestind = i;
    }
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    OMP_ONLY(if(rngstates != &baserngstate) std::free(rngstates);)
#endif
    return bestind;
}


template<LoadFormat aln>
uint64_t float_simd_sampling_fmt(const float * weights, size_t n, uint64_t seed)
{
    uint64_t bestind = 0;
    wy::WyRand<uint64_t> baserng(seed * seed + 13);
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
    for(auto &i: rngs) i.seed(baserng());
#endif
#ifdef __AVX512F__
    #if __AVX512DQ__
    using simdpcg_t = avx512_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        x.state = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.inc = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
    };
    #else
    using simdpcg_t = avx256_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
#if 0
        x.state[0] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.inc[1] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
#else
        x.state = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
#endif
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    #endif
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 vmaxv = _mm512_set1_ps(-std::numeric_limits<float>::max());
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto rngptr = OMP_ELSE(&rngstates[omp_get_thread_num()],
                               &baserngstate);
        __m512i v =
#ifdef __AVX512DQ__
        _mm512_srli_epi32(_mm512_inserti32x8(_mm512_castsi256_si512(avx512_pcg32_random_r(rngptr)), avx512_pcg32_random_r(rngptr), 1), 3);
#else
        _mm512_srli_epi32(pack_result(avx256_pcg32_random_r(rngptr), avx256_pcg32_random_r(rngptr),avx256_pcg32_random_r(rngptr),avx256_pcg32_random_r(rngptr)), 3);
#endif
        auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(LSS_FLOAT_PSMUL));
        __m512 v5 = _mm512_ss_alog_ps(v4);
        __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        __m512 divv = _mm512_div_ps(v5, lv);
        auto cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_ps(_mm512_reduce_max_ps(divv));
            if((cmpmask = _mm512_cmp_ps_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                OMP_CRITICAL
                if(_mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ)) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    float maxv = _mm512_cvtss_f32(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    __m256 vmaxv = _mm256_set1_ps(-std::numeric_limits<float>::max());
    using simdpcg_t = avx2_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state[0] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.inc[1] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m256) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngstates[omp_get_thread_num()],
                             baserngstate);
        __m256i v = _mm256_srli_epi32(avx2_pcg32_random_r(&rng), 3);
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(LSS_FLOAT_PSMUL));
#ifndef NDEBUG
        float sum = 0.;
        for(size_t i = 0; i < sizeof(v) / sizeof(uint32_t); ++i) {
            float nextv;
            std::memcpy(&nextv, (float *)&v2 + i, sizeof(nextv));
            sum += nextv;
        }
#endif
        auto v3 = _mm256_ss_alog_ps(v2);
        __m256 ov6 = load<aln>((const float *) &weights[o * nperel]);
        auto divv = _mm256_div_ps(v3, ov6);
        auto cmp = _mm256_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_ps(cmp);
        if(cmpmask) {
            const __m256 permHalves = _mm256_permute2f128_ps(divv, divv, 1);
            const __m256 m0 = _mm256_max_ps(permHalves, divv);
            const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
            const __m256 m1 = _mm256_max_ps(m0, perm0);
            const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
            const __m256 m2 = _mm256_max_ps(perm1, m1);
            cmpmask = _mm256_movemask_ps(_mm256_cmp_ps(m2, divv, _CMP_EQ_OQ));
            OMP_CRITICAL
            if(_mm256_movemask_ps(_mm256_cmp_ps(m2, vmaxv, _CMP_GT_OQ))) {
                vmaxv = m2;
                bestind = ctz(cmpmask) + o * nperel;
            }
        }
    }
    float maxv = _mm256_cvtss_f32(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(float);
    const size_t e = n / nperel;
    float maxv = -std::numeric_limits<float>::max();
    __m128 vmaxv = _mm_set1_ps(maxv);
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v3 = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(LSS_FLOAT_PSMUL));
        auto v5 = _mm_ss_alog_ps(v3);
        __m128 ov6 = load<aln>((const float *) &weights[o * nperel]);
        auto divv = _mm_div_ps(v5, ov6);
        auto cmp = _mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm_movemask_ps(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            if((cmpmask = _mm_movemask_ps(_mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ)))) {
                vmaxv = broadcast_max(divv);
                bestind = ctz(_mm_movemask_ps(_mm_cmp_ps(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
            }
        }
    }
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    double bestv = std::log(std::uniform_real_distribution<double>()(baserng)) / weights[0];
    OMP_PFOR
    for(size_t i = 1; i < n; ++i) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
        if(v > bestv) {
#ifdef _OPENMP
            OMP_CRITICAL {
                if(v > bestv)
#endif
                    bestv = v, bestind = i;
#ifdef _OPENMP
            }
#endif
        }
    }
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    OMP_ONLY(if(rngstates != &baserngstate) std::free(rngstates);)
#endif
    return bestind;
}

template<typename FT>
struct pq_t: public std::priority_queue<std::pair<FT, uint64_t>, std::vector<std::pair<FT, uint64_t>>, std::less<std::pair<FT, uint64_t>>> {
    using value_t = std::pair<FT, uint64_t>;
    using vec_t = std::vector<std::pair<FT, uint64_t>>;
    uint32_t k_;
    pq_t(int k): k_(k) {
        this->c.reserve(k);
    }
    const vec_t &getc() const {return this->c;}
    vec_t &getc() {return this->c;}
    typename vec_t::const_iterator end() const {
        return this->getc().end();
    }
    typename vec_t::const_iterator begin() const {
        return this->getc().begin();
    }
    template<typename...Args>
    void pop_push(Args &&...args) {
        this->pop(); this->push(std::forward<Args>(args)...);
    }
    INLINE void add(std::pair<FT, uint64_t> item) {
        if(this->size() < k_) this->push(item);
        else {
            if(item.first < this->top().first) pop_push(item);
        }
    }
    INLINE void add(FT val, uint64_t id) {add(std::pair<FT, uint64_t>(val, id));}
    void add(const pq_t<FT> &o) {
        for(const auto item: o.getc()) add(item);
    }
};

constexpr const int CMPGQINT = _CMP_LT_OQ;

template<typename FT>
void reduce_pqs(std::vector<pq_t<FT>> &pqs) {
#if _OPENMP
    const size_t npq = pqs.size();
    // Let pqs.size() == 5
    unsigned p2 = 64 - __builtin_clzl(npq);
    // p2 = 3, p2p = 8
    size_t p2p = 1ull << p2;
    for(auto i = 0u; i < p2 - 1; ++i) {
        const auto nper = 1u << (i + 1), step = 1u << i;
        const auto nsets = p2p / nper;
        #pragma omp parallel for schedule(dynamic)
        for(auto j = 0u; j < nsets; ++j) {
            const auto desti = nper * j, srci = desti + step;
            // For step 1, this is one away, for 2, it's 2 away, etc.
            if(srci < pqs.size()) {
                pqs[desti].add(pqs[srci]);
                pqs[srci].getc().clear(); // Free memory
            }
        }
    }
    while(pqs.size() > 1) pqs.pop_back();
#else
    while(pqs.size() > 1) pqs.front().add(pqs.back()), pqs.pop_back();
#endif
}

template<LoadFormat aln>
SIMD_SAMPLING_API int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement)
{
    if(k <= 0) throw std::invalid_argument("k must be > 0");
    wy::WyRand<uint64_t> baserng(seed * seed + 13);
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
    for(auto &i: rngs) i.seed(baserng());
    std::vector<pq_t<double>> pqs;
    while(pqs.size() < (unsigned)nt) pqs.emplace_back(k);
#else
    pq_t<double> basepq(k);
#endif

#ifdef __AVX512F__
    #if __AVX512DQ__
    using simdpcg_t = avx512bis_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        x.state[0] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
        x.inc[1] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
    };
    #else
    using simdpcg_t = avx2_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state[0] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.inc[1] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    #endif
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngstates[tid],
                             baserngstate);
        auto &pq = OMP_ELSE(pqs[tid],
                            basepq);
        __m512i v =
#if __AVX512DQ__
                    avx512bis_pcg32_random_r(&rng);
#else
                    pack_result(avx2_pcg32_random_r(&rng), avx2_pcg32_random_r(&rng));
                    //pack_result(avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng),avx256_pcg32_random_r(&rng),avx256_pcg32_random_r(&rng));
#endif

        // Generate the vector

        const __m512d v2 =
#ifdef __AVX512DQ__
            _mm512_mul_pd(_mm512_cvtepi64_pd(_mm512_srli_epi64(v, 12)), _mm512_set1_pd(LSS_DOUBLE_PDMUL));
#else
            _mm512_mul_pd(_mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(_mm512_srli_epi64(v, 12), _mm512_castpd_si512(_mm512_set1_pd(0x0010000000000000)))), _mm512_set1_pd(0x0010000000000000)),  _mm512_set1_pd(LSS_DOUBLE_PDMUL));
#endif
        // Shift right by 12, convert from ints to doubles, and then multiply by 2^-52
        // resulting in uniform [0, 1] sampling

        const __m512d v3 = Sleef_logd8_u35(v2);
        // Log-transform the [0, 1] sampling
        __m512d ov = load<aln>((const double *)&weights[o * nperel]);;
        auto divv = -_mm512_div_pd(v3, ov);
        int cmpmask;
        if(pq.size() < pq.k_ || (cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, CMPGQINT)) == 0xFFu) {
            #pragma GCC unroll 8
            for(unsigned i = 0; i < 8; ++i)
                pq.add(divv[i], i + o * nperel);
        } else if(cmpmask) {
            switch(__builtin_popcount(cmpmask)) {
                case 7: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 6: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 5: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 4: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 3: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 2: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 1: {auto ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel);}
            }
        } else continue;
        vmaxv = _mm512_set1_pd(pq.top().first);
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p)
        if(weights[p] > 0.)
            pq.add(-std::log(std::uniform_real_distribution<double>()(baserng)) / weights[p], p);
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    __m256d vmaxv = _mm256_set1_pd(-std::numeric_limits<double>::max());
    using simdpcg_t = avx256_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngstates[tid],
                             baserngstate);
        pq_t<double> &pq = OMP_ELSE(pqs[tid],
                                    basepq);
        const size_t onp = o * nperel;
        __m256i v = _mm256_set_m128i(avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng));
        auto v2 = _mm256_or_si256(_mm256_srli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
        auto v5 = Sleef_logd4_u35(_mm256_mul_pd(v3, _mm256_set1_pd(LSS_DOUBLE_PDMUL)));
        auto divv = _mm256_xor_pd(_mm256_div_pd(v5, load<aln>((const double *)&weights[onp])), _mm256_set1_pd(-0.0));
        int cmpmask;
        if(pq.size() < pq.k_ || (cmpmask = _mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, CMPGQINT))) == 0xFu) {
            pq.add(divv[0], onp); pq.add(divv[1], onp + 1); pq.add(divv[2], onp + 2); pq.add(divv[3], onp + 3);
        } else if(cmpmask) {
            switch(__builtin_popcount(cmpmask)) {
                case 3: {int ind = ctz(cmpmask); pq.add(divv[ind], ind + onp); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 2: {int ind = ctz(cmpmask); pq.add(divv[ind], ind + onp); cmpmask ^= (1 << ind);}FALLTHROUGH
                case 1: {int ind = ctz(cmpmask); pq.add(divv[ind], ind + onp);}
            }
        } else continue;
        vmaxv = _mm256_set1_pd(pq.top().first);
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        if(weights[p] > 0.)
            pq.add(-std::log(urd(baserng)) / weights[p], p);
    }
#elif __AVX__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(double);
    const size_t e = n / nperel;
    double maxv = -std::numeric_limits<double>::max();
    __m128d vmaxv = _mm_set1_pd(maxv);
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid],
                             baserng);
        pq_t<double> &pq = OMP_ELSE(pqs[tid],
                                    basepq);
        const size_t onp = o * nperel;
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(LSS_DOUBLE_PDMUL));
        auto v5 = Sleef_logd2_u35(v4);
        auto divv = _mm_xor_pd(_mm_div_pd(v5, load<aln>((const double *) &weights[onp])), _mm_set1_pd(-0.0));
        int cmpmask;
        if(pq.size() < pq.k_ || (cmpmask = _mm_movemask_pd(_mm_cmp_pd(divv, vmaxv, CMPGQINT))) == 3u) {
            pq.add(divv[0], onp);
            pq.add(divv[1], onp + 1);
        } else if(cmpmask) {
            pq.add(divv[cmpmask - 1], onp + cmpmask - 1);
            // branchless equivalent to
            //     if(cmpmask & 1) pq.add(divv[0], onp); else pq.add(divv[1], onp + 1);
            // if cmpmask is 1, then it accesses onp, otherwise onp + 1
            // since the value is only 1 or u
        } else continue;
        vmaxv = _mm_set1_pd(pq.top().first);
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        if(weights[p])
            pq.add(-std::log(std::uniform_real_distribution<double>()(baserng)) / weights[p], p);
    }
#else
    OMP_PFOR
    for(size_t i = 0; i < n; ++i) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[i], baserng);
        auto &pq = OMP_ELSE(pqs[tid], basepq);
        if(weights[i] > 0.)
            pq.add(-std::log(std::uniform_real_distribution<double>()(rng)) / weights[i], i);
    }
#endif
#ifdef _OPENMP
    // We have to merge the priority queues
    // This could be parallelized, but let's assume k is small
    reduce_pqs(pqs);
    VERBOSE_ONLY(std::fprintf(stderr, "lastpq has %zu items (expecting k=%d)\n", pqs.front().size(), k);)
#endif
    auto &rpq = OMP_ELSE(pqs[0], basepq);
    const size_t be = rpq.size();
    int nret;
    if(with_replacement) {
        auto tmp = std::unique_ptr<double[]>(new double[be]);
        auto tmpw = std::unique_ptr<double[]>(new double[be]);
        if(unlikely(tmp == nullptr)) throw std::bad_alloc();
        for(size_t i = 0; i < be; ++i) {
            const auto ind = be - i - 1;
            const auto rrank = rpq.top().first;
            const auto rind = rpq.top().second;
            ret[ind] = rind;
            tmp[ind] = rrank;
            tmpw[ind] = weights[rind];
            rpq.pop();
        }
        auto rettmp = std::unique_ptr<uint64_t[]>(new uint64_t[be]);
        auto rtmp = std::unique_ptr<double[]>(new double[be]);
        std::copy(ret, ret + be, rettmp.get());
        std::copy(tmp.get(), tmp.get() + be, rtmp.get());
        const uint64_t baseseed = OMP_ELSE(rngs.front(), baserng)();
        OMP_PFOR
        for(ssize_t i = 1; i < k; ++i) {
            thread_local wy::WyRand<uint64_t> rng(baseseed + std::hash<std::thread::id>()(std::this_thread::get_id()));
            std::uniform_real_distribution<double> urd;
            double diff = (i < ssize_t(be)) ? tmp[i] - tmp[i - 1]: rtmp[i] - rtmp[i - 1];
            for(ssize_t j = 0; j < i; ++j) {
                auto v = -std::log(urd(rng)) / weights[ret[j]];
                if(v < diff) {
                    rettmp[i] = ret[j];
                    rtmp[i] = diff - v;
                    diff -= v;
                }
            }
        }
        nret = k;
    } else {
        for(size_t i = 0; i < be; ++i) {
            ret[i] = rpq.getc()[i].second;
        }
        nret = be;
    }
    std::sort(ret, ret + nret);
    return nret;
}

template<LoadFormat aln>
SIMD_SAMPLING_API int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed, int with_replacement)
{
    if(k <= 0) throw std::invalid_argument("k must be > 0");
    wy::WyRand<uint64_t> baserng(seed * seed + 13);
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
    for(auto &i: rngs) i.seed(baserng());
    std::vector<pq_t<float>> pqs;
    while(pqs.size() < (unsigned)nt) pqs.emplace_back(k);
#else
    pq_t<float> basepq(k);
#endif

#ifdef __AVX512F__
    #if __AVX512DQ__
    using simdpcg_t = avx512bis_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        x.state[0] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
        x.inc[1] = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
    };
    #else
    using simdpcg_t = avx2_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state[0] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.inc[1] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
    #endif
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 vmaxv = _mm512_set1_ps(std::numeric_limits<float>::max());

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto rngptr = OMP_ELSE(&rngstates[tid],
                               &baserngstate);
        auto &pq = OMP_ELSE(pqs[tid],
                            basepq);
        __m512i v =
#if __AVX512DQ__
                    _mm512_srli_epi32(avx512bis_pcg32_random_r(rngptr), 3);
#else
                    _mm512_srli_epi32(pack_result(avx2_pcg32_random_r(rngptr), avx2_pcg32_random_r(rngptr)), 3);
#endif
        __m512 v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(LSS_FLOAT_PSMUL));
        __m512 v5 = Sleef_logf16_u35(v4);
        __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        auto divv = _mm512_div_ps(v5, lv);
        int cmpmask;
        if(pq.size() < pq.k_ || (cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, CMPGQINT)) == 0xFFFFu) {
            #pragma GCC unroll 16
            for(unsigned i = 0; i < 16u; ++i)
                pq.add(divv[i], i + o * nperel);
        } else if(cmpmask) {
            int ind;
            switch(__builtin_popcount(cmpmask)) {
                case 15: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 14: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 13: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 12: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 11: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 10: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 9: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 8: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 7: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 6: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 5: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 4: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 3: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 2: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 1: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel);}
            }
        } else continue;
        vmaxv = _mm512_set1_ps(pq.top().first);
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        if(weights[p] > 0.)
            pq.add(-std::log(std::uniform_real_distribution<float>()(baserng)) / weights[p], p);
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    __m256 vmaxv = _mm256_set1_ps(std::numeric_limits<float>::max());
#ifdef USE_AVX256_RNG
    using simdpcg_t = avx256_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
#else
    using simdpcg_t = avx2_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.state[0] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.state[1] = _mm256_set_epi64x(baserng(), baserng(), baserng(), baserng());
        x.inc[0] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.inc[1] = _mm256_set_epi64x(baserng() | 1u, baserng() | 1u, baserng() | 1u, baserng() | 1u);
        x.pcg32_mult_l = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) & 0xffffffff);
        x.pcg32_mult_h = _mm256_set1_epi64x(UINT64_C(0x5851f42d4c957f2d) >> 32);
    };
#endif
    simdpcg_t baserngstate;
#ifdef _OPENMP
    simdpcg_t *rngstates = &baserngstate;
    if(nt > 1) {
        if(posix_memalign((void **)&rngstates, sizeof(__m512) / sizeof(char), sizeof(*rngstates) * nt))
            throw std::bad_alloc();
        for(int i = 0; i < nt; ++i) init(rngstates[i]);
    } else
#endif
    {
        init(baserngstate);
    }
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto rngptr = OMP_ELSE(&rngstates[tid], &baserngstate);
        pq_t<float> &pq = OMP_ELSE(pqs[tid], basepq);
#if USE_AVX256_RNG
        __m256i v = _mm256_srli_epi32(_mm256_inserti128_si256(_mm256_castsi128_si256(avx256_pcg32_random_r(rngptr)), avx256_pcg32_random_r(rngptr), 1), 3);
#else
        __m256i v = _mm256_srli_epi32(avx2_pcg32_random_r(rngptr), 3);
#endif
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(LSS_FLOAT_PSMUL));
        const __m256 divv = -_mm256_div_ps(Sleef_logf8_u35(v2), load<aln>((const float *) &weights[o * nperel]));
        int cmpmask, ind;
        if(pq.size() < pq.k_ || (cmpmask = _mm256_movemask_ps(_mm256_cmp_ps(divv, vmaxv, CMPGQINT))) == 0xFF) {
            pq.add(divv[0], o * nperel);     pq.add(divv[1], 1 + o * nperel);
            pq.add(divv[2], 2 + o * nperel); pq.add(divv[3], 3 + o * nperel);
            pq.add(divv[4], 4 + o * nperel); pq.add(divv[5], 5 + o * nperel);
            pq.add(divv[6], 6 + o * nperel); pq.add(divv[7], 7 + o * nperel);
        } else if(cmpmask) {
            switch(__builtin_popcount(cmpmask)) {
                case 7: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 6: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 5: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 4: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 3: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 2: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel); cmpmask ^= (1 << ind);}FALLTHROUGH;
                case 1: {ind = ctz(cmpmask); pq.add(divv[ind], ind + o * nperel);}
            }
        } else continue;
        vmaxv = _mm256_set1_ps(pq.top().first);
    }

    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        if(weights[p] > 0.)
            pq.add(-std::log(urd(baserng)) / weights[p], p);
    }
#else
    OMP_PFOR
    for(size_t i = 0; i < n; ++i) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid], baserng);
        auto &pq = OMP_ELSE(pqs[tid], basepq);
        if(weights[i] > 0.)
            pq.add(-std::log((LSS_FLOAT_PSMUL * rng()) / weights[i]), i);
    }
#endif
#ifdef _OPENMP
    // We have to merge the priority queues
    // This could be parallelized, but let's assume k is small
    reduce_pqs(pqs);
#endif
    auto &rpq = OMP_ELSE(pqs.front(), basepq);
    const size_t be = rpq.size();
    int nret;
    if(with_replacement) {
        auto tmp = std::unique_ptr<float[]>(new float[be]);
        auto tmpw = std::unique_ptr<float[]>(new float[be]);
        for(size_t i = 0; i < be; ++i) {
            const auto ind = be - i - 1;
            auto rrank = rpq.top().first;
            auto rind = rpq.top().second;
            ret[ind] = rind;
            tmp[ind] = rrank;
            tmpw[ind] = weights[rind];
            rpq.pop();
        }
        auto rettmp = std::unique_ptr<uint64_t[]>(new uint64_t[k]);
        auto rtmp = std::unique_ptr<float[]>(new float[k]);
        std::copy(ret, ret + be, rettmp.get());
        std::copy(tmp.get(), tmp.get() + be, rtmp.get());
        const uint64_t baseseed = OMP_ELSE(rngs.front(), baserng)();
        for(ssize_t i = 1; i < k; ++i) {
            thread_local wy::WyRand<uint64_t> rng(baseseed + std::hash<std::thread::id>()(std::this_thread::get_id()));
            std::uniform_real_distribution<float> urd;
            float diff;
            if(i < ssize_t(be)) diff = tmp[i] - tmp[i - 1];
            else                diff = rtmp[i] - rtmp[i - 1];
            for(ssize_t j = 0; j < i; ++j) {
                auto v = -std::log(urd(rng)) / weights[ret[j]];
                if(v < diff) {
                    rettmp[i] = ret[j];
                    rtmp[i] = diff - v;
                    diff -= v;
                }
            }
        }
        std::copy(rettmp.get(), rettmp.get() + k, ret);
        nret = k;
    } else {
        auto tmpp = ret;
        for(const auto &item: rpq.getc())
            *tmpp++ = item.second;
        std::fill(ret + be, ret + k, uint64_t(-1));
        nret = be;
    }
    std::sort(ret, ret + nret);
    return nret;
}


namespace reservoir_simd {


}

#if SIMD_SAMPLING_HIGH_PRECISION || defined(SIMD_SAMPLING_USE_APPROX_LOG)
#undef Sleef_logd2_u35
#undef Sleef_logd4_u35
#undef Sleef_logd8_u35
#undef Sleef_logf4_u35
#undef Sleef_logf8_u35
#undef Sleef_logf16_u35
#endif

extern "C" {

SIMD_SAMPLING_API int simd_sample_get_version() {
    return LIB_SIMDSAMPLING_VERSION;
}
SIMD_SAMPLING_API int simd_sample_get_major_version() {
    return LIB_SIMDSAMPLING_MAJOR;
}
SIMD_SAMPLING_API int simd_sample_get_minor_version() {
    return LIB_SIMDSAMPLING_MINOR;
}
SIMD_SAMPLING_API int simd_sample_get_revision_version() {
    return LIB_SIMDSAMPLING_REVISION;
}

}
