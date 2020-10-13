#ifdef _OPENMP
#include "omp.h"
#endif
#include "x86intrin.h"
#include "ctz.h"
#include "simdsampling.h"
#include "aesctr/wy.h"
#include "sleef.h"
#include <limits>
#include <queue>

#if __AVX512F__ || __AVX2__
#include "simdpcg32.h"
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

// Ensure that we don't divide by 0
template<typename T>
struct INC {
    static constexpr T value = 0;
};
template<> struct INC<float> {static constexpr float value = 1.40129846E-45;};
template<> struct INC<double> {static constexpr double value = 4.9406564584124654e-324;};


INLINE __m128 broadcast_max(__m128 x) {
    __m128 max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 max2 = _mm_max_ps(x, max1);
    __m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0,0,0,1));
    return _mm_max_ps(max2, max3);
}


enum LoadFormat {
    ALIGNED,
    UNALIGNED
};

// Forward declaratios of core kernels
// Single-sample
template<LoadFormat aln>
uint64_t double_simd_sampling_fmt(const double *weights, size_t n, uint64_t seed);
template<LoadFormat aln>
uint64_t float_simd_sampling_fmt(const float *weights, size_t n, uint64_t seed);

// Multiple-sample
template<LoadFormat aln>
int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed);
template<LoadFormat aln>
int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed);

uint64_t dsimd_sample(const double *weights, size_t n, uint64_t seed)
{
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? double_simd_sampling_fmt<UNALIGNED>(weights, n, seed)
        : double_simd_sampling_fmt<ALIGNED>(weights, n, seed);
}

uint64_t fsimd_sample(const float *weights, size_t n, uint64_t seed)
{
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? float_simd_sampling_fmt<UNALIGNED>(weights, n, seed)
        : float_simd_sampling_fmt<ALIGNED>(weights, n, seed);
}

int dsimd_sample_k(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed)
{
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? double_simd_sample_k_fmt<UNALIGNED>(weights, n, k, ret, seed)
        : double_simd_sample_k_fmt<ALIGNED>(weights, n, k, ret, seed);
}

int fsimd_sample_k(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed)
{
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT
        ? float_simd_sample_k_fmt<UNALIGNED>(weights, n, k, ret, seed)
        : float_simd_sample_k_fmt<ALIGNED>(weights, n, k, ret, seed);
}


template<LoadFormat aln>
int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed);
template<LoadFormat aln>
int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed);

#ifdef __AVX512F__
INLINE __m512 load(const float *ptr, std::false_type) {
    return _mm512_loadu_ps(ptr);
}
INLINE __m512d load(const double *ptr, std::false_type) {
    return _mm512_loadu_pd(ptr);
}
INLINE __m512 load(const float *ptr, std::true_type) {
    return _mm512_load_ps(ptr);
}
INLINE __m512d load(const double *ptr, std::true_type) {
    return _mm512_load_pd(ptr);
}
template<LoadFormat aln>
__m512d load(const double *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
template<LoadFormat aln>
__m512 load(const float *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
#elif defined(__AVX2__)
INLINE __m256 load(const float *ptr, std::false_type) {
    return _mm256_loadu_ps(ptr);
}
INLINE __m256d load(const double *ptr, std::false_type) {
    return _mm256_loadu_pd(ptr);
}
INLINE __m256 load(const float *ptr, std::true_type) {
    return _mm256_load_ps(ptr);
}
INLINE __m256d load(const double *ptr, std::true_type) {
    return _mm256_load_pd(ptr);
}
template<LoadFormat aln>
__m256 load(const float *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
template<LoadFormat aln>
__m256d load(const double *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
#elif defined(__SSE2__)
INLINE __m128 load(const float *ptr, std::false_type) {
    return _mm_loadu_ps(ptr);
}
INLINE __m128d load(const double *ptr, std::false_type) {
    return _mm_loadu_pd(ptr);
}
INLINE __m128 load(const float *ptr, std::true_type) {
    return _mm_load_ps(ptr);
}
INLINE __m128d load(const double *ptr, std::true_type) {
    return _mm_load_pd(ptr);
}
template<LoadFormat aln>
__m128d load(const double *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
template<LoadFormat aln>
__m128 load(const float *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
#endif

#if __AVX512F__ && (!defined(__AVX512DQ__) || !__AVX512DQ__)
INLINE __m512i pack_result(__m128i a, __m128i b, __m128i c, __m128i d) {
    __m512i ret = _mm512_setzero_si512();
    ret = _mm512_inserti32x4(ret, a, 0);
    ret = _mm512_inserti32x4(ret, b, 1);
    ret = _mm512_inserti32x4(ret, c, 2);
    ret = _mm512_inserti32x4(ret, d, 3);
    return ret;
}
#endif

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
    constexpr double pdmul = 1. / (1ull<<52);
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngstates[omp_get_thread_num()],
                             baserngstate);
        __m512i v =
#if __AVX512DQ__
                    avx512bis_pcg32_random_r(&rng);
#else
                    pack_result(avx256_pcg32_random_r(&rng), avx256_pcg32_random_r(&rng),avx256_pcg32_random_r(&rng),avx256_pcg32_random_r(&rng));
#endif

        const __m512d v2 =
#ifdef __AVX512DQ__
            _mm512_mul_pd(_mm512_cvtepi64_pd(_mm512_srli_epi64(v, 12)), _mm512_set1_pd(pdmul));
#else
            _mm512_mul_pd(_mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(_mm512_srli_epi64(v, 12), _mm512_castpd_si512(_mm512_set1_pd(0x0010000000000000)))), _mm512_set1_pd(0x0010000000000000)),  _mm512_set1_pd(pdmul));
#endif
        // Shift right by 12, convert from ints to doubles, and then multiply by 2^-52
        // resulting in uniform [0, 1] sampling

        const __m512d v3 = Sleef_logd8_u35(v2);
        // Log-transform the [0, 1] sampling
        __m512d ov = _mm512_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm512_set1_pd(INC<double>::value));
        auto divv = _mm512_div_pd(v3, ov);
        auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(divv));
            if((cmpmask = _mm512_cmp_pd_mask(divv, newmaxv, _CMP_EQ_OQ))) {
            OMP_CRITICAL
                if(_mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ)) {
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
    constexpr double pdmul = 1. / (1ull<<52);
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
        auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
        auto v5 = Sleef_logd4_u35(v4);
        __m256d ov = _mm256_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm256_set1_pd(INC<double>::value));
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
    constexpr double pdmul = 1. / (1ull<<52);
    double maxv = -std::numeric_limits<double>::max();
    __m128d vmaxv = _mm_set1_pd(maxv);
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(pdmul));
        auto v5 = Sleef_logd2_u35(v4);
        __m128d ov6 = _mm_add_pd(load<aln>((const double *) &weights[o * nperel]), _mm_set1_pd(INC<double>::value));
        auto divv = _mm_div_pd(v5, ov6);
        __m128d cmp;
        int cmpmask;
#if __AVX__
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
                vmaxv = _mm_set1_pd(maxv);
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
    OMP_ONLY(if(rngstates != &baserngstate) std::free(rngstates);)
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
    constexpr float psmul = 1. / (1ull<<29);
#ifdef __AVX512F__
    #if __AVX512DQ__
    using simdpcg_t = avx512_pcg32_random_t;
    auto init = [&](simdpcg_t &x) {
        x.multiplier = _mm512_set1_epi64(0x5851f42d4c957f2d);
        x.state = _mm512_set_epi64(baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng(), baserng());
        x.inc = _mm512_set_epi64(baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull, baserng() | 1ull);
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
    __m512 vmaxv = _mm512_set1_ps(-std::numeric_limits<float>::max());
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m512i v = _mm512_srli_epi32(_mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()), 3);
        auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(psmul));
        auto v5 = Sleef_logf16_u35(v4);
        __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        auto ov6 = _mm512_add_ps(lv, _mm512_set1_ps(INC<float>::value));
        auto divv = _mm512_div_ps(v5, ov6);
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
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(psmul));
#ifndef NDEBUG
        float sum = 0.;
        for(size_t i = 0; i < sizeof(v) / sizeof(uint32_t); ++i) {
            float nextv;
            std::memcpy(&nextv, (float *)&v2 + i, sizeof(nextv));
            sum += nextv;
        }
#endif
        auto v3 = Sleef_logf8_u35(v2);
        __m256 ov6 = _mm256_add_ps(load<aln>((const float *) &weights[o * nperel]), _mm256_set1_ps(INC<float>::value));
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
        auto v3 = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(psmul));
        auto v5 = Sleef_logf4_u35(v3);
        __m128 ov6 = _mm_max_ps(load<aln>((const float *) &weights[o * nperel]), _mm_set1_ps(INC<float>::value));
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
    OMP_ONLY(if(rngstates != &baserngstate) std::free(rngstates);)
    return bestind;
}

template<typename FT>
struct pq_t: public std::priority_queue<std::pair<FT, uint64_t>, std::vector<std::pair<FT, uint64_t>>, std::greater<std::pair<FT, uint64_t>>> {
    using value_t = std::pair<FT, uint64_t>;
    using vec_t = std::vector<std::pair<FT, uint64_t>>;
    uint32_t k_;
    pq_t(int k): k_(k) {
        this->c.reserve(k);
    }
    INLINE void add(FT val, uint64_t id) {
        if(this->size() < k_) {
            this->push(value_t(val, id));
        } else if(val > this->top().first) {
            this->pop();
            this->push(value_t(val, id));
        }
    }
};

template<LoadFormat aln>
int double_simd_sample_k_fmt(const double *weights, size_t n, int k, uint64_t *ret, uint64_t seed)
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

    constexpr double pdmul = 1. / (1ull<<52);
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid],
                             baserng);
        auto &pq = OMP_ELSE(pqs[tid],
                            basepq);
        __m512i v = _mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng());
        // Generate the vector

        const __m512d v2 =
#ifdef __AVX512DQ__
            _mm512_mul_pd(_mm512_cvtepi64_pd(_mm512_srli_epi64(v, 12)), _mm512_set1_pd(pdmul));
#else
            _mm512_mul_pd(_mm512_sub_pd(_mm512_castsi512_pd(_mm512_or_si512(_mm512_srli_epi64(v, 12), _mm512_castpd_si512(_mm512_set1_pd(0x0010000000000000)))), _mm512_set1_pd(0x0010000000000000)),  _mm512_set1_pd(pdmul));
#endif
        // Shift right by 12, convert from ints to doubles, and then multiply by 2^-52
        // resulting in uniform [0, 1] sampling

        const __m512d v3 = Sleef_logd8_u35(v2);
        // Log-transform the [0, 1] sampling
        __m512d ov = _mm512_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm512_set1_pd(INC<double>::value));
        auto divv = _mm512_div_pd(v3, ov);
        auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            for(;;) {
                auto ind = ctz(cmpmask);
                pq.add(((double *)&divv)[ind], ind + o * nperel);
                cmpmask ^= (1 << ind);
                if(cmpmask == 0) {
                    vmaxv = _mm512_set1_pd(pq.top().first);
                    break;
                }
            }
        }
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        if(weights[p] > 0.)
            pq.add(std::log(urd(baserng)) / weights[p], p);
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    __m256d vmaxv = _mm256_set1_pd(-std::numeric_limits<double>::max());
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid],
                             baserng);
        pq_t<double> &pq = OMP_ELSE(pqs[tid],
                                    basepq);
        __m256i v = _mm256_set_epi64x(rng(), rng(), rng(), rng());
        auto v2 = _mm256_or_si256(_mm256_srli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
        auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
        auto v5 = Sleef_logd4_u35(v4);
        __m256d ov = _mm256_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm256_set1_pd(INC<double>::value));
        auto divv = _mm256_div_pd(v5, ov);
        auto cmp = _mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_pd(cmp);
        if(cmpmask) {
            for(;;) {
                auto ind = ctz(cmpmask);
                pq.add(((double *)&divv)[ind], ind + o * nperel);
                cmpmask ^= (1 << ind);
                if(!cmpmask) {
                    vmaxv = _mm256_set1_pd(pq.top().first);
                    break;
                }
            }
        }
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        if(weights[p] > 0.)
            pq.add(std::log(urd(baserng)) / weights[p], p);
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
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(pdmul));
        auto v5 = Sleef_logd2_u35(v4);
        __m128d ov6 = _mm_add_pd(load<aln>((const double *) &weights[o * nperel]), _mm_set1_pd(INC<double>::value));
        auto divv = _mm_div_pd(v5, ov6);
        auto cmp = _mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        const auto cmpmask = _mm_movemask_pd(cmp);
        if(cmpmask) {
            switch(cmpmask) {
            case 1:
                pq.add(((double *)&divv)[0], o * nperel); break;
            case 2:
                pq.add(((double *)&divv)[1], o * nperel + 1); break;
            case 3:
                pq.add(((double *)&divv)[0], o * nperel);
                pq.add(((double *)&divv)[1], o * nperel + 1); break;
            }
            vmaxv = _mm_set1_pd(pq.top().first);
        }
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        if(weights[p]) {
            std::uniform_real_distribution<double> urd;
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > pq.top().first)
                pq.add(v, p);
        }
    }
#else
    OMP_PFOR
    for(size_t i = 0; i < n; ++i) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &pq = OMP_ELSE(pqs[tid], basepq);
        if(weights[i] > 0.) {
            auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
            pq.add(v, i);
        }
    }
#endif
#ifdef _OPENMP
    // We have to merge the priority queues
    // This could be parallelized, but let's assume k is small
    auto &lastpq = pqs[0];
    while(pqs.size() > 1) {
        auto &p = pqs.back();
        while(p.size()) {
            lastpq.push(p.top());
            p.pop();
            lastpq.pop();
        }
        pqs.pop_back();
    }
    const size_t be = lastpq.size();
    for(size_t i = 0; i < be; ++i) {
        ret[i] = lastpq.top().second; lastpq.pop();
    }
#else
    const size_t be = basepq.size();
    for(size_t i = 0; i < be; ++i) {
        ret[i] = basepq.top().second;
        basepq.pop();
    }
#endif
    std::sort(ret, ret + be);
    return static_cast<int>(be);
}
template<LoadFormat aln>
int float_simd_sample_k_fmt(const float *weights, size_t n, int k, uint64_t *ret, uint64_t seed) {
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

    constexpr float psmul = 1. / (1ull<<29);
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 vmaxv = _mm512_set1_ps(-std::numeric_limits<float>::max());
    size_t o;

    OMP_PFOR
    for(o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid],
                             baserng);
        auto &pq = OMP_ELSE(pqs[tid],
                            basepq);
        __m512i v = _mm512_srli_epi32(_mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng()), 3);
        auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(psmul));
        auto v5 = Sleef_logf16_u35(v4);
        __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        auto ov6 = _mm512_add_ps(lv, _mm512_set1_ps(INC<float>::value));
        auto divv = _mm512_div_ps(v5, ov6);
        auto cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            for(;;) {
                auto ind = ctz(cmpmask);
                pq.add(((float *)&divv)[ind], ind + o * nperel);
                cmpmask ^= (1 << ind);
                if(cmpmask == 0) {
                    vmaxv = _mm512_set1_ps(pq.top().first);
                    break;
                }
            }
        }
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        if(weights[p] > 0.)
            pq.add(std::log(urd(baserng)) / weights[p], p);
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    __m256 vmaxv = _mm256_set1_ps(-std::numeric_limits<float>::max());
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid],
                             baserng);
        pq_t<float> &pq = OMP_ELSE(pqs[tid],
                                    basepq);
        __m256i v = _mm256_srli_epi32(_mm256_set_epi64x(rng(), rng(), rng(), rng()), 3);
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(psmul));
        auto v3 = Sleef_logf8_u35(v2);
        __m256 ov6 = _mm256_add_ps(load<aln>((const float *) &weights[o * nperel]), _mm256_set1_ps(INC<float>::value));
        auto divv = _mm256_div_ps(v3, ov6);
        auto cmp = _mm256_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_ps(cmp);
        if(cmpmask) {
            for(;;) {
                auto ind = ctz(cmpmask);
                pq.add(((float *)&divv)[ind], ind + o * nperel);
                cmpmask ^= (1 << ind);
                if(!cmpmask) {
                    vmaxv = _mm256_set1_ps(pq.top().first);
                    break;
                }
            }
        }
    }
    auto &pq = OMP_ELSE(pqs[0], basepq);
    for(size_t p = e * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        if(weights[p] > 0.)
            pq.add(std::log(urd(baserng)) / weights[p], p);
    }
#else
    OMP_PFOR
    for(size_t i = 0; i < n; ++i) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngs[tid], baserng);
        auto &pq = OMP_ELSE(pqs[tid], basepq);
        if(weights[i] > 0.) {
            auto v = std::log((psmul * rng()) / weights[i]);
            pq.add(v, i);
        }
    }
#endif
#ifdef _OPENMP
    // We have to merge the priority queues
    // This could be parallelized, but let's assume k is small
    auto &lastpq = pqs[0];
    for(size_t i = 1; i < pqs.size(); ++i) {
        auto &p = pqs[i];
        while(p.size()) {
            lastpq.push(p.top());
            p.pop();
            lastpq.pop();
        }
    }
    const size_t be = lastpq.size();
    for(size_t i = 0; i < be; ++i) {
        ret[i] = lastpq.top().second; lastpq.pop();
    }
#else
    const size_t be = basepq.size();
    for(size_t i = 0; i < be; ++i) {
        ret[i] = basepq.top().second;
        basepq.pop();
    }
#endif
    std::sort(ret, ret + be);
    return static_cast<int>(be);
}

int simd_sample_get_version() {
    return LIB_SIMDSAMPLING_VERSION;
}
int simd_sample_get_major_version() {
    return LIB_SIMDSAMPLING_MAJOR;
}
int simd_sample_get_minor_version() {
    return LIB_SIMDSAMPLING_MINOR;
}
int simd_sample_get_revision_version() {
    return LIB_SIMDSAMPLING_REVISION;
}
