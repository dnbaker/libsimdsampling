#ifdef _OPENMP
#include "omp.h"
#endif
#include "simdsampling.h"
#include "aesctr/wy.h"
#include "sleef.h"
#include "tsg.h"


#ifdef __AVX512F__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m512) / sizeof(char))
#elif __AVX__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m256) / sizeof(char))
#elif __SSE2__
#define SIMD_SAMPLING_ALIGNMENT (sizeof(__m128) / sizeof(char))
#else
#define SIMD_SAMPLING_ALIGNMENT 1
#endif

// Ensure that we don't divide by 0
template<typename T> constexpr T INC = 0;
template<> constexpr double INC<double> = 4.9406564584124654e-324;
template<> constexpr float INC<float> = 1.40129846E-45;


enum LoadFormat {
    ALIGNED,
    UNALIGNED
};
template<LoadFormat aln>
uint64_t double_simd_sampling_fmt(const double *weights, size_t n, uint64_t seed);
template<LoadFormat aln>
uint64_t float_simd_sampling_fmt(const float *weights, size_t n, uint64_t seed);

uint64_t double_simd_sampling(const double *weights, size_t n, uint64_t seed)
{
    if(reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT == 0ul) {
        return double_simd_sampling_fmt<ALIGNED>(weights, n, seed);
    } else {
        return double_simd_sampling_fmt<UNALIGNED>(weights, n, seed);
    }
}

uint64_t float_simd_sampling(const float *weights, size_t n, uint64_t seed)
{
    if(reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT == 0ul) {
        return float_simd_sampling_fmt<ALIGNED>(weights, n, seed);
    } else {
        return float_simd_sampling_fmt<UNALIGNED>(weights, n, seed);
    }
}

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
#elif defined(__AVX__) || defined(__AVX2__)
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
    return _mm128_loadu_ps(ptr);
}
INLINE __m128d load(const double *ptr, std::false_type) {
    return _mm128_loadu_pd(ptr);
}
INLINE __m128 load(const float *ptr, std::true_type) {
    return _mm128_load_ps(ptr);
}
INLINE __m128d load(const double *ptr, std::true_type) {
    return _mm128_load_pd(ptr);
}
template<LoadFormat aln>
__m128d load(const double *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
}
#endif

template<LoadFormat aln>
uint64_t double_simd_sampling_fmt(const double *weights, size_t n, uint64_t seed)
{
    uint64_t bestind;
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
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    constexpr double pdmul = 1. / (1ull<<52);
    bestind = 0;
    __m512d vmaxv = _mm512_set1_pd(-std::numeric_limits<double>::max());
    size_t o;

    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
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
        __m512d ov = _mm512_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm512_set1_pd(INC<double>));
        auto divv = _mm512_div_pd(v3, ov);
        auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(divv));
            if((cmpmask = _mm512_cmp_pd_mask(divv, newmaxv, _CMP_EQ_OQ))) {
            OMP_CRITICAL
                if(_mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ)) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm512_cvtsd_f64(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256d) / sizeof(double);
    const size_t e = (n / nperel);
    constexpr double pdmul = 1. / (1ull<<52);
    bestind = 0;
    __m256d vmaxv = _mm256_set1_pd(-std::numeric_limits<double>::max());
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m256i v = _mm256_set_epi64x(rng(), rng(), rng(), rng());
        auto v2 = _mm256_or_si256(_mm256_srli_epi64(v, 12), _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
        auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
        auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
        auto v5 = Sleef_logd4_u35(v4);
        __m256d ov = _mm256_add_pd(load<aln>((const double *)&weights[o * nperel]), _mm256_set1_pd(INC<double>));
        auto divv = _mm256_div_pd(v5, ov);
        auto cmp = _mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm256_movemask_pd(cmp);
        if(cmpmask) {
            __m256d y = _mm256_permute2f128_pd(divv, divv, 1);
            __m256d m1 = _mm256_max_pd(divv, y);
            __m256d m2 = _mm256_permute_pd(m1, 5);
            auto newmaxv = _mm256_max_pd(m1, m2);
            cmpmask = _mm256_movemask_pd(_mm256_cmp_pd(divv, newmaxv, _CMP_EQ_OQ));
            if(_mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ))) {
                OMP_CRITICAL
                if(_mm256_movemask_pd(_mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ))) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm256_cvtsd_f64(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
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
    bestind = 0;
    __m128d vmaxv = _mm_set1_pd(maxv);
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m128i v = _mm_set_epi64x(rng(), rng());
        auto v2 = _mm_or_si128(_mm_srli_epi64(v, 12), _mm_castpd_si128(_mm_set1_pd(0x0010000000000000)));
        auto v3 = _mm_sub_pd(_mm_castsi128_pd(v2), _mm_set1_pd(0x0010000000000000));
        auto v4 = _mm_mul_pd(v3, _mm_set1_pd(pdmul));
        auto v5 = Sleef_logd2_u35(v4);
        __m128d ov6 = _mm_add_pd(load<aln>((const double *) &weights[o * nperel]), _mm_set1_pd(INC<double>));
        auto divv = _mm_div_pd(v5, ov6);
        auto cmp = _mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm_movemask_pd(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            cmpmask = _mm_movemask_pd(_mm_cmp_pd(divv, vmaxv, _CMP_GT_OQ));
            if(cmpmask) {
                vmaxv = _mm_max_pd(divv, _mm_permute_pd(divv, 1));
                bestind = __builtin_ctz(_mm_movemask_pd(_mm_cmp_pd(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
            }
        }
    }
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<double> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    bestind = 0;
    double bestv = std::log(std::uniform_real_distribution<double>()(rng)) / weights[0];
    for(size_t i = 1; i < n; ++i) {
        auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
        if(v > bestv) bestv = v, bestind = i;
    }
#endif
    return bestind;
}


template<LoadFormat aln>
uint64_t float_simd_sampling_fmt(const float * weights, size_t n, uint64_t seed)
{
    uint64_t bestind;
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
    constexpr size_t nperel = sizeof(__m512d) / sizeof(float);
    const size_t e = n / nperel;
    constexpr float psmul = 1. / (1ull<<32);
    bestind = 0;
    __m512 vmaxv = _mm512_set1_ps(-std::numeric_limits<float>::max());
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m512i v = _mm512_set_epi64(rng(), rng(), rng(), rng(), rng(), rng(), rng(), rng());
        auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(psmul));
        auto v5 = Sleef_logf16_u35(v4);
        __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        auto ov6 = _mm512_add_ps(lv, _mm512_set1_ps(INC<float>));  
        auto divv = _mm512_div_ps(v5, ov6);
        auto cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ);
        if(cmpmask) {
            auto newmaxv = _mm512_set1_ps(_mm512_reduce_max_ps(divv));
            if((cmpmask = _mm512_cmp_ps_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                OMP_CRITICAL
                if(_mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ)) {
                    vmaxv = newmaxv;
                    bestind = __builtin_ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    float maxv = _mm512_cvtss_f32(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
    constexpr float psmul = 1. / (1ull<<32);
    bestind = 0;
    __m256 vmaxv = _mm256_set1_ps(-std::numeric_limits<float>::max());
    size_t o = 0;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m256i v = _mm256_set_epi64x(rng(), rng(), rng(), rng());
        auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(psmul));
        auto v3 = Sleef_logf8_u35(v2);
        __m256 ov6 = _mm256_add_ps(load<aln>((const float *) &weights[o * nperel]), _mm256_set1_ps(INC<float>));
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
                bestind = __builtin_ctz(cmpmask) + o * nperel;
            }
        }
    }
    float maxv = _mm256_cvtss_f32(vmaxv);
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#elif __SSE2__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(float);
    const size_t e = n / nperel;
    constexpr float psmul = 1. / (1ull<<32);
    float maxv = -std::numeric_limits<float>::max();
    bestind = 0;
    __m128 vmaxv = _mm_set1_ps(maxv);
    size_t o;
    OMP_PFOR
    for(o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        __m128i v = _mm_set_epi64(rng(), rng());
        auto v3 = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(psmul));
        auto v5 = Sleef_logf4_u35(v3);
        __m128 ov6 = _mm_max_ps(load<aln>((const float *) &weights[o * nperel]), _mm_set1_ps(INC<float>));
        auto divv = _mm_div_ps(v5, ov6);
        auto cmp = _mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
        auto cmpmask = _mm_movemask_ps(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            if((cmpmask = _mm_movemask_ps(_mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ)))) {
                vmaxv = _mm_set1_ps(horizontal_max(divv));
                bestind = __builtin_ctz(_mm_movemask_ps(_mm_cmp_ps(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
            }
        }
    }
    for(size_t p = o * nperel; p != n; ++p) {
        std::uniform_real_distribution<float> urd;
        auto v = std::log(urd(baserng)) / weights[p];
        if(v > maxv)
            bestind = p, maxv = v;
    }
#else
    bestind = 0;
    double bestv = std::log(std::uniform_real_distribution<double>()(baserng)) / weights[0];
    for(size_t i = 1; i < n; ++i) {
        auto v = std::log(std::uniform_real_distribution<double>()(baserng)) / weights[i];
        if(v > bestv) bestv = v, bestind = i;
    }
#endif
    return bestind;
}
