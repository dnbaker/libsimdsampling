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

// Do we really need to `max(0, weights)`?
// It should keep us robust to slightly off results due to precision
// but it does waste a couple instructions

#ifndef LSS_MAX_0
#define LSS_MAX_0 0
#endif


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

#ifdef __AVX512F__
INLINE __m512 load(const float *ptr, std::false_type) {
#if LSS_MAX_0
    return _mm512_max_ps(_mm512_loadu_ps(ptr), _mm512_setzero_ps());
#else
    return _mm512_loadu_ps(ptr);
#endif
}
INLINE __m512d load(const double *ptr, std::false_type) {
#if LSS_MAX_0
    return _mm512_max_pd(_mm512_loadu_pd(ptr), _mm512_setzero_pd());
#else
    return _mm512_loadu_pd(ptr);
#endif
}
INLINE __m512 load(const float *ptr, std::true_type) {
#if LSS_MAX_0
    return _mm512_max_ps(_mm512_load_ps(ptr), _mm512_setzero_ps());
#else
    return _mm512_load_ps(ptr);
#endif
}
INLINE __m512d load(const double *ptr, std::true_type) {
#if LSS_MAX_0
    return _mm512_max_pd(_mm512_load_pd(ptr), _mm512_setzero_pd());
#else
    return _mm512_load_pd(ptr);
#endif
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
#if LSS_MAX_0
    return _mm256_max_ps(_mm256_loadu_ps(ptr), _mm256_setzero_ps());
#else
    return _mm256_loadu_ps(ptr);
#endif
}
INLINE __m256d load(const double *ptr, std::false_type) {
#if LSS_MAX_0
    return _mm256_max_pd(_mm256_loadu_pd(ptr), _mm256_setzero_pd());
#else
    return _mm256_loadu_pd(ptr);
#endif
}
INLINE __m256 load(const float *ptr, std::true_type) {
#if LSS_MAX_0
    return _mm256_max_ps(_mm256_load_ps(ptr), _mm256_setzero_ps());
#else
    return _mm256_load_ps(ptr);
#endif
}
INLINE __m256d load(const double *ptr, std::true_type) {
#if LSS_MAX_0
    return _mm256_max_pd(_mm256_load_pd(ptr), _mm256_setzero_pd());
#else
    return _mm256_load_pd(ptr);
#endif
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
   __m128d ret = load(ptr, std::integral_constant<bool, aln == ALIGNED>());
#if LSS_MAX_0
    ret = _mm_max_pd(ret, _mm_setzero_pd());
#endif
    return ret;
}
template<LoadFormat aln>
__m128 load(const float *ptr) {
    return load(ptr, std::integral_constant<bool, aln == ALIGNED>());
   __m128 ret = load(ptr, std::integral_constant<bool, aln == ALIGNED>());
#if LSS_MAX_0
    ret = _mm_max_ps(ret, _mm_setzero_ps());
#endif
    return ret;
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
std::vector<uint64_t> double_simd_sampling_wrep_fmt(const double * weights, size_t n, size_t nsample, uint64_t seed);
std::vector<uint64_t> double_simd_sampling_wrep(const double * weights, size_t n, size_t nsample, uint64_t seed) {
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT ?
        double_simd_sampling_wrep_fmt<UNALIGNED>(weights, n, nsample, seed):
        double_simd_sampling_wrep_fmt<ALIGNED>(weights, n, nsample, seed);
}

template<LoadFormat aln>
std::vector<uint64_t> float_simd_sampling_wrep_fmt(const float * weights, size_t n, size_t nsample, uint64_t seed);
std::vector<uint64_t> float_simd_sampling_wrep(const float * weights, size_t n, size_t nsample, uint64_t seed) {
    return reinterpret_cast<uint64_t>(weights) % SIMD_SAMPLING_ALIGNMENT ?
        float_simd_sampling_wrep_fmt<UNALIGNED>(weights, n, nsample, seed):
        float_simd_sampling_wrep_fmt<ALIGNED>(weights, n, nsample, seed);
}

template<LoadFormat aln>
std::vector<uint64_t> float_simd_sampling_wrep_fmt(const float * weights, size_t n, size_t nsample, uint64_t seed)
{
    std::vector<uint64_t> bestinds(nsample);
    std::vector<float> maxv(nsample, -std::numeric_limits<float>::max());
    wy::WyRand<uint64_t> baserng(seed * seed + 13);

    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[nsample]);)
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
    for(auto &i: rngs) i.seed(baserng());
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr float psmul = 1. / (1ull<<29);
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

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngstates[tid],
                             baserngstate);
        const __m512 lv = load<aln>((const float *)&weights[o * nperel]);
        SK_UNROLL_8
        for(size_t mi = 0; mi < nsample; ++mi ) {
            auto &maxv = maxv[mi];
            auto &bestind = bestinds[mi];
            __m512 vmaxv = _mm512_set1_ps(maxv);
            __m512i v = _mm512_srli_epi32(_mm512_inserti32x8(_mm512_castsi256_si512(avx512_pcg32_random_r(&rng)), avx512_pcg32_random_r(&rng), 1), 3);
            auto v4 = _mm512_mul_ps(_mm512_cvtepi32_ps(v), _mm512_set1_ps(psmul));
            auto v5 = Sleef_logf16_u35(v4);
            auto divv = _mm512_div_ps(v5, lv);
            auto cmpmask = _mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ);
            if(cmpmask) {
                const newmaxv = _mm512_set1_ps(_mm512_reduce_max_ps(divv));
                if((cmpmask = _mm512_cmp_ps_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                    OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
                    if(_mm512_cmp_ps_mask(divv, vmaxv, _CMP_GT_OQ)) {
                        maxv = newmaxv[0];
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    }
    float maxv = _mm512_cvtss_f32(vmaxv);
    OMP_PFOR
    for(size_t mi = 0; mi < nsamples; ++mi) {
        std::uniform_real_distribution<float> urd;
        for(size_t p = e * nperel; p != n; ++p) {
            auto &maxv = maxv[mi];
            auto &bestind = bestinds[mi];
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > maxv)
                bestind = p, maxv = v;
        }
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(float);
    const size_t e = (n / nperel);
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
        const __m256 ov6 = load<aln>((const float *) &weights[o * nperel]);
        SK_UNROLL_8
        for(size_t mi = 0; mi < nsample; ++mi) {
            auto &bestind = bestinds[mi];
            auto &bestv = maxv[mi];
            __m256 vmaxv = _mm256_set1_ps(bestv);
            __m256i v = _mm256_srli_epi32(avx2_pcg32_random_r(&rng), 3);
            auto v2 = _mm256_mul_ps(_mm256_cvtepi32_ps(v), _mm256_set1_ps(psmul));
            auto v3 = Sleef_logf8_u35(v2);
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
                OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
                if(_mm256_movemask_ps(_mm256_cmp_ps(m2, vmaxv, _CMP_GT_OQ))) {
                    bestv = m2[0];
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    OMP_PFOR
    for(size_t mi = 0; mi < nsample; ++mi) {
        auto &bestind = bestinds[mi];
        auto &bestv = maxv[mi];
        for(size_t p = e * nperel; p != n; ++p) {
            std::uniform_real_distribution<float> urd;
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > bestv)
                bestind = p, bestv = v;
        }
    }
#elif __AVX__
    constexpr size_t nperel = sizeof(__m128d) / sizeof(float);
    const size_t e = n / nperel;
    float maxv = -std::numeric_limits<float>::max();
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        const __m128 ov6 = load<aln>((const float *) &weights[o * nperel]);
        for(size_t mi = 0; mi < nsample; ++mi) {
            auto &maxv = maxv[mi];
            auto &bestind = bestinds[mi];
            __m128i v = _mm_set_epi64x(rng(), rng());
            const __m128 vmaxv = _mm_set1_ps(maxv);
            auto v3 = _mm_mul_ps(_mm_cvtepi32_ps(v), _mm_set1_ps(psmul));
            auto v5 = Sleef_logf4_u35(v3);
            auto divv = _mm_div_ps(v5, ov6);
            auto cmp = _mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ);
            auto cmpmask = _mm_movemask_ps(cmp);
            if(cmpmask) {
                OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
                if((cmpmask = _mm_movemask_ps(_mm_cmp_ps(divv, vmaxv, _CMP_GT_OQ)))) {
                    maxv = broadcast_max(divv)[0];
                    bestind = ctz(_mm_movemask_ps(_mm_cmp_ps(vmaxv, divv, _CMP_EQ_OQ))) + o * nperel;
                }
            }
        }
    }
    OMP_PFOR
    for(size_t mi = 0; mi < nsample; ++mi) {
        auto &bestind = bestinds[mi];
        auto &bestv = maxv[mi];
        for(size_t p = e * nperel; p != n; ++p) {
            std::uniform_real_distribution<float> urd;
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > maxv)
                bestind = p, maxv = v;
        }
    }
#else
    for(size_t mi = 0; mi < nsample; ++mi)
        maxv[mi] = std::log(std::uniform_real_distribution<double>()(baserng)) / weights[mi];
    OMP_PFOR
    for(size_t i = 1; i < n; ++i) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        for(size_t mi = 0; mi < nsample; ++mi) {
            auto &bestv = maxv[mi];
            auto &beestind = bestinds[mi];
            auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
            if(v > bestv) {
#ifdef _OPENMP
                OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
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
    return bestinds;
}
template<LoadFormat aln>
std::vector<uint64_t> double_simd_sampling_wrep_fmt(const double * weights, size_t n, size_t nsample, uint64_t seed)
{
    std::vector<uint64_t> bestinds(nsample);
    std::vector<double> maxv(nsample, -std::numeric_limits<double>::max());
    wy::WyRand<uint64_t> baserng(seed * seed + 13);

    OMP_ONLY(std::unique_ptr<std::mutex[]> mutexes(new std::mutex[nsample]);)
#ifdef _OPENMP
    int nt;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::vector<wy::WyRand<uint64_t>> rngs(nt);
    for(auto &i: rngs) i.seed(baserng());
#endif
#if defined(__AVX512F__) || defined(__AVX2__)
    constexpr double pdmul = 1. / (1ull<<52);
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
    constexpr size_t nperel = sizeof(__m512) / sizeof(double);
    const size_t e = n / nperel;

    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        OMP_ONLY(const int tid = omp_get_thread_num();)
        auto &rng = OMP_ELSE(rngstates[tid],
                             baserngstate);
        const __m512 lv = load<aln>((const double *)&weights[o * nperel]);
        SK_UNROLL_8
        for(size_t mi = 0; mi < nsample; ++mi ) {
            auto &maxv = maxv[mi];
            auto &bestind = bestinds[mi];
            __m512 vmaxv = _mm512_set1_pd(maxv);
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
			const __m512d v3 = Sleef_logd8_u35(v2);
			auto divv = _mm512_div_pd(v3, lv);
            auto cmpmask = _mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ);
            if(cmpmask) {
                const newmaxv = _mm512_set1_pd(_mm512_reduce_max_pd(divv));
                if((cmpmask = _mm512_cmp_pd_mask(divv, newmaxv, _CMP_EQ_OQ))) {
                    OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
                    if(_mm512_cmp_pd_mask(divv, vmaxv, _CMP_GT_OQ)) {
                        maxv = newmaxv[0];
                        bestind = ctz(cmpmask) + o * nperel;
                    }
                }
            }
        }
    }
    OMP_PFOR
    for(size_t mi = 0; mi < nsamples; ++mi) {
        std::uniform_real_distribution<double> urd;
        auto &maxv = maxv[mi];
        auto &bestind = bestinds[mi];
        for(size_t p = e * nperel; p != n; ++p) {
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > maxv)
                bestind = p, maxv = v;
        }
    }
#elif __AVX2__
    constexpr size_t nperel = sizeof(__m256) / sizeof(double);
    const size_t e = (n / nperel);
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
        const __m256d ov6 = load<aln>((const double *) &weights[o * nperel]);
        SK_UNROLL_8
        for(size_t mi = 0; mi < nsample; ++mi) {
            auto &bestind = bestinds[mi];
            auto &bestv = maxv[mi];
            __m256d vmaxv = _mm256_set1_pd(bestv);
            __m256i v = _mm256_srli_epi32(avx2_pcg32_random_r(&rng), 12);
            auto v2 = _mm256_or_si256(v, _mm256_castpd_si256(_mm256_set1_pd(0x0010000000000000)));
            auto v3 = _mm256_sub_pd(_mm256_castsi256_pd(v2), _mm256_set1_pd(0x0010000000000000));
            auto v4 = _mm256_mul_pd(v3, _mm256_set1_pd(pdmul));
            auto v5 = Sleef_logd4_u35(v4);
            auto divv = _mm256_div_pd(v5, ov6);
            auto cmp = _mm256_cmp_pd(divv, vmaxv, _CMP_GT_OQ);
            auto cmpmask = _mm256_movemask_pd(cmp);
            if(cmpmask) {
                __m256d y = _mm256_permute2f128_pd(divv, divv, 1);                                      
                __m256d m1 = _mm256_max_pd(divv, y);                                                    
                __m256d m2 = _mm256_permute_pd(m1, 5);  
                __m256d newmax = _mm256_max_pd(m2, m1);
                OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
                if(_mm256_movemask_pd(_mm256_cmp_pd(newmax, vmaxv, _CMP_GT_OQ))) {
                    bestv = newmax[0];
                    bestind = ctz(_mm256_movemask_pd(_mm256_cmp_pd(newmax, divv, _CMP_EQ_OQ))) + o * nperel;
                }
            }
        }
    }
    OMP_PFOR
    for(size_t mi = 0; mi < nsample; ++mi) {
        auto &bestind = bestinds[mi];
        auto &bestv = maxv[mi];
        for(size_t p = e * nperel; p != n; ++p) {
            std::uniform_real_distribution<double> urd;
            auto v = std::log(urd(baserng)) / weights[p];
            if(v > bestv)
                bestind = p, bestv = v;
        }
    }
#else
    for(size_t mi = 0; mi < nsample; ++mi)
        maxv[mi] = std::log(std::uniform_real_distribution<double>()(baserng)) / weights[mi];
    OMP_PFOR
    for(size_t i = 1; i < n; ++i) {
        auto &rng = OMP_ELSE(rngs[omp_get_thread_num()],
                             baserng);
        for(size_t mi = 0; mi < nsample; ++mi) {
            auto &bestv = maxv[mi];
            auto &beestind = bestinds[mi];
            auto v = std::log(std::uniform_real_distribution<double>()(rng)) / weights[i];
            if(v > bestv) {
#ifdef _OPENMP
                OMP_ONLY(std::lock_guard<std::mutex[]> lock(mutexes[mi]);)
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
    return bestinds;
}

#ifdef WREP_MAIN
int main() {

std::vector<float> data(1000, 5.);
std::vector<double> ddata(1000, 5.);
for(const auto v: float_simd_sampling_wrep(data.data(), data.size(), 5, 0)) {
    std::fprintf(stderr, "value: %zu\n", size_t(v));
}
for(const auto v: double_simd_sampling_wrep(ddata.data(), ddata.size(), 5, 0)) {
    std::fprintf(stderr, "value: %zu\n", size_t(v));
}
}

#endif
