#ifdef _OPENMP
#include "omp.h"
#endif
#include "argminmax.h"
#include "x86intrin.h"
#include "aesctr/wy.h"
#include "ctz.h"
#include "macros.h"
#include <limits>
#include <queue>


#ifdef __AVX512F__
#define LSS_ALIGNMENT (sizeof(__m512) / sizeof(char))
#elif __AVX2__
#define LSS_ALIGNMENT (sizeof(__m256) / sizeof(char))
#elif __SSE2__
#define LSS_ALIGNMENT (sizeof(__m128) / sizeof(char))
#else
#define LSS_ALIGNMENT 1
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

INLINE __m128 broadcast_min(__m128 x) {
    __m128 max1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 max2 = _mm_min_ps(x, max1);
    __m128 max3 = _mm_shuffle_ps(max2, max2, _MM_SHUFFLE(0,0,0,1));
    return _mm_min_ps(max2, max3);
}


enum LoadFormat {
    ALIGNED,
    UNALIGNED
};

// Forward declaratios of core kernels
// Single-sample
template<LoadFormat aln, ArgReduction AR>uint64_t double_argsel_fmt(const double *weights, size_t n);
template<LoadFormat aln, ArgReduction AR> uint64_t float_argsel_fmt(const float * weights, size_t n);


// TODO: add top-k selection methods
extern "C" {
uint64_t fargsel(const float *weights, size_t n, ArgReduction ar)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT;
    uint64_t ret;
    switch((int(aligned) << 1) | (ar == ARGMAX)) {
        case 0: ret = float_argsel_fmt<UNALIGNED, ARGMIN>(weights, n) ;break;
        case 1: ret = float_argsel_fmt<UNALIGNED, ARGMAX>(weights, n) ;break;
        case 2: ret = float_argsel_fmt<ALIGNED, ARGMIN>(weights, n) ;break;
        case 3: ret = float_argsel_fmt<ALIGNED, ARGMAX>(weights, n) ;break;
        default: __builtin_unreachable();
    }
    return ret;
}

uint64_t dargmax(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMAX>(weights, n): double_argsel_fmt<ALIGNED, ARGMAX>(weights, n);
}
uint64_t fargmax(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMAX>(weights, n): float_argsel_fmt<ALIGNED, ARGMAX>(weights, n);
}
uint64_t dargmin(const double *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        double_argsel_fmt<UNALIGNED, ARGMIN>(weights, n): double_argsel_fmt<ALIGNED, ARGMIN>(weights, n);
}
uint64_t fargmin(const float *weights, size_t n) {
    return reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT ?
        float_argsel_fmt<UNALIGNED, ARGMIN>(weights, n): float_argsel_fmt<ALIGNED, ARGMIN>(weights, n);
}


uint64_t dargsel(const double *weights, size_t n, ArgReduction ar)
{
    const bool aligned = reinterpret_cast<uint64_t>(weights) % LSS_ALIGNMENT;
    uint64_t ret;
    switch((int(aligned) << 1) | (ar == ARGMAX)) {
        case 0: ret = double_argsel_fmt<UNALIGNED, ARGMIN>(weights, n) ;break;
        case 1: ret = double_argsel_fmt<UNALIGNED, ARGMAX>(weights, n) ;break;
        case 2: ret = double_argsel_fmt<ALIGNED, ARGMIN>(weights, n) ;break;
        case 3: ret = double_argsel_fmt<ALIGNED, ARGMAX>(weights, n) ;break;
        default: __builtin_unreachable();
    }
    return ret;
}
} // extern "C"


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
#endif
#ifdef __SSE2__
    static INLINE __m128 cmp(__m128 x, __m128 y, std::true_type) {
        return _mm_cmp_ps(x, y, _CMP_GT_OQ);
    }
    static INLINE __m128 cmp(__m128 x, __m128 y, std::false_type) {
        return _mm_cmp_ps(x, y, _CMP_LT_OQ);
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
    static INLINE __m128 max(__m128 x, std::false_type) {
        return broadcast_min(x);
    }
    static INLINE __m128 max(__m128 x, std::true_type) {
        return broadcast_max(x);
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

template<LoadFormat aln, ArgReduction AR>
uint64_t double_argsel_fmt(const double *weights, size_t n)
{
    uint64_t bestind = 0;
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr double STARTVAL = IS_MAX ? -std::numeric_limits<double>::max(): std::numeric_limits<double>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512d) / sizeof(double);
    const size_t e = n / nperel;
    __m512d vmaxv = _mm512_set1_pd(STARTVAL);
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
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        __m256d ov = load<aln>((const double *)&weights[o * nperel]);
        auto cmp = Cmp<AR>::cmp(ov, vmaxv);
        auto cmpmask = _mm256_movemask_pd(cmp);
        if(cmpmask) {
            __m256d y = _mm256_permute2f128_pd(ov, ov, 1);
            __m256d m1 = Cmp<AR>::max(ov, y);
            __m256d m2 = _mm256_permute_pd(m1, 5);
            __m256d newmaxv = Cmp<AR>::max(m1, m2);
            {
                OMP_CRITICAL
                if(_mm256_movemask_pd(Cmp<AR>::cmp(ov, vmaxv))) {
                    vmaxv = newmaxv;
                    bestind = ctz(cmpmask) + o * nperel;
                }
            }
        }
    }
    double maxv = _mm256_cvtsd_f64(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(Cmp<AR>::cmp(weights[p], maxv)) maxv = weights[p], bestind = p;
    }
#else
    double bestv = STARTVAL;
    OMP_PFOR
    for(size_t i = 1; i < n; ++i) {
        auto v = weights[i];
        if(Cmp<AR>::cmp(v,  bestv)) {
            OMP_CRITICAL
            if(Cmp<AR>::cmp(v,  bestv)) bestv = v, bestind = i;
        }
    }
#endif
    return bestind;
}


template<LoadFormat aln, ArgReduction AR>
uint64_t float_argsel_fmt(const float * weights, size_t n)
{
    uint64_t bestind = 0;
    static constexpr bool IS_MAX = AR == ARGMAX;
    static constexpr float STARTVAL = IS_MAX ? -std::numeric_limits<float>::max(): std::numeric_limits<float>::max();
#ifdef __AVX512F__
    constexpr size_t nperel = sizeof(__m512) / sizeof(float);
    const size_t e = n / nperel;
    __m512 vmaxv = _mm512_set1_ps(STARTVAL);
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
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        __m256 divv = load<aln>((const float *) &weights[o * nperel]);
        auto cmp = Cmp<AR>::cmp(divv, vmaxv);
        auto cmpmask = _mm256_movemask_ps(cmp);
        if(cmpmask) {
            const __m256 permHalves = _mm256_permute2f128_ps(divv, divv, 1);
            const __m256 m0 = Cmp<AR>::max(permHalves, divv);
            const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
            const __m256 m1 = Cmp<AR>::max(m0, perm0);
            const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
            const __m256 m2 = Cmp<AR>::max(perm1, m1);
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
    float maxv = _mm256_cvtss_f32(vmaxv);
    for(size_t p = e * nperel; p != n; ++p) {
        if(Cmp<AR>::cmp(weights[p], maxv)) {
            OMP_CRITICAL
            if(Cmp<AR>::cmp(weights[p], maxv))
                maxv = weights[p], bestind = p;
        }
    }
#elif 0
    constexpr size_t nperel = sizeof(__m128d) / sizeof(float);
    const size_t e = n / nperel;
    float maxv = STARTVAL;
    __m128 vmaxv = _mm_set1_ps(maxv);
    OMP_PFOR
    for(size_t o = 0; o < e; ++o) {
        __m128 divv = load<aln>((const float *) &weights[o * nperel]);
        auto cmp = Cmp<AR>::cmp(divv, vmaxv);
        auto cmpmask = _mm_movemask_ps(cmp);
        if(cmpmask) {
            OMP_CRITICAL
            if((cmpmask = _mm_movemask_ps(Cmp<AR>::cmp(divv, vmaxv)))) {
                vmaxv = Cmp<AR>::max(divv);
                bestind = ctz(_mm_movemask_ps(Cmp<AR>::eq(vmaxv, divv))) + o * nperel;
            }
        }
    }
    maxv = vmaxv[0];
    bestind = 0;
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
    OMP_PFOR
    for(size_t i = 1; i < n; ++i) {
        if(Cmp<AR>::cmp(weights[i], bestv)) {
            OMP_CRITICAL
            if(Cmp<AR>::cmp(weights[i], bestv))
                bestv = weights[i], bestind = i;
        }
    }
#endif
    return bestind;
}

