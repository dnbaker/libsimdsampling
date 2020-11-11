#pragma once
#include <x86intrin.h>

#ifndef LSS_ALIGNMENT
#  ifdef __AVX512F__
#  define LSS_ALIGNMENT (sizeof(__m512) / sizeof(char))
#  elif __AVX2__
#  define LSS_ALIGNMENT (sizeof(__m256) / sizeof(char))
#  elif __SSE2__
#  define LSS_ALIGNMENT (sizeof(__m128) / sizeof(char))
#  else
#  define LSS_ALIGNMENT 1
#  endif
#endif


namespace reservoir_simd {

enum LoadFormat {
    ALIGNED,
    UNALIGNED
};

template<typename Func>
INLINE __m128 broadcast_reduce(__m128 x, const Func &func) {
    __m128 m1 = _mm_shuffle_ps(x, x, _MM_SHUFFLE(0,0,3,2));
    __m128 m2 = func(x, m1);
    __m128 m3 = _mm_shuffle_ps(m2, m2, _MM_SHUFFLE(0,0,0,1));
    return func(m2, m3);
}

INLINE __m128 broadcast_max(__m128 x) {
    return broadcast_reduce<decltype(_mm_max_ps)>(x, _mm_max_ps);
}
INLINE __m128 broadcast_min(__m128 x) {
    return broadcast_reduce<decltype(_mm_min_ps)>(x, _mm_min_ps);
}
template<typename Func>
INLINE __m256 broadcast_reduce(__m256 x, const Func &func) {
    const __m256 permHalves = _mm256_permute2f128_ps(x, x, 1);
    const __m256 m0 = func(permHalves, x);
    const __m256 perm0 = _mm256_permute_ps(m0, 0b01001110);
    const __m256 m1 = func(m0, perm0);
    const __m256 perm1 = _mm256_permute_ps(m1, 0b10110001);
    const __m256 m2 = func(perm1, m1);
    return m2;
}

INLINE __m256 broadcast_max(__m256 x) {
    return broadcast_reduce<decltype(_mm256_max_ps)>(x, _mm256_max_ps);
}
INLINE __m256 broadcast_min(__m256 x) {
    return broadcast_reduce<decltype(_mm256_min_ps)>(x, _mm256_min_ps);
}
INLINE __m256 broadcast_mul(__m256 x) {
    return broadcast_reduce<decltype(_mm256_mul_ps)>(x, _mm256_mul_ps);
}
INLINE __m256 broadcast_add(__m256 x) {
    return broadcast_reduce<decltype(_mm256_add_ps)>(x, _mm256_add_ps);
}
template<typename Func>
INLINE __m256d broadcast_reduce(__m256d x, const Func &func) {
    __m256d y = _mm256_permute2f128_pd(x, x, 1);
    __m256d m1 = func(x, y);
    __m256d m2 = _mm256_permute_pd(m1, 5);
    __m256d newmaxv = func(m1, m2);
    return newmaxv;
};
INLINE __m256d broadcast_max(__m256d x) {
    return broadcast_reduce<decltype(_mm256_max_pd)>(x, _mm256_max_pd);
}
INLINE __m256d broadcast_min(__m256d x) {
    return broadcast_reduce<decltype(_mm256_min_pd)>(x, _mm256_min_pd);
}
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
    __m512i ret = _mm512_castsi128_si512(a);
    ret = _mm512_inserti32x4(ret, b, 1);
    ret = _mm512_inserti32x4(ret, c, 2);
    ret = _mm512_inserti32x4(ret, d, 3);
    return ret;
}
#endif
#if __AVX512F__
INLINE __m512i pack_result(__m256i a, __m256i b) {
    __m512i ret = _mm512_castsi256_si512(a);
     ret = _mm512_inserti64x4(ret, b, 1);
    return ret;
}
#endif

} // namespace reservoir_simd
