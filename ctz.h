#ifndef CTZ_H__
#define CTZ_H__
#include "macros.h"
#include <limits>
#include <climits>
#include <cstdint>
#ifdef _MSC_VER
#include <intrin.h>
#endif


template<typename T>
INLINE unsigned ctz(T x) {
    int ret;
    if(x) {
        x = (x ^ (x - 1)) >> 1;
        for(ret = 0;x; ++ret) {
            x >>= 1;
        }
    } else {
        return sizeof(T) * CHAR_BIT;
    }
}

template<> INLINE unsigned ctz<uint32_t>(uint32_t x) {
#ifdef __GNUC__
    return __builtin_ctz(x);
#elif _MSC_VER
    unsigned ret;
    if(!_BitScanForward(&ret, x)) ret = sizeof(x) * CHAR_BIT;
    return ret;
#else
    static constexpr const uint8_t Mod37BitPosition[] = // map a bit value mod 37 to its position
{
  32, 0, 1, 26, 2, 23, 27, 0, 3, 16, 24, 30, 28, 11, 0, 13, 4,
  7, 17, 0, 25, 22, 31, 15, 29, 10, 12, 6, 0, 21, 14, 9, 5,
  20, 8, 19, 18
};
    // From https://graphics.stanford.edu/~seander/bithacks.html
    // but changing the table from uint32_t to uint8_t to be more compact
    return Mod37BitPosition[(-x & x) % 37];
#endif
}

template<> INLINE unsigned ctz<uint16_t>(uint16_t x) {
    return ctz(static_cast<unsigned>(x));
}
template<> INLINE unsigned ctz<int16_t>(int16_t x) {
    return ctz(static_cast<uint16_t>(x));
}
template<> INLINE unsigned ctz<uint8_t>(uint8_t x) {
#ifdef __GNUC__
    return ctz(static_cast<unsigned>(x));
#else
    unsigned ret;
    if(x & 1) {
        ret = 0;
    } else {
        static constexpr const uint8_t lut[] {0, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 7, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 6, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1, 5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1};
        return lut[ret >> 1];
    }
    return ret;
#endif
}

template<> INLINE unsigned ctz<int8_t>(int8_t x) {
    return ctz(static_cast<uint8_t>(x));
}

#ifdef __GNUC__
template<> INLINE unsigned ctz<unsigned long long>(unsigned long long x) {
    return __builtin_ctzll(x);
}
template<> INLINE unsigned ctz<unsigned long>(unsigned long x) {
    return __builtin_ctzl(x);
}
#endif

template<> INLINE unsigned ctz<int>(int x) {return ctz(static_cast<unsigned>(x));}
template<> INLINE unsigned ctz<long>(long x) {return ctz(static_cast<long unsigned>(x));}
template<> INLINE unsigned ctz<long long>(long long x) {return ctz(static_cast<long long unsigned>(x));}

#endif
