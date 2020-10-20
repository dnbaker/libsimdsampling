#ifndef _SAMPLING_SHARED_H__
#define _SAMPLING_SHARED_H__
#include <vector>
#include <random>
#include <type_traits>
#include <cstdlib>
#include <cstdint>
#include <cassert>
#include "./div.h"
#include "aesctr/wy.h"

namespace DOGS {
inline namespace shared {
using std::size_t;
using std::int64_t;
using std::uint64_t;

template<typename C>
struct Emplacer {
    static void reserve(C &c, size_t n) {
        c.reserve(n);
    }
    template<typename Item>
    static void insert(C &c, Item &&item) {
        c.emplace_back(std::move(item));
    }
};


template<typename T>
struct pass_by_val_faster: public std::integral_constant<bool, (sizeof(T) & 15 == 0) || !std::is_trivially_copyable<T>::value> {};
#if __cplusplus >= 201403L
template<typename T>
bool pass_by_val_faster_v = pass_by_val_faster<T>::value;
#endif
} // shared
} // DOGS

#endif /* _SAMPLING_SHARED_H__ */
