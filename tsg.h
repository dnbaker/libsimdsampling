#ifndef THREAD_SEEDED_GEN_H__
#define THREAD_SEEDED_GEN_H__
#include <thread>
#include <utility>
#include <cstdio>

namespace tsg {
template<typename RNG>
struct ThreadSeededGen: public RNG {
    template<typename...Args>
    ThreadSeededGen(Args &&...args): RNG(std::forward<Args>(args)...) {
        std::fprintf(stderr, "initializing tsg at %p\n", (void *)this);
        this->seed(std::hash<std::thread::id>{}(std::this_thread::get_id()));
    }
    ThreadSeededGen(uint64_t seed): RNG(seed) {
        std::fprintf(stderr, "initializing tsg at %p with seed = %zu\n", (void *)this, seed);
    }
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) {return RNG::operator()(std::forward<Args>(args)...);}
    template<typename...Args>
    decltype(auto) operator()(Args &&...args) const {return RNG::operator()(std::forward<Args>(args)...);}
};

} // tsg

#endif
