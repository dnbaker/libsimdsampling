#include "simdsampling.h"
#include <map>
#include <cmath>
#include <cassert>
#include <getopt.h>
#ifdef _OPENMP
#include <omp.h>
#endif

//using simdsample::simd_sampling;
int main(int argc, char **argv) {
#ifdef _OPENMP
    int nt = 1;
    #pragma omp parallel
    {
        nt = omp_get_num_threads();
    }
    std::fprintf(stderr, "Using OpenMP with %u threads\n", nt);
#else
    std::fprintf(stderr, "No OpenMP\n");
#endif
    uint64_t seed = 0;
    unsigned int k = 5;
    int n = 100;
    for(int c;(c = getopt(argc, argv, "k:n:s:h?")) >= 0;) { switch(c) {
        case 'h': std::fprintf(stderr, "Usage: %s [flags]\n-k:k\n-n:n\t-s: seed\n", *argv); return 1;
        case 'k': k = std::atoi(optarg); break;
        case 'n': n = std::atoi(optarg); break;
        case 's': seed = std::strtoull(optarg, nullptr, 10); break;
    }}

    std::fprintf(stderr, "Selecting k = %d\n", k);
    double *ptr = new double[n];
    std::fill(ptr, ptr + n, 1.);
    auto sel = reservoir_simd::sample_k(ptr, n, k, seed);
    auto sel2 = reservoir_simd::sample_k(ptr, n, k * 3, seed + 1, WITH_REPLACEMENT);
    for(const auto v: sel) std::fprintf(stderr, "%u\n", (int)v);
    std::fill((float *)ptr, (float *)ptr + n, 1.);
    auto sel3 = reservoir_simd::sample_k((float *)ptr, n, k, seed);
    auto sel4 = reservoir_simd::sample_k((float *)ptr, n, k * 3, seed + 1, WITH_REPLACEMENT);
    size_t nex = k * 3;
    for(size_t i = 0; i < nex; ++i) std::fprintf(stderr, "fsel[%zu] = %zu\n", i, size_t(sel4[i]));
    for(size_t i = 0; i < nex; ++i) std::fprintf(stderr, "dsel[%zu] = %zu\n", i, size_t(sel2[i]));
    std::map<uint64_t, uint32_t> m, m2;
#if 1
    for(const auto v: sel2) {
        ++m2[v];
        std::fprintf(stderr, "v = %zu, current count = %u\n", size_t(v), m2[v]);
    }
#endif
    for(const auto v: sel4) {
        ++m[v];
    }
    std::map<uint32_t, uint32_t> mc, mc2;
    for(const auto &v: m) ++mc[v.second];
    for(const auto &v: m2) ++mc2[v.second];
    for(const auto &pair: mc) std::fprintf(stderr, "%u:%u\n", int(pair.first), pair.second);
    for(const auto &pair: mc2) std::fprintf(stderr, "[double]%u:%u\n", int(pair.first), pair.second);
    delete[] ptr;
    assert(sel3.size() == k);
    assert(sel.size() == k);
    assert(sel4.size() == unsigned(k * 3) || n < int(k * 3));
    assert(sel2.size() == k * 3);
}
