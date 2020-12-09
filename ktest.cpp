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
    size_t heavyw = 100000;
    ptr[13] = heavyw;
    size_t nex = k * 1.1;
    auto sel = reservoir_simd::sample_k(ptr, n, k, seed);
    auto sel2 = reservoir_simd::sample_k(ptr, n, nex, seed + 1, WITH_REPLACEMENT);
    for(const auto v: sel) std::fprintf(stderr, "%u\n", (int)v);
    std::fill((float *)ptr, (float *)ptr + n, 1.);
    ptr[13] = heavyw;
    auto sel3 = reservoir_simd::sample_k((float *)ptr, n, k, seed);
    auto sel4 = reservoir_simd::sample_k((float *)ptr, n, nex, seed + 1, WITH_REPLACEMENT);
    for(size_t i = 0; i < sel.size(); ++i) std::fprintf(stderr, "fsel[%zu] = %zu\n", i, size_t(sel4[i]));
    for(size_t i = 0; i < sel2.size(); ++i) std::fprintf(stderr, "dsel[%zu] = %zu\n", i, size_t(sel2[i]));
    std::map<uint64_t, uint32_t> m, m2;
#if 1
    for(const auto v: sel2) {
        ++m2[v];
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
    std::fprintf(stderr, "sizes of sels: %zu, %zu, %zu, %zu\n", sel.size(), sel2.size(), sel3.size(), sel4.size());
    assert(sel3.size() == k);
    assert(sel.size() == k);
    assert(sel4.size() == unsigned(nex) || n < int(nex));
    assert(sel2.size() == nex || n < int(nex) || !std::fprintf(stderr, "sel.size() = %zu (vs expected) %zu\n", sel.size(), nex));
}
