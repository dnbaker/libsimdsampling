#include "simdsampling.h"
#include <map>
#include <cmath>
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
    int k = 5;
    if(argc > 1) {
        seed = std::atoi(argv[1]);
    }
    if(argc > 2) {
        k = std::atoi(argv[2]);
    }
    std::fprintf(stderr, "Selecting k = %d\n", k);
    const int n = 100000;
    double *ptr = new double[n];
    for(int i = 0; i < n; ++i) {
        ptr[i] = 1.;
    }
    auto sel = reservoir_simd::sample_k(ptr, n, k, seed);
    auto sel2 = reservoir_simd::sample_k(ptr, n, k * 10, seed + 1, WITH_REPLACEMENT);
    for(const auto v: sel) std::fprintf(stderr, "%u\n", (int)v);
    for(int i = 0; i < n * 2; ++i)
        ((float *)ptr)[i] = 1.;
    auto sel3 = reservoir_simd::sample_k((float *)ptr, n * 2, k, seed);
    auto sel4 = reservoir_simd::sample_k((float *)ptr, n * 2, k * 10, seed + 1, WITH_REPLACEMENT);
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
}
