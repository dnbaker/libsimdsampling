#include "simdsampling.h"
#include <chrono>
#include <random>
#include "aesctr/wy.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//using simdsample::simd_sampling;

#ifndef FLOAT_TYPE
#define FLOAT_TYPE double
#endif

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
    if(argc > 1) {
        seed = std::strtoull(argv[1], nullptr, 10);
    }
    const int n = 100000;
    FLOAT_TYPE *ptr = new FLOAT_TYPE[n];
    std::exponential_distribution<FLOAT_TYPE> cd;
    wy::WyRand<uint64_t> rng(13);
    for(int i = 0; i < n; ++i) {
        if(i % 7 == 0) ptr[i] = 0.;
        else ptr[i] = cd(rng);
    }
    std::vector<uint64_t> vals(5);
    auto t = std::chrono::high_resolution_clock::now();
    for(size_t i = 0; i < 5; ++i) {
        vals[i] = reservoir_simd::sample(ptr, n, seed + i);
    }
    auto e = std::chrono::high_resolution_clock::now();
    std::fprintf(stderr, "Selected: %u with weight %g\n", int(vals[0]), ptr[vals[0]]);
    std::fprintf(stderr, "time: %gms\n", std::chrono::duration<double, std::milli>(e - t).count());
    delete[] ptr;
    return EXIT_SUCCESS;
}
