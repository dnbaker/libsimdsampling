#include "simdsampling.h"
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
    std::cauchy_distribution<FLOAT_TYPE> cd;
    wy::WyRand<uint64_t> rng(13);
    for(int i = 0; i < n; ++i) {
        ptr[i] = std::abs(cd(rng));
    }
    auto sel = reservoir_simd::sample(ptr, n, seed);
    std::fprintf(stderr, "Selected: %u with weight %g\n", int(sel), ptr[sel]);
    delete[] ptr;
    return EXIT_SUCCESS;
}
