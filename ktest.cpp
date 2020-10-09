#include "simdsampling.h"
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
    if(argc > 1) {
        seed = std::atoi(argv[1]);
    }
    const int n = 100000;
    double *ptr = new double[n];
    for(int i = 0; i < n; ++i) {
        ptr[i] = 1.;
    }
    auto sel = reservoir_simd::sample_k(ptr, n, 5, seed);
    for(const auto v: sel) std::fprintf(stderr, "%u\n", (int)v);
    delete[] ptr;
}
