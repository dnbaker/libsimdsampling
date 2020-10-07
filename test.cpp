#include "simdsampling.h"
#ifdef _OPENMP
#include <omp.h>
#endif

//using simdsample::simd_sampling;
int main() {
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
    const int n = 100000;
    double *ptr = new double[n];
    auto sel = simd_sampling(ptr, n);
    delete[] ptr;
}
