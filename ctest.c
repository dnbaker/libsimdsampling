#include "simdsampling.h"
#include <stdio.h>
int main() {
    const int n = 10000;
    double *p = malloc(n * 8);
    const double RMI = 100. / RAND_MAX;
    // Between 0 and 100 for these
    double psum = 0.;
    for(int i = 0; i < n; ++i) {
        p[i] = RMI * rand();
        psum += p[i];
    }
    int f = dsimd_sample(p, n, 5);
    fprintf(stderr, "Selected item %u, with weight %g\n", f, p[f]);
    p[13] = psum - p[13];
    // Now this item has half of the total weight
    int ntimes = 5000;
    int n13 = 0;
    for(int i = 0; i < ntimes; ++i) {
        n13 += (dsimd_sample(p, n, i + 10000) == 13);
    }
    fprintf(stderr, "Selected point 13 (with half of the weight) %u/%d (%g)\n", n13, ntimes, (double)n13 / ntimes);
    float *fp = (float *)p;
    const int fn = n * 2;
    for(int i = 0; i < ntimes; ++i) {
        n13 += (fsimd_sample(fp, fn, i) == 13);
    }
    free(p);
}
