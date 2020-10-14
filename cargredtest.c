#include <assert.h>
#include <argminmax.h>
#include <stdio.h>


int main() {
    size_t nd = 1000000;
    double *v = malloc(nd * sizeof(v));
    for(size_t i = 0; i < nd; ++i)
        v[i] = (double)rand() / RAND_MAX;
    uint32_t maxind = 100000;
    uint32_t minind = 17371;
    v[maxind] = 122323.;
    v[minind] = -133000;
    size_t argmin = dargsel(v, nd, ARGMIN);
    size_t argmax = dargsel(v, nd, ARGMAX);
    fprintf(stderr, "argmax %zu, expected %u\n", argmax, maxind);
    fprintf(stderr, "argmin %zu, expected %u\n", argmin, minind);
    assert(argmax == maxind);
    assert(argmin == minind);
    float *ptr = (float *)&v[0];
    size_t nf = nd * 2;
    for(size_t i = 0; i < nf; ++i) {
        ptr[i] = (float)rand() / RAND_MAX;
    }
    unsigned randmaxind = rand() % nf;
    unsigned randminind = rand() % nf;
    ptr[randmaxind] = 122323.;
    ptr[randminind] = -133000;
    fprintf(stderr, "randmaxind %u, randminind %u\n", randmaxind, randminind);
    size_t fargmin = fargsel(ptr, nf, ARGMIN);
    size_t fargmax = fargsel(ptr, nf, ARGMAX);
    fprintf(stderr, "found maxind %zu, found minind %zu\n", fargmax, fargmin);
    assert(randmaxind == fargmax);
    assert(randminind == fargmin);
    free(v);
}
