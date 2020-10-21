#include <vector>
#include <cassert>
#include <argminmax.h>
#include <cstdio>
#include <string>


int main(int argc, char **argv) {
    size_t seed = argc == 1 ?  int(std::hash<std::string>()(argv[0])) : std::atoi(argv[1]) ? std::atoi(argv[1]): int(std::hash<std::string>()(argv[1]));
    std::srand(seed);
    std::vector<double> v(1000000);
    for(auto &i: v) i = double(std::rand()) / RAND_MAX;
    uint32_t maxind = 100000;
    uint32_t minind = 17371;
    v[maxind] = 122323.;
    v[minind] = -133000;
    size_t argmin = dargsel(v.data(), v.size(), ARGMIN, true);
    size_t argmax = dargsel(v.data(), v.size(), ARGMAX, true);
    assert(argmax == maxind);
    assert(argmin == minind);
    std::fprintf(stderr, "argmax %zu, expected %u\n", argmax, maxind);
    std::fprintf(stderr, "argmin %zu, expected %u\n", argmin, minind);
    float *ptr = (float *)&v[0];
    size_t nf = v.size() * 2;
    for(size_t i = 0; i < nf; ++i) {
        ptr[i] = float(std::rand()) / RAND_MAX;
    }
    unsigned randmaxind = std::rand() % nf;
    unsigned randminind = std::rand() % nf;
    ptr[randmaxind] = 122323.;
    ptr[randminind] = -133000;
    std::fprintf(stderr, "randmaxind %u, randminind %u\n", randmaxind, randminind);
    size_t fargmin = fargsel(ptr, nf, ARGMIN, true);
    size_t fargmax = fargsel(ptr, nf, ARGMAX, true);
    std::fprintf(stderr, "found maxind %zu, found minind %zu\n", fargmax, fargmin);
    assert(randmaxind == fargmax);
    assert(randminind == fargmin);
}
