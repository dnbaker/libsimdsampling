## libsimdsampling  [![Build Status](https://travis-ci.com/dnbaker/libsimdsampling.svg?branch=main)](https://travis-ci.com/dnbaker/libsimdsampling)

libsimdsampling uses vectorized random number generation with reservoir sampling to perform data-parallel weighted point selection.

This can make a significant improvement in speed of applications sampling from streaming weights, such as kmeans++ or D2 sampling.

Compile with `make`, and link against either `-lsimdsampling` or `-lsimdsampling-st`. libsimdsampling parallelizes using OpenMP, and libsimdsampling\_st performs serial sampling.

## Usage

Example usage:
```
#ifdef _OPENMP
    // If OpenMP is defined, set the number of threads to determine parallelism.
#endif
blaze::DynamicVector<double> data(1000000);
// Initialize data with nonnegative weights
uint64_t selected_point = reservoir_simd::sample(data.data(), data.size()); // Manual selection of pointer
uint64_t other_point = reservoir_simd::sample(data);                        // Using the container overload to access these functions automatically

// To sample k points randomly without replacement:
const std::vector<uint64_t> selected = reservoir_simd::sample_k(data.data(), data.size(), k, /*seed=*/13);
```

### Usage in C

Simply include the same header and link as using C++; however, you will need to call unqualified names;

Fmt determines whether to sample with replacement (not possible for k = 1)
and whether to use dense SIMD computation or use exponential skips instead.

```
size_t n = 10000;
int k = 25;
uint64_t seed = 0;
double *data = malloc(sizeof(double) * n);
float *fdata = malloc(sizeof(float) * n);
enum SampleFmt fmt = 0;
// Initialization here...
uint64_t selected_point = dsimd_sample(data, n, seed, fmt);
uint64_t float_selected_point = fsimd_sample(data, n, seed, fmt);
```

It's more complicated to sample k points without replacement; one must pre-allocate a buffer of 64-bit integers,
and you will be responsible for freeing that memory.

```
uint64_t *retv = malloc(sizeof(uint64_t) * k);
dsimd_sample_k(data, n, k, retv, seed);
fsimd_sample_k(fdata, n, k, retv, seed); // Overwrites result from dsimd_sample_k
free(retv);
```

## Sampling k points

One can sample k points at a time via `simd_sample_k`.
The same algorithm is used, except that a heap of the lowest-priority (highest value) elements are kept, rather than
the lowest-priority element.

## argmin/argmax

We also provide vectorized (and optionally parallelized) argmin/argmax for *float*s and *double*s. We may in the future add 
versions for other fundamental types. This is also within the namespace `reservoir_simd`.

The C api is:
1. dargsel(double \*, size\_t, enum ArgReduction), where ArgReduction is either `ARGMIN` or `ARGMAX`.
2. fargsel(float \*, size\_t, enum ArgReduction), where ArgReduction is either `ARGMIN` or `ARGMAX`.
3. dargmin(double \*, size\_t)
4. fargmin(float \*, size\_t)
5. dargmax(double \*, size\_t)
6. fargmax(float \*, size\_t)

See argminmax.h for convenience functions in C++.

### Dependencies

Requires [libsleef](https://github.com/shibatch/sleef), which can be installed easily with homebrew or apt-get on Ubuntu, or can be built from source.

Add `INCLUDE_PATHS=` or `LINK_PATHS=` arguments to `sleef/build/{include/lib}` to ensure the compiler can find the library if necessary.
