## libsimdsampling

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
uint64_t selected_point = simd_sampling(data.data(), data.size()); // Manual selection of pointer
uint64_t other_point = simd_sampling(data);                        // Using the container overload to access these functions automatically
```

### Caveats
libsimdsampling is currently implemented in C++17. Soon, we may rewrite the aligned/unaligned store selection to avoid using `if constexpr`.

If the number of points is small (<10000), it may be slower than the naive approach.


### Future work

For kmeans++ with large k, it may be valuable to sample more than one point per iteration. We may add a generalized sampler
using heaps soon.

### Dependencies

Requires [libsleef](https://github.com/shibatch/sleef).

Add `INCLUDE_PATHS=` or `LINK_PATHS=` arguments to `sleef/build/{include/lib}` to ensure the compiler can find the library.
