## libsimdsampling

libsimdsampling uses vectorized random number generation with reservoir sampling to perform data-parallel weighted point selection.

This can make a significant improvement in speed of applications sampling from streaming weights, such as kmeans++ or D2 sampling.

Compile with `make`, and link against either `-lsimdsampling` or `-lsimdsampling-st`. libsimdsampling parallelizes using OpenMP, and libsimdsampling\_st performs serial sampling.


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
