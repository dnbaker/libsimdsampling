## libsimdsampling

libsimdsampling uses vectorized random number generation with reservoir sampling to perform data-parallel weighted point selection.

This can make a significant improvement in speed of applications sampling from streaming weights, such as kmeans++ or D2 sampling.

Compile with `make`, and link against either `-lsimdsampling` or `-lsimdsampling-st`. libsimdsampling parallelizes using OpenMP, and libsimdsampling\_st performs serial sampling.
