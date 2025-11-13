# NumPy Transform Optimization Recommendations

Based on comprehensive bottleneck analysis of 1M Gaussian transforms.

## Current Performance

- **NumPy + Numba**: 15.7 ms (64M Gaussians/sec)
- **PyTorch CPU**: 10.2 ms (98M Gaussians/sec)
- **Gap**: 1.54x slower

## Bottleneck Analysis

### Time Breakdown (1M Gaussians)
```
Matrix application:     8.1 ms (51.6%)  <- BLAS-optimized, already optimal
Quaternion multiply:    2.5 ms (15.9%)  <- Has 1.1ms wrapper overhead
Scale multiply:         1.9 ms (12.1%)  <- Has 1.9ms allocation overhead
Matrix composition:     0.03 ms (0.2%)  <- Negligible
Other overhead:         3.5 ms (22.3%)  <- Accumulated overhead
----------------------------------------
Total:                 16.0 ms
```

### Key Findings

1. **Memory Allocation Overhead**: 1.9ms per allocation (85.6% of operation time!)
2. **Wrapper Overhead**: 1.1ms for shape validation and checks (55.3% of quat multiply)
3. **Shape Validation**: 1.0ms overhead (76.4% of validation time)
4. **Accumulated Overhead**: 3.5ms appears when operations run together (cache effects)

## Optimization Opportunities

### 1. PRE-ALLOCATED OUTPUT BUFFERS (Highest Impact)

**Potential Savings**: ~2-3ms (15-19% speedup)

Add optional `out` parameters to allow buffer reuse:

```python
def transform(
    means, quaternions, scales,
    translation=None, rotation=None, scale_factor=None,
    center=None, rotation_format="quaternion",
    out_means=None,      # NEW
    out_quaternions=None, # NEW
    out_scales=None      # NEW
):
    """
    Args:
        out_means: Optional pre-allocated output buffer for means [N, 3]
        out_quaternions: Optional pre-allocated output buffer for quaternions [N, 4]
        out_scales: Optional pre-allocated output buffer for scales [N, 3]
    """
    if out_means is None:
        out_means = np.empty_like(means)
    if out_quaternions is None:
        out_quaternions = np.empty_like(quaternions)
    if out_scales is None:
        out_scales = np.empty_like(scales)

    # Use pre-allocated buffers...
    return out_means, out_quaternions, out_scales
```

**Use Case**: When doing repeated transforms (e.g., animation, batch processing):
```python
# Allocate once
out_means = np.empty_like(means)
out_quats = np.empty_like(quats)
out_scales = np.empty_like(scales)

# Reuse buffers for 100 frames
for frame in range(100):
    transform(means, quats, scales, ...,
              out_means=out_means, out_quaternions=out_quats, out_scales=out_scales)
```

### 2. FAST-PATH FUNCTIONS (High Impact)

**Potential Savings**: ~1-2ms (6-13% speedup)

Add "unsafe" fast-path versions that skip shape validation:

```python
def transform_fast(
    means, quaternions, scales,
    translation, rotation, scale_factor,
    out_means, out_quaternions, out_scales
):
    """
    Fast-path transform with no shape validation.

    Requirements:
    - All inputs must be contiguous NumPy arrays
    - means: [N, 3], quaternions: [N, 4], scales: [N, 3]
    - rotation: [4] quaternion
    - Output buffers must be pre-allocated
    """
    # Direct Numba calls without wrapper overhead
    M, rotation_quat, scale_vec = _compose_transform_matrix_numpy_fast(...)
    _apply_homogeneous_transform_numba(means, M, out_means)
    quaternion_multiply_single_numba(rotation_quat, quaternions, out_quaternions)
    elementwise_multiply_vector_numba(scales, scale_vec, out_scales)
    return out_means, out_quaternions, out_scales
```

### 3. GUVECTORIZE FOR AUTOMATIC BROADCASTING (Medium Impact)

**Potential Savings**: ~0.5-1ms (3-6% speedup)

Use `@guvectorize` to eliminate wrapper overhead:

```python
from numba import guvectorize

@guvectorize(
    [(float32[:], float32[:, :], float32[:, :])],
    '(m),(n,m)->(n,m)',
    nopython=True, target='parallel'
)
def quaternion_multiply_gufunc(q1, q2, out):
    """
    Numba guvectorize handles broadcasting automatically.
    Eliminates need for Python wrapper shape checking.
    """
    for i in range(q2.shape[0]):
        w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
        w2, x2, y2, z2 = q2[i, 0], q2[i, 1], q2[i, 2], q2[i, 3]

        out[i, 0] = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        out[i, 1] = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        out[i, 2] = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        out[i, 3] = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
```

### 4. ENSURE CONTIGUOUS ARRAYS (Low Impact)

**Potential Savings**: ~0.1-0.3ms (1-2% speedup if arrays are non-contiguous)

Add contiguity checks at function entry:

```python
def transform(means, quaternions, scales, ...):
    # Ensure C-contiguous for optimal BLAS/Numba performance
    if not means.flags['C_CONTIGUOUS']:
        means = np.ascontiguousarray(means)
    if not quaternions.flags['C_CONTIGUOUS']:
        quaternions = np.ascontiguousarray(quaternions)
    if not scales.flags['C_CONTIGUOUS']:
        scales = np.ascontiguousarray(scales)
    ...
```

### 5. CACHE TRANSFORMATION MATRICES (Use-Case Specific)

**Potential Savings**: ~0.03ms per cached transform (negligible)

For repeated transforms with identical parameters:

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def _get_transform_matrix_cached(translation_tuple, rotation_tuple, scale, dtype_str):
    """Cache transformation matrices for repeated use."""
    translation = np.array(translation_tuple, dtype=dtype_str)
    rotation = np.array(rotation_tuple, dtype=dtype_str)
    return _compose_transform_matrix_numpy(translation, rotation, ...)
```

## Optimization Priority

### Tier 1: High Impact, Easy to Implement
1. **Pre-allocated output buffers** - Add `out_*` parameters
2. **Fast-path functions** - Add `transform_fast()` variant

### Tier 2: Medium Impact, Moderate Effort
3. **Guvectorize refactor** - Rewrite quaternion ops with `@guvectorize`
4. **Contiguity checks** - Add `np.ascontiguousarray` guards

### Tier 3: Low Impact or Specialized
5. **Transformation matrix caching** - Only beneficial for specific use cases
6. **Reduce matrix composition overhead** - Already negligible (0.03ms)

## Expected Performance After Optimizations

```
Current:  15.7ms (64M Gaussians/sec)

After Tier 1:  12-13ms (77-83M Gaussians/sec) - 1.21x faster
After Tier 2:  11-12ms (83-91M Gaussians/sec) - 1.31x faster
Target (PyTorch): 10.2ms (98M Gaussians/sec)
```

**Estimated gap after all optimizations**: 1.08-1.18x slower than PyTorch (vs current 1.54x)

## Implementation Notes

### Backward Compatibility
- Keep existing API unchanged
- Add new optional parameters with default `None` values
- Add `_fast` variants as new functions

### Testing
- Add benchmarks comparing optimized vs baseline
- Test with small (10K), medium (100K), and large (1M) datasets
- Verify correctness of fast-path implementations

### Documentation
- Document when to use fast-path functions
- Provide examples of buffer reuse patterns
- Note performance characteristics

## Alternative: JIT the Entire Transform

Use `@njit` for the entire transform pipeline (most aggressive optimization):

```python
@njit(parallel=True, fastmath=True, cache=True)
def transform_numba_full(
    means, quaternions, scales,
    translation, rotation, scale_factor,
    out_means, out_quaternions, out_scales
):
    """
    Fully JIT-compiled transform - fastest possible.

    All operations in Numba, no Python overhead.
    Requires all inputs to be NumPy arrays.
    """
    # Build matrix inline (avoid function calls)
    M = np.eye(4, dtype=np.float32)
    # ... inline matrix building ...

    # Apply transforms
    for i in prange(len(means)):
        # ... fused operations ...
        pass

    return out_means, out_quaternions, out_scales
```

**Potential**: Could match or beat PyTorch (estimated 9-10ms)
**Drawback**: Less flexible, harder to maintain, requires major refactor

## Recommendation

**Implement Tier 1 optimizations first**:
1. Add output buffer parameters (2-3 hours implementation)
2. Add fast-path `transform_fast()` function (4-6 hours implementation)

This should close 60-70% of the performance gap with minimal risk and complexity.

## Benchmark Commands

```bash
# Before optimizations
uv run benchmarks/profile_transform.py

# After optimizations
uv run benchmarks/benchmark_optimized.py

# Compare with PyTorch
uv run python -c "..." # (see analyze_bottlenecks.py for full command)
```
