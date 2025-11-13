# CPU Color Processing Optimization - Complete Analysis

## TL;DR

**Ultra-fused Numba kernel is 52x faster in isolation**, but integration overhead hides the gains.
Current bottleneck: **PyTorch↔NumPy conversions** and other overhead.

**Recommended fix**: Bypass PyTorch entirely for CPU color processing.

---

## Performance Analysis

###Component Benchmarks (100K colors)

| Component | Time | Throughput | Notes |
|-----------|------|------------|-------|
| **NumPy LUT lookup** | 2.389 ms | 42 M/s | Vectorized, but slow |
| **Numba LUT lookup** | 0.038 ms | 2,659 M/s | **70x faster than NumPy!** |
| **Numba Phase 2** | 0.047 ms | 2,150 M/s | Already optimal |
| **Combined (NumPy+Numba)** | 3.096 ms | 32 M/s | Current approach |
| **Ultra-fused (all Numba)** | 0.059 ms | 1,693 M/s | **52x faster!** |

### Key Findings

1. **Numba LUT lookup is 70x faster than NumPy** (0.038 ms vs 2.389 ms)
   - NumPy's "fancy indexing" is actually slow for this use case
   - Numba's parallel loop with direct indexing dominates

2. **Ultra-fused kernel is 52x faster than separate operations**
   - Single pass over data
   - No intermediate allocations
   - Better cache locality

3. **But ColorLUT.apply() doesn't see this speedup**
   - Integration overhead dominates
   - Likely: PyTorch↔NumPy conversions, memory allocations

---

## The Bottleneck: Integration Overhead

### Where Time Is Spent

From earlier analysis:
```
Phase 1 (LUT lookup):     0.899 ms (28%)
Phase 2 (Numba):          0.071 ms (2%)
OVERHEAD:                 2.253 ms (70%)  <-- THE PROBLEM
Total:                    3.223 ms
```

The overhead includes:
- PyTorch tensor ↔ NumPy array conversions
- Memory allocations
- Function call overhead
- Type checking and validation

### Why Ultra-Fused Doesn't Help (Yet)

The ultra-fused kernel eliminates Phase 1 + Phase 2 overhead (0.97 ms total),
but the **integration overhead (2.25 ms) remains**.

Result: 3.2 ms - 0.97 ms + 0.06 ms = **2.29 ms** (minimal improvement)

---

## Solutions (Ranked by Impact)

### 1. **BYPASS PYTORCH ENTIRELY** (Highest Impact: 10-20x)

**Problem**: ColorLUT currently requires PyTorch tensors as input

**Solution**: Add pure NumPy API

```python
class ColorLUT:
    def apply_numpy(
        self,
        colors: np.ndarray,  # Pure NumPy input
        ...
    ) -> np.ndarray:  # Pure NumPy output
        """Fast path - no PyTorch overhead."""

        # Compile LUTs (NumPy)
        self._compile_independent_luts_numpy(...)

        # Single ultra-fused kernel call
        out = np.empty_like(colors)
        fused_color_full_pipeline_numba(
            colors,
            self.r_lut, self.g_lut, self.b_lut,
            saturation, shadows, highlights,
            out
        )

        return out  # No conversions!
```

**Expected performance**: **0.06-0.1 ms** for 100K colors (1,000-1,700 M/s)
**Speedup**: **30-50x faster** than current

**Effort**: 2-4 hours
- Add `apply_numpy()` method
- Duplicate LUT compilation for NumPy
- Add documentation

---

### 2. **SMALLER LUT + LINEAR INTERPOLATION** (Medium Impact: 1.5-2x)

**Problem**: Current LUT (1024 entries, 12 KB) doesn't fit optimally in L1 cache

**Solution**: Use 64-128 entry LUT with linear interpolation

```python
@njit(parallel=True, fastmath=True)
def fused_color_with_linear_interp(colors, r_lut, g_lut, b_lut, ...):
    lut_size = r_lut.shape[0]
    lut_max_f = float(lut_size - 1)

    for i in prange(N):
        r, g, b = colors[i, 0], colors[i, 1], colors[i, 2]

        # Linear interpolation for LUT lookup
        r_pos = r * lut_max_f
        r_idx = int(r_pos)
        r_frac = r_pos - r_idx
        r = r_lut[r_idx] * (1 - r_frac) + r_lut[min(r_idx + 1, lut_size-1)] * r_frac

        # Same for g, b...

        # Phase 2 operations...
```

**Benefits**:
- Smaller LUT (64-128 entries = 0.75-1.5 KB) fits in L1 cache
- Linear interpolation = better quality
- Faster LUT access (L1 vs L2 latency)

**Expected**: 1.5-2x faster

**Effort**: 4-6 hours

---

### 3. **AVX-512 VECTORIZATION** (High Impact: 2-4x)

**Problem**: Current code processes one color at a time

**Solution**: Use AVX-512 to process 8-16 colors simultaneously

```python
# Conceptual - would need Numba intrinsics or C extension
@njit(parallel=True)
def fused_color_simd(colors, ...):
    # Process 8 colors at once with AVX-512
    for i in prange(0, N, 8):
        # Load 8x R, G, B values
        # Apply LUT lookup (8 parallel)
        # Apply Phase 2 (8 parallel)
        # Store 8 results
```

**Expected**: 2-4x faster (combined with other optimizations)

**Effort**: 1-2 weeks (complex, requires low-level optimization)

---

### 4. **DIRECT MEMORY LAYOUT (SOA)** (Medium Impact: 1.3-1.5x)

**Problem**: Colors are stored as `[R,G,B], [R,G,B], ...` (AOS = Array of Structures)

**Better**: Store as `[R,R,R,...], [G,G,G,...], [B,B,B,...]` (SOA = Structure of Arrays)

**Benefits**:
- Better cache usage
- Better SIMD (load 8 R values contiguously)
- Better memory bandwidth utilization

**Expected**: 1.3-1.5x faster

**Effort**: 6-8 hours (requires API changes)

---

## Recommended Implementation Plan

### Phase 1: Quick Win (2-4 hours) - **30-50x speedup**

1. Add `ColorLUT.apply_numpy()` method (pure NumPy API)
2. Use ultra-fused kernel (already implemented)
3. Bypass all PyTorch overhead

**Result**: 0.06-0.1 ms for 100K colors (1,000-1,700 M/s)

### Phase 2: Optimize Further (1 week) - **60-100x total**

4. Smaller LUT + linear interpolation (1.5-2x)
5. AVX-512 vectorization (2-4x additional)

**Result**: 0.02-0.04 ms for 100K colors (2,500-5,000 M/s)

---

## Performance Targets

| Optimization Level | Time (100K) | Throughput | Speedup | Effort |
|-------------------|-------------|------------|---------|--------|
| **Current (ColorLUT)** | 3.2 ms | 31 M/s | 1x | - |
| **Pure NumPy API** | 0.06-0.1 ms | 1,000-1,700 M/s | **30-50x** | 2-4 hrs |
| **+ Smaller LUT** | 0.04-0.05 ms | 2,000-2,500 M/s | **60-80x** | +4-6 hrs |
| **+ AVX-512** | 0.02-0.04 ms | 2,500-5,000 M/s | **80-160x** | +1-2 weeks |

---

## Next Steps

**Immediate action**: Implement `apply_numpy()` method

```python
# Usage
colors_np = np.random.rand(100000, 3).astype(np.float32)
lut = ColorLUT(device="cpu")

# Fast path (no PyTorch overhead)
result = lut.apply_numpy(
    colors_np,
    saturation=1.3,
    shadows=1.1,
    highlights=0.9,
    temperature=0.7,
    brightness=1.2,
    contrast=1.1,
    gamma=0.9,
)
# Result: ~0.06-0.1 ms (vs 3.2 ms currently)
```

**Want me to implement this?** It's a 2-4 hour change for 30-50x speedup!

---

## Appendix: Why NumPy is Slow for LUT Lookup

Counterintuitive finding: NumPy's "fancy indexing" is slower than Numba's manual loop.

**NumPy path**: 2.389 ms
```python
indices = (colors * 1023).astype(np.int64)
result = lut[indices]  # Fancy indexing
```

**Numba path**: 0.038 ms (70x faster!)
```python
for i in prange(N):
    idx = int(colors[i] * 1023)
    result[i] = lut[idx]
```

**Why?**
- NumPy fancy indexing creates intermediate arrays
- Not parallelized effectively
- Cache-unfriendly memory access pattern

**Lesson**: Numba's explicit parallel loop > NumPy vectorization for this pattern
