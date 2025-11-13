# Color Processing Ultra-Optimization: Recommendations

## TL;DR

**Current performance**: 30.6 M colors/sec (3.27 ms for 100K colors on CPU)
**Target performance**: 1,000-5,000 M colors/sec (10-150x faster)

**Recommended path**: Implement Triton fused kernel -> 10-50x speedup in 1-2 days

---

## Executive Summary

Your color processing is **slow** because:

1. **No GPU optimization** - Current GPU path uses standard PyTorch (not optimized)
2. **Multiple memory passes** - Read LUT, write, read Phase 2, write (2x memory traffic)
3. **Kernel launch overhead** - Each PyTorch operation launches a separate GPU kernel
4. **Temporary allocations** - Luminance tensors, masks, intermediate results

The solution: **Fused GPU kernel** that does everything in one pass.

---

## Current Architecture (Bottleneck Analysis)

### Phase 1: LUT Operations (FAST)
- Temperature, Brightness, Contrast, Gamma
- Pre-compiled into 3x 1D LUTs
- Performance: **46.3 M/s** on CPU

### Phase 2: Sequential Operations (SLOW)
- Saturation, Shadows, Highlights
- Requires luminance calculation
- Performance: **27.3 M/s** on CPU (35 M/s with Numba)
- **BOTTLENECK**: This is the slow part!

### Combined Pipeline
- Total: **30.6 M/s** (3.27 ms for 100K colors)
- Memory traffic: 2.4 MB (read + write)
- Theoretical limit (DDR4): 0.12 ms
- **We're 27x slower than memory bandwidth limit!**

Why? Because we're doing multiple passes over the data instead of one fused operation.

---

## Optimization Strategies (Ranked by Impact)

### 1. TRITON FUSED KERNEL (Recommended - 10-50x speedup)

**What**: Single GPU kernel that does LUT lookup + Phase 2 in one pass

**Why this works**:
- Single kernel launch (no overhead)
- Read data once, write once (eliminate intermediate memory traffic)
- Coalesced memory access (full GPU bandwidth)
- Process all pixels in parallel (utilize thousands of GPU cores)

**Expected performance**:
- Small batches (1K-10K): **10-20x faster**
- Large batches (100K-1M): **30-50x faster**
- Throughput: **1,000-2,000 M/s** (1-2 billion colors/sec)

**Implementation effort**: 1-2 days
- Prototype already created in `src/gslut/color_triton.py`
- Needs testing and integration
- Requires `pip install triton`

**Code**: See `src/gslut/color_triton.py` for complete implementation

**Benchmark**: Run `uv run benchmarks/benchmark_triton_vs_standard.py` (requires CUDA + Triton)

---

### 2. CUDA KERNEL WITH SHARED MEMORY (Ultimate - 50-150x speedup)

**What**: Hand-optimized CUDA kernel with LUTs in shared memory

**Why this works**:
- Shared memory is 100x faster than global memory (1 cycle vs 200 cycles)
- LUT lookups are essentially free
- Full control over memory hierarchy
- Can use texture memory for hardware interpolation

**Expected performance**:
- **50-150x faster than current**
- Throughput: **3,000-5,000 M/s** (3-5 billion colors/sec)
- Latency: **0.02-0.05 ms** for 100K colors

**Implementation effort**: 2-3 days
- Write CUDA kernel
- Create PyTorch extension (use `torch.utils.cpp_extension`)
- Test on different GPUs
- Optimize block size

**When to use**: If Triton isn't fast enough (unlikely)

---

### 3. FLOAT16 PRECISION MODE (Easy - 1.5-2x speedup)

**What**: Use fp16 instead of fp32 for all operations

**Why this works**:
- 2x memory bandwidth (half the bytes)
- Modern GPUs have dedicated fp16 units (2-16x faster compute)
- Minimal quality loss (0.1% precision for colors in [0, 1])

**Expected performance**:
- **1.5-2x faster** (combined with Triton: 20-100x total)
- Memory bandwidth: 2x improvement
- Compute: 2-16x faster on modern GPUs (Tensor Cores)

**Implementation effort**: 2-4 hours
- Add `use_fp16` parameter to ColorLUT
- Convert tensors and LUTs to fp16
- Optionally convert back to fp32 for output

**Recommended**: Yes, especially for large batches

---

### 4. ZERO-COPY DEVICE HANDLING (Easy - Eliminate transfer overhead)

**What**: Auto-detect input device and process on same device

**Current problem**: If you pass GPU tensors to CPU ColorLUT, it silently transfers to CPU!

**Solution**:
```python
def apply(self, colors, ...):
    input_device = colors.device

    # Warn if device mismatch
    if input_device.type == "cuda" and self.device == "cpu":
        logger.warning("GPU->CPU transfer detected! Use ColorLUT(device='cuda')")

    # Or: auto-create GPU version
    if input_device.type == "cuda" and self.device == "cpu":
        # Transfer to GPU version if available
        pass

    # ... processing ...

    # Always return on same device as input
    return result.to(input_device)
```

**Expected performance**: Saves 1-10 ms for large batches (transfer overhead)

**Implementation effort**: 1-2 hours

---

### 5. CPU AVX-512 VECTORIZATION (Medium - 2-3x on CPU)

**What**: Explicit SIMD instructions for CPU path

**Why this works**:
- Process 8 colors at once with AVX-512 (or 4 with AVX2)
- Combined with Numba parallelization
- Better instruction-level parallelism

**Expected performance**: **2-3x faster on CPU** (60-90 M/s total)

**Implementation effort**: 6-8 hours
- Use Numba `@vectorize` decorator
- Or write explicit SIMD code

**When to use**: If GPU is not available or for small batches

---

### 6. CUDA STREAMS FOR LARGE BATCHES (Advanced - 1.5-2x for >1M)

**What**: Split large batches and process in parallel streams

**Why this works**:
- Overlap compute and memory transfers
- Better GPU utilization
- Hide latency

**Expected performance**: **1.5-2x for very large batches** (>1M colors)

**Implementation effort**: 4-6 hours

**When to use**: For very large batches (>1M colors) after implementing Triton/CUDA kernel

---

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
**Goal**: 10-30x speedup with minimal effort

1. **Implement Triton fused kernel** (1 day)
   - Already prototyped in `src/gslut/color_triton.py`
   - Add to ColorLUT as automatic optimization
   - Test correctness
   - Benchmark

2. **Add float16 mode** (2-4 hours)
   - Add `use_fp16` parameter
   - Convert tensors to fp16
   - Benchmark

3. **Fix device handling** (1-2 hours)
   - Add warnings for device mismatches
   - Auto-detect input device
   - Return on same device

**Expected result**: **20-60x speedup** (3.27 ms -> 0.05-0.15 ms for 100K colors)

---

### Phase 2: Ultimate Optimization (1 week)
**Goal**: 50-150x speedup for maximum performance

4. **CUDA kernel with shared memory** (2-3 days)
   - Write optimized CUDA kernel
   - Use shared memory for LUTs
   - Test on various GPUs

5. **Texture memory for LUTs** (1 day)
   - Use CUDA texture memory
   - Hardware-accelerated interpolation
   - Better quality + speed

6. **CUDA streams for large batches** (1 day)
   - Batched processing
   - Overlap compute and memory

**Expected result**: **100-150x speedup** (3.27 ms -> 0.02-0.03 ms for 100K colors)

---

## Performance Comparison

| Optimization | Effort | Speedup | Throughput | When to use |
|-------------|--------|---------|------------|-------------|
| **Current (CPU + Numba)** | - | 1x | 30 M/s | Baseline |
| **Triton fused kernel** | 1-2 days | 10-50x | 1,000 M/s | Always (if GPU available) |
| **+ Float16** | 2-4 hours | 20-100x | 2,000 M/s | Large batches |
| **CUDA + shared mem** | 2-3 days | 50-150x | 3,000-5,000 M/s | Ultimate performance |

---

## Recommended Implementation Order

1. **Start with Triton** (1-2 days)
   - Biggest bang for buck
   - 10-50x speedup
   - Easy to implement (prototype already done)
   - Works on NVIDIA and AMD GPUs

2. **Add float16 if needed** (2-4 hours)
   - Extra 1.5-2x on top of Triton
   - Minimal effort
   - Negligible quality loss

3. **Only do CUDA if Triton isn't enough** (2-3 days)
   - 2-3x faster than Triton
   - Much more effort
   - NVIDIA-only
   - Harder to maintain

**My recommendation**: **Implement Triton first**. It will likely be fast enough (1-2 billion colors/sec), and you can always optimize further if needed.

---

## Validation Plan

After implementing Triton kernel:

1. **Correctness**: Verify max difference < 1e-5 vs standard path
2. **Performance**: Benchmark across batch sizes (1K, 10K, 100K, 1M)
3. **Device compatibility**: Test on different GPU models
4. **Memory usage**: Verify no memory leaks
5. **Integration**: Add to ColorLUT with automatic activation

---

## Code Artifacts

All prototype code is ready:

1. **Triton kernel**: `src/gslut/color_triton.py` (complete implementation)
2. **Benchmark**: `benchmarks/benchmark_triton_vs_standard.py` (requires CUDA + Triton)
3. **Analysis**: `benchmarks/COLOR_ULTRA_OPTIMIZATION.md` (detailed technical analysis)
4. **Bottleneck profiling**: `benchmarks/analyze_color_bottlenecks.py`

---

## Next Steps

**To implement Triton optimization**:

1. Install Triton:
   ```bash
   pip install triton
   ```

2. Test the prototype:
   ```bash
   uv run benchmarks/benchmark_triton_vs_standard.py
   ```

3. If it works well, integrate into `ColorLUT`:
   ```python
   # In color.py
   from gslut.color_triton import apply_fused_color_triton, is_triton_available

   def apply(self, colors, ...):
       # ... existing code ...

       # Use Triton if available (GPU only)
       if is_triton_available() and colors.is_cuda:
           return apply_fused_color_triton(...)

       # ... existing fallback code ...
   ```

4. Add tests and documentation

5. Ship it!

---

## Comparison to Industry Standards

**Adobe Lightroom / DaVinci Resolve**:
- Process 4K video at 60 FPS
- 8.3M pixels/frame * 60 FPS = **500M pixels/sec**

**Our target (with Triton)**:
- **1,000-2,000 M pixels/sec**
- **2-4x faster than industry standard!**

**Our target (with CUDA)**:
- **3,000-5,000 M pixels/sec**
- **6-10x faster than industry standard!**

---

## Conclusion

Your color processing **can be 10-150x faster** with GPU optimization.

**The key insight**: Stop using standard PyTorch operations (each launches a separate kernel). Instead, use a **single fused kernel** that does everything in one pass.

**Recommended action**:
1. Implement Triton fused kernel (10-50x, 1-2 days)
2. Add float16 mode (1.5-2x, 2-4 hours)
3. Ship it!

**Total effort**: 1-2 days for 20-100x speedup

Let me know when you want to implement this, and I can help integrate it!
