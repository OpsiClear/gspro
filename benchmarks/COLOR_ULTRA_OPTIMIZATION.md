# ColorLUT Ultra-Optimization Plan

## Executive Summary

Current performance: **30.6 M colors/sec** (3.27 ms for 100K colors) on CPU
Target performance: **300-500 M colors/sec** (10-15x faster)

This document outlines radical optimizations that could achieve 10-100x speedup.

---

## Current Architecture Analysis

### Two-Phase Pipeline

**Phase 1: LUT-capable operations (independent, per-channel)**
- Temperature, Brightness, Contrast, Gamma
- Pre-compiled into 3x 1D LUTs (R, G, B)
- Fast: 46.3 M/s on CPU

**Phase 2: Sequential operations (cross-channel dependent)**
- Saturation, Shadows, Highlights
- Requires luminance calculation (0.299*R + 0.587*G + 0.114*B)
- Slow: 27.3 M/s on CPU (35 M/s with Numba)
- BOTTLENECK: Multiple memory passes, temporary allocations

### Current Optimizations

1. **Separated 1D LUTs** (vs 3D LUT): 10x faster
2. **NumPy path on CPU**: 2-3x faster than PyTorch
3. **Fused Numba Phase 2**: 1.2-6.4x faster (small batches best)

### Remaining Bottlenecks

1. **No GPU optimization** - Phase 2 uses standard PyTorch (not optimized)
2. **Multiple memory passes** - Read LUT, write, read Phase 2, write
3. **Temporary allocations** - Luminance tensors, masks, intermediate results
4. **Kernel launch overhead** - Multiple PyTorch operations = multiple kernels
5. **No vectorization** - Not using GPU's massive parallelism effectively

---

## Optimization Opportunities (Ranked by Impact)

### 1. FUSED TRITON KERNEL (HIGHEST IMPACT: 10-50x)

**Idea**: Single kernel that does EVERYTHING in one pass
- LUT lookup + Phase 2 operations
- No intermediate memory traffic
- Process each pixel completely before moving to next

**Why Triton?**
- Python-based (easier than CUDA)
- Automatic optimization
- Works on both NVIDIA and AMD GPUs

**Implementation**:
```python
@triton.jit
def fused_color_kernel(
    colors_ptr,         # Input colors [N, 3]
    r_lut_ptr,          # R channel LUT [1024]
    g_lut_ptr,          # G channel LUT [1024]
    b_lut_ptr,          # B channel LUT [1024]
    out_ptr,            # Output colors [N, 3]
    N,                  # Number of colors
    lut_size,           # LUT resolution
    saturation,         # Phase 2 params
    shadows,
    highlights,
    BLOCK_SIZE: tl.constexpr,
):
    # Get pixel index
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < N

    # Load input colors (coalesced)
    r = tl.load(colors_ptr + idx * 3 + 0, mask=mask)
    g = tl.load(colors_ptr + idx * 3 + 1, mask=mask)
    b = tl.load(colors_ptr + idx * 3 + 2, mask=mask)

    # PHASE 1: LUT lookup (inline)
    r_idx = (r * (lut_size - 1)).to(tl.int32)
    g_idx = (g * (lut_size - 1)).to(tl.int32)
    b_idx = (b * (lut_size - 1)).to(tl.int32)

    r = tl.load(r_lut_ptr + r_idx, mask=mask)
    g = tl.load(g_lut_ptr + g_idx, mask=mask)
    b = tl.load(b_lut_ptr + b_idx, mask=mask)

    # PHASE 2: Saturation + Shadows/Highlights (inline)
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Saturation
    r = lum + saturation * (r - lum)
    g = lum + saturation * (g - lum)
    b = lum + saturation * (b - lum)
    r = tl.maximum(0.0, tl.minimum(1.0, r))
    g = tl.maximum(0.0, tl.minimum(1.0, g))
    b = tl.maximum(0.0, tl.minimum(1.0, b))

    # Recalculate luminance after saturation
    lum = 0.299 * r + 0.587 * g + 0.114 * b

    # Shadows/Highlights (branchless)
    is_shadow = lum < 0.5
    factor = tl.where(is_shadow, shadows - 1.0, highlights - 1.0)
    r = r + r * factor
    g = g + g * factor
    b = b + b * factor

    # Clamp and store (coalesced)
    r = tl.maximum(0.0, tl.minimum(1.0, r))
    g = tl.maximum(0.0, tl.minimum(1.0, g))
    b = tl.maximum(0.0, tl.minimum(1.0, b))

    tl.store(out_ptr + idx * 3 + 0, r, mask=mask)
    tl.store(out_ptr + idx * 3 + 1, g, mask=mask)
    tl.store(out_ptr + idx * 3 + 2, b, mask=mask)
```

**Expected Performance**:
- Memory bandwidth: Read 1.2 MB + Write 1.2 MB = 2.4 MB for 100K colors
- GPU bandwidth: 400 GB/s (typical)
- Theoretical limit: 2.4 MB / 400 GB/s = **0.006 ms**
- Realistic with compute: **0.05-0.1 ms** (accounting for LUT lookup, arithmetic)
- **Speedup: 30-60x faster than current (3.27 ms -> 0.05-0.1 ms)**

**Benefits**:
- Single kernel launch (no overhead)
- Coalesced memory access (full GPU bandwidth)
- No temporary allocations
- Process all 100K colors in parallel
- Works on NVIDIA and AMD GPUs

**Effort**: 4-8 hours
- Install triton
- Port kernel logic
- Test correctness
- Benchmark

---

### 2. CUDA KERNEL WITH SHARED MEMORY (ULTRA FAST: 50-100x)

**Idea**: Custom CUDA kernel with shared memory for LUTs
- Load LUTs into shared memory (fast access)
- Fused operations (same as Triton)
- Optimized for specific hardware

**Why better than Triton?**
- More control over memory hierarchy
- Can use texture memory for LUTs
- Potentially 2-3x faster than Triton

**Implementation sketch**:
```cpp
__global__ void fused_color_kernel(
    const float* colors,
    const float* r_lut,
    const float* g_lut,
    const float* b_lut,
    float* out,
    int N,
    int lut_size,
    float saturation,
    float shadows,
    float highlights
) {
    // Load LUTs into shared memory
    __shared__ float s_r_lut[1024];
    __shared__ float s_g_lut[1024];
    __shared__ float s_b_lut[1024];

    // Cooperative loading (all threads help)
    for (int i = threadIdx.x; i < lut_size; i += blockDim.x) {
        s_r_lut[i] = r_lut[i];
        s_g_lut[i] = g_lut[i];
        s_b_lut[i] = b_lut[i];
    }
    __syncthreads();

    // Process colors
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    // Load input
    float r = colors[idx * 3 + 0];
    float g = colors[idx * 3 + 1];
    float b = colors[idx * 3 + 2];

    // Phase 1: LUT lookup from shared memory (FAST!)
    int r_idx = min(max((int)(r * (lut_size - 1)), 0), lut_size - 1);
    int g_idx = min(max((int)(g * (lut_size - 1)), 0), lut_size - 1);
    int b_idx = min(max((int)(b * (lut_size - 1)), 0), lut_size - 1);

    r = s_r_lut[r_idx];
    g = s_g_lut[g_idx];
    b = s_b_lut[b_idx];

    // Phase 2: Saturation + Shadows/Highlights (inline, same as Triton)
    float lum = 0.299f * r + 0.587f * g + 0.114f * b;

    // Saturation
    if (saturation != 1.0f) {
        r = lum + saturation * (r - lum);
        g = lum + saturation * (g - lum);
        b = lum + saturation * (b - lum);
        r = fminf(fmaxf(r, 0.0f), 1.0f);
        g = fminf(fmaxf(g, 0.0f), 1.0f);
        b = fminf(fmaxf(b, 0.0f), 1.0f);
    }

    // Recalculate luminance
    lum = 0.299f * r + 0.587f * g + 0.114f * b;

    // Shadows/Highlights
    float factor = (lum < 0.5f) ? (shadows - 1.0f) : (highlights - 1.0f);
    r = fminf(fmaxf(r + r * factor, 0.0f), 1.0f);
    g = fminf(fmaxf(g + g * factor, 0.0f), 1.0f);
    b = fminf(fmaxf(b + b * factor, 0.0f), 1.0f);

    // Store
    out[idx * 3 + 0] = r;
    out[idx * 3 + 1] = g;
    out[idx * 3 + 2] = b;
}
```

**Expected Performance**: **0.03-0.05 ms** (50-100x faster)
- Shared memory latency: ~1 cycle (vs 200-400 cycles for global memory)
- LUT lookups are essentially free
- Memory bandwidth still limits (2.4 MB / 400 GB/s = 0.006 ms)

**Effort**: 1-2 days
- Write CUDA kernel
- Create PyTorch extension
- Test on different GPUs
- Optimize block size

---

### 3. TEXTURE MEMORY FOR LUT (2-3x FASTER LUT LOOKUP)

**Idea**: Use CUDA texture memory for LUT storage
- Hardware-accelerated interpolation (linear interp for free!)
- Cached in texture cache (separate from L1/L2)
- Automatic clamping

**Implementation**:
```cpp
texture<float, 1, cudaReadModeElementType> r_lut_tex;
texture<float, 1, cudaReadModeElementType> g_lut_tex;
texture<float, 1, cudaReadModeElementType> b_lut_tex;

__global__ void fused_color_kernel_texture(...) {
    // ...

    // LUT lookup with hardware interpolation (FREE!)
    r = tex1D(r_lut_tex, r * (lut_size - 1));
    g = tex1D(g_lut_tex, g * (lut_size - 1));
    b = tex1D(b_lut_tex, b * (lut_size - 1));

    // ... Phase 2 ...
}
```

**Benefits**:
- Linear interpolation in hardware (better quality, same speed)
- Texture cache separate from L1/L2 (more cache available)
- Automatic clamping

**Effort**: 2-4 hours (add to CUDA kernel above)

---

### 4. FLOAT16 PRECISION MODE (2x MEMORY BANDWIDTH)

**Idea**: Use fp16 for all operations
- 2x memory bandwidth (read/write half the bytes)
- Modern GPUs have dedicated fp16 units (2-16x faster compute)
- Minimal quality loss for color grading

**Implementation**:
```python
class ColorLUT:
    def __init__(self, device="cuda", lut_size=1024, use_fp16=False):
        self.use_fp16 = use_fp16
        self.dtype = torch.float16 if use_fp16 else torch.float32
        # ...

    def apply(self, colors, ...):
        if self.use_fp16 and colors.dtype != torch.float16:
            colors = colors.half()

        # ... processing ...

        if self.use_fp16:
            result = result.float()  # Convert back if needed

        return result
```

**Expected Performance**: **1.5-2x faster**
- Memory bandwidth: 2x improvement
- Compute: 2-16x faster on modern GPUs (Tensor Cores)
- Combined: 1.5-2x realistic speedup

**Quality Impact**: Minimal
- fp16 has 10-bit mantissa (vs 23-bit for fp32)
- For colors in [0, 1], precision is ~0.001 (0.1%)
- Imperceptible for most use cases

**Effort**: 2-4 hours

---

### 5. BATCHED PROCESSING WITH CUDA STREAMS (1.5-2x FOR LARGE BATCHES)

**Idea**: Split large batches and process in parallel streams
- Overlap compute and memory transfers
- Better GPU utilization
- Hide latency

**Implementation**:
```python
def apply_batched(self, colors, batch_size=50000, num_streams=4):
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    results = []

    for i in range(0, len(colors), batch_size):
        stream_idx = (i // batch_size) % num_streams

        with torch.cuda.stream(streams[stream_idx]):
            batch = colors[i:i+batch_size]
            result = self._fused_kernel(batch, ...)
            results.append(result)

    # Synchronize all streams
    for stream in streams:
        stream.synchronize()

    return torch.cat(results)
```

**Expected Performance**: **1.5-2x for very large batches (>1M)**
- Hides memory latency
- Better GPU occupancy

**Effort**: 4-6 hours

---

### 6. CPU VECTORIZATION WITH AVX-512 (2-3x ON CPU)

**Idea**: Use explicit SIMD instructions for CPU path
- Process 8 colors at once with AVX-512
- Or 4 colors with AVX2
- Combined with Numba parallelization

**Implementation**: Use Numba with `@vectorize` decorator
```python
from numba import vectorize, float32

@vectorize([float32(float32, float32, float32, float32, float32)])
def process_pixel(r, g, b, saturation, ...):
    # Process single pixel (Numba will vectorize)
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    # ...
    return r_out, g_out, b_out
```

**Expected Performance**: **2-3x on CPU** (combined with parallel)

**Effort**: 6-8 hours

---

### 7. ZERO-COPY PROCESSING (ELIMINATE TRANSFERS)

**Idea**: Never transfer between CPU/GPU
- Auto-detect input device
- Process on same device
- Return on same device

**Implementation**:
```python
def apply(self, colors, ...):
    input_device = colors.device

    # If input is GPU but ColorLUT is CPU, warn and transfer
    if input_device.type == "cuda" and self.device == "cpu":
        logger.warning("GPU->CPU transfer! Consider using ColorLUT(device='cuda')")
        # Or: auto-create GPU version
        colors = colors.cpu()

    # ... processing ...

    # Return on same device as input
    if result.device != input_device:
        result = result.to(input_device)

    return result
```

**Expected Performance**: **Saves 1-10ms for large batches** (transfer overhead)

**Effort**: 1-2 hours

---

## Implementation Priority

### Phase 1: Quick Wins (1-2 days)
1. **Float16 mode** (2-4 hours) - 1.5-2x speedup, minimal effort
2. **Zero-copy processing** (1-2 hours) - Eliminate transfer overhead
3. **CPU AVX-512 vectorization** (6-8 hours) - 2-3x on CPU

### Phase 2: Fused GPU Kernel (1 week)
4. **Triton fused kernel** (1-2 days) - 10-30x speedup on GPU
5. **Test on various GPUs** (1 day)
6. **Correctness verification** (1 day)
7. **Benchmark suite** (1 day)

### Phase 3: Ultra Optimization (1 week)
8. **CUDA kernel with shared memory** (2-3 days) - 50-100x speedup
9. **Texture memory for LUTs** (1 day) - Better quality + speed
10. **CUDA streams for large batches** (1 day)

---

## Expected Final Performance

| Batch Size | Current | After Triton | After CUDA | Speedup |
|-----------|---------|--------------|------------|---------|
| **1K** | 0.10 ms | 0.02 ms | 0.01 ms | **10x** |
| **10K** | 0.50 ms | 0.05 ms | 0.02 ms | **25x** |
| **100K** | 3.27 ms | 0.10 ms | 0.05 ms | **65x** |
| **1M** | 31 ms | 0.50 ms | 0.20 ms | **150x** |

**Target throughput**:
- **Current**: 30 M/s (CPU)
- **After Triton**: 1,000 M/s (1 billion/sec)
- **After CUDA**: 2,000-5,000 M/s (2-5 billion/sec)

---

## Comparison to State-of-the-Art

**Industry standard**: Adobe Lightroom, DaVinci Resolve
- Use GPU-accelerated color grading
- Process 4K video at 60 FPS = 8.3M pixels/frame * 60 FPS = **500M pixels/sec**
- Our target: **2-5 billion/sec** (4-10x faster!)

---

## Conclusion

Current ColorLUT is fast for CPU, but GPU optimization is untapped.

**Recommended approach**:
1. Start with Triton kernel (easiest, 10-30x)
2. If needed, move to CUDA (50-100x)
3. Add float16 mode for even more speed

**Total effort**: 1-2 weeks for complete optimization
**Expected result**: **50-150x speedup** (3.27 ms -> 0.02-0.06 ms for 100K colors)
