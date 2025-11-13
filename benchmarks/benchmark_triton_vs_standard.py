"""
Benchmark Triton fused kernel vs standard PyTorch path.

This benchmark compares:
1. Standard PyTorch path (separate operations)
2. Triton fused kernel (single GPU kernel)

Expected: Triton should be 10-50x faster for large batches.

Requirements:
    pip install triton
"""

import time

import numpy as np
import torch

# Check if Triton is available
try:
    from gspro.color_triton import apply_fused_color_triton, is_triton_available

    TRITON_AVAILABLE = is_triton_available()
except ImportError:
    TRITON_AVAILABLE = False

from gspro import ColorLUT

print("=" * 80)
print("TRITON FUSED KERNEL BENCHMARK")
print("=" * 80)

if not torch.cuda.is_available():
    print("\n[SKIP] CUDA not available on this system")
    print("This benchmark requires a CUDA-capable GPU")
    exit(0)

if not TRITON_AVAILABLE:
    print("\n[SKIP] Triton not available")
    print("Install with: pip install triton")
    print("(Requires CUDA-capable GPU)")
    exit(0)

print(f"\n[OK] CUDA available: {torch.cuda.get_device_name(0)}")
print(f"[OK] Triton available: {TRITON_AVAILABLE}")

# Test different batch sizes
batch_sizes = [1_000, 10_000, 100_000, 1_000_000]

print("\n" + "=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

for N in batch_sizes:
    print(f"\n[Testing {N:,} colors]")
    print("-" * 80)

    # Create test data on GPU
    colors = torch.rand(N, 3, device="cuda")

    # Create ColorLUT and compile LUTs
    lut = ColorLUT(device="cuda", lut_size=1024)

    # Pre-compile LUTs (warmup)
    _ = lut.apply(
        colors[:100],
        temperature=0.7,
        brightness=1.2,
        contrast=1.1,
        gamma=0.9,
        saturation=1.3,
        shadows=1.1,
        highlights=0.9,
    )

    # Standard PyTorch path
    print("\n[1] Standard PyTorch Path:")
    torch.cuda.synchronize()
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result_standard = lut.apply(
            colors,
            temperature=0.7,
            brightness=1.2,
            contrast=1.1,
            gamma=0.9,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    standard_time = np.mean(times)
    standard_std = np.std(times)
    print(f"  Time:       {standard_time:.3f} ms +/- {standard_std:.3f} ms")
    print(f"  Throughput: {N/standard_time*1000/1e6:.1f} M colors/sec")

    # Triton fused kernel
    print("\n[2] Triton Fused Kernel:")
    torch.cuda.synchronize()

    # Warmup
    for _ in range(5):
        _ = apply_fused_color_triton(
            colors,
            lut.r_lut,
            lut.g_lut,
            lut.b_lut,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
        )
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(20):
        start = time.perf_counter()
        result_triton = apply_fused_color_triton(
            colors,
            lut.r_lut,
            lut.g_lut,
            lut.b_lut,
            saturation=1.3,
            shadows=1.1,
            highlights=0.9,
        )
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    triton_time = np.mean(times)
    triton_std = np.std(times)
    print(f"  Time:       {triton_time:.3f} ms +/- {triton_std:.3f} ms")
    print(f"  Throughput: {N/triton_time*1000/1e6:.1f} M colors/sec")

    # Speedup
    speedup = standard_time / triton_time
    print(f"\n[SPEEDUP] {speedup:.2f}x faster ({standard_time:.3f} ms -> {triton_time:.3f} ms)")

    # Correctness check
    diff = torch.abs(result_standard - result_triton).max().item()
    print(f"[CORRECTNESS] Max difference: {diff:.2e}", end="")
    if diff < 1e-5:
        print(" [OK]")
    else:
        print(" [WARNING: Large difference!]")

# Overall summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

print("""
The Triton fused kernel combines all color operations into a single GPU kernel:
- Phase 1: LUT lookup (temperature, brightness, contrast, gamma)
- Phase 2: Saturation + Shadows/Highlights

Benefits:
- Single kernel launch (no overhead)
- Coalesced memory access (full GPU bandwidth)
- No temporary allocations
- Process all colors in parallel

This is the recommended path for GPU color processing when Triton is available.

Next steps:
1. Integrate into ColorLUT as automatic optimization
2. Add float16 mode for even more speed (2x)
3. Consider CUDA kernel for ultimate performance (50-100x)
""")
