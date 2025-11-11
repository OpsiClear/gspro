"""
Run all gslut benchmarks
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("\n" + "=" * 80)
print("Running ALL gslut Benchmarks")
print("=" * 80)

# Import and run all benchmarks
from benchmark_color_lut import (
    benchmark_color_lut,
    benchmark_individual_operations,
    benchmark_lut_sizes,
)
from benchmark_activation_lut import (
    benchmark_activation_lut,
    benchmark_lut_cluster_sizes,
    benchmark_accuracy_vs_performance,
)
from benchmark_conversions import (
    benchmark_conversions,
    benchmark_tensor_shapes,
    benchmark_dtype_comparison,
    benchmark_memory_transfer,
)

try:
    print("\n" + "#" * 80)
    print("# 1. ColorLUT Benchmarks")
    print("#" * 80)
    benchmark_color_lut()
    benchmark_individual_operations()
    benchmark_lut_sizes()

    print("\n" + "#" * 80)
    print("# 2. ActivationLUT Benchmarks")
    print("#" * 80)
    benchmark_activation_lut()
    benchmark_lut_cluster_sizes()
    benchmark_accuracy_vs_performance()

    print("\n" + "#" * 80)
    print("# 3. Conversion Benchmarks")
    print("#" * 80)
    benchmark_conversions()
    benchmark_tensor_shapes()
    benchmark_dtype_comparison()
    benchmark_memory_transfer()

    print("\n" + "=" * 80)
    print("ALL BENCHMARKS COMPLETE!")
    print("=" * 80)

except KeyboardInterrupt:
    print("\n\nBenchmarks interrupted by user.")
except Exception as e:
    print(f"\n\nError during benchmarks: {e}")
    import traceback

    traceback.print_exc()
