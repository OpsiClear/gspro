"""
Run all performance benchmarks.
"""

import subprocess
import sys

benchmarks = [
    "benchmark_transform.py",
    "benchmark_color.py",
]

print("=" * 80)
print("RUNNING ALL BENCHMARKS")
print("=" * 80)

for benchmark in benchmarks:
    print(f"\n{'=' * 80}")
    print(f"Running {benchmark}...")
    print("=" * 80)

    result = subprocess.run([sys.executable, benchmark], capture_output=False)

    if result.returncode != 0:
        print(f"\n[FAIL] {benchmark} failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\n[OK] {benchmark} completed successfully")

print("\n" + "=" * 80)
print("ALL BENCHMARKS COMPLETED")
print("=" * 80)
