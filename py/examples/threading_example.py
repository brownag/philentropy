"""
Example demonstrating threading configuration in philentropy.

The package supports three ways to configure the number of threads:
1. Explicit parameter: num_threads=N
2. PHILENTROPY_NUM_THREADS environment variable (takes precedence)
3. RCPP_PARALLEL_NUM_THREADS environment variable (fallback)
4. Default: 2 threads
"""

import os
import numpy as np
from philentropy import distance, distance_one_many

np.random.seed(42)

print("=== Threading Configuration Examples ===\n")

ref = np.random.rand(10)
ref = ref / ref.sum()
others = np.random.rand(1000, 10)
others = others / others.sum(axis=1, keepdims=True)

print("1. Default threading (2 threads):")
result1 = distance_one_many(ref, others, method="euclidean")
print(f"   Computed {len(result1)} distances\n")

print("2. Explicit num_threads parameter (4 threads):")
result2 = distance_one_many(ref, others, method="euclidean", num_threads=4)
print(f"   Computed {len(result2)} distances")
print(f"   Results match: {np.allclose(result1, result2)}\n")

print("3. Using RCPP_PARALLEL_NUM_THREADS environment variable:")
old_rcpp = os.environ.get("RCPP_PARALLEL_NUM_THREADS")
os.environ["RCPP_PARALLEL_NUM_THREADS"] = "3"
result3 = distance_one_many(ref, others, method="euclidean")
print(f"   Computed {len(result3)} distances")
print(f"   Results match: {np.allclose(result1, result3)}")
if old_rcpp:
    os.environ["RCPP_PARALLEL_NUM_THREADS"] = old_rcpp
else:
    del os.environ["RCPP_PARALLEL_NUM_THREADS"]
print()

print("4. Using PHILENTROPY_NUM_THREADS (takes precedence):")
os.environ["RCPP_PARALLEL_NUM_THREADS"] = "1"
os.environ["PHILENTROPY_NUM_THREADS"] = "8"
result4 = distance_one_many(ref, others, method="euclidean")
print(f"   RCPP_PARALLEL_NUM_THREADS=1 (ignored)")
print(f"   PHILENTROPY_NUM_THREADS=8 (used)")
print(f"   Computed {len(result4)} distances")
print(f"   Results match: {np.allclose(result1, result4)}")
del os.environ["RCPP_PARALLEL_NUM_THREADS"]
del os.environ["PHILENTROPY_NUM_THREADS"]
print()

print("5. Explicit parameter overrides environment variables:")
os.environ["PHILENTROPY_NUM_THREADS"] = "8"
result5 = distance_one_many(ref, others, method="euclidean", num_threads=1)
print(f"   PHILENTROPY_NUM_THREADS=8 (ignored)")
print(f"   num_threads=1 (used)")
print(f"   Computed {len(result5)} distances")
print(f"   Results match: {np.allclose(result1, result5)}")
del os.environ["PHILENTROPY_NUM_THREADS"]
print()

print("All configurations produce identical results!")
print("Use the method that best fits your workflow.")
