#!/usr/bin/env python3
"""
Example script demonstrating philentropy Python bindings.
Run after building: cd py && python examples/basic_usage.py
"""

import numpy as np

try:
    from philentropy import distance, distance_one_one, get_dist_methods
except ImportError:
    print("Error: philentropy not installed.")
    print("Build with: python setup.py build_ext --inplace")
    print("Then install with: pip install -e .")
    exit(1)


def main():
    print("=" * 60)
    print("Philentropy Python Bindings - Basic Usage Examples")
    print("=" * 60)
    
    # Show available methods
    methods = get_dist_methods()
    print(f"\nAvailable distance methods: {len(methods)}")
    print(f"Examples: {', '.join(methods[:10])}...")
    
    # Example 1: Distance between two distributions
    print("\n" + "-" * 60)
    print("Example 1: Distance between two distributions")
    print("-" * 60)
    
    p = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    p = p / p.sum()
    q = np.array([5, 4, 3, 2, 1], dtype=np.float64)
    q = q / q.sum()
    
    print(f"P: {p}")
    print(f"Q: {q}")
    
    euclidean = distance_one_one(p, q, method="euclidean")
    print(f"\nEuclidean distance: {euclidean:.6f}")
    
    manhattan = distance_one_one(p, q, method="manhattan")
    print(f"Manhattan distance: {manhattan:.6f}")
    
    jensen_shannon = distance_one_one(p, q, method="jensen-shannon")
    print(f"Jensen-Shannon divergence: {jensen_shannon:.6f}")
    
    # Example 2: Distance matrix for multiple distributions
    print("\n" + "-" * 60)
    print("Example 2: Distance matrix for multiple distributions")
    print("-" * 60)
    
    x = np.array([
        [1, 2, 3, 4, 5],
        [5, 4, 3, 2, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2]
    ], dtype=np.float64)
    x = x / x.sum(axis=1, keepdims=True)
    
    print(f"Input shape: {x.shape} (4 distributions, 5 dimensions)")
    
    dist_matrix = distance(x, method="euclidean", mute_message=True)
    print(f"\nDistance matrix shape: {dist_matrix.shape}")
    print("Distance matrix:")
    print(dist_matrix)
    
    # Example 3: Probability estimation from counts
    print("\n" + "-" * 60)
    print("Example 3: Probability estimation from count data")
    print("-" * 60)
    
    counts = np.array([
        [10, 20, 30, 40],
        [40, 30, 20, 10]
    ], dtype=np.float64)
    
    print(f"Count data:\n{counts}")
    
    result = distance(
        counts, 
        method="euclidean", 
        est_prob="empirical",
        mute_message=True
    )
    print(f"\nEuclidean distance (with probability estimation): {result:.6f}")
    
    # Example 4: Different logarithm units
    print("\n" + "-" * 60)
    print("Example 4: Information-theoretic measures with different units")
    print("-" * 60)
    
    p = np.array([0.5, 0.3, 0.2])
    q = np.array([0.2, 0.3, 0.5])
    
    kl_log = distance_one_one(p, q, method="kullback-leibler", unit="log")
    kl_log2 = distance_one_one(p, q, method="kullback-leibler", unit="log2")
    kl_log10 = distance_one_one(p, q, method="kullback-leibler", unit="log10")
    
    print(f"Kullback-Leibler (log):   {kl_log:.6f}")
    print(f"Kullback-Leibler (log2):  {kl_log2:.6f}")
    print(f"Kullback-Leibler (log10): {kl_log10:.6f}")
    
    # Example 5: Minkowski distance with different p values
    print("\n" + "-" * 60)
    print("Example 5: Minkowski distance with different p values")
    print("-" * 60)
    
    p = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
    q = np.array([5, 4, 3, 2, 1], dtype=np.float64) / 15
    
    for p_val in [1, 2, 3, 5]:
        dist = distance_one_one(p, q, method="minkowski", p_param=float(p_val))
        print(f"Minkowski (p={p_val}): {dist:.6f}")
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
