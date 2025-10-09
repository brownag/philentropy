# Python Bindings for Philentropy

This directory contains Python bindings for the philentropy package, providing access to 46 distance and similarity measures between probability distributions.

## Installation

### From Source

The package uses modern Python packaging (PEP 517/518/621) with `pyproject.toml`. Build dependencies (setuptools, pybind11) are automatically installed in an isolated environment.

```bash
cd py

# Install runtime dependencies
pip install -r requirements.txt

# Install in editable mode (for development)
pip install -e .

# Or build and install
pip install .
```

For development with testing tools:

```bash
pip install -r requirements-dev.txt
```

## Usage

```python
import numpy as np
from philentropy import distance, get_dist_methods

# Get available methods
methods = get_dist_methods()
print(f"Available methods: {len(methods)}")

# Compute distance between two distributions
x = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1]
], dtype=np.float64)
x = x / x.sum(axis=1, keepdims=True)

result = distance(x, method="euclidean")
print(f"Euclidean distance: {result}")

# Compute distance matrix for multiple distributions
x = np.array([
    [1, 2, 3, 4, 5],
    [5, 4, 3, 2, 1],
    [1, 1, 1, 1, 1]
], dtype=np.float64)
x = x / x.sum(axis=1, keepdims=True)

dist_matrix = distance(x, method="jensen-shannon", unit="log2")
print(f"Distance matrix shape: {dist_matrix.shape}")
```

## Testing

Run the test suite:

```bash
cd py
python -m pytest tests/ -v
```

Run with coverage:

```bash
python -m pytest tests/ -v --cov=philentropy --cov-report=html
```

## Thread Control

Control threading via environment variable (mirrors R behavior):

```python
import os
os.environ["RCPP_PARALLEL_NUM_THREADS"] = "4"

# Or pass explicitly
from philentropy import distance_one_many
result = distance_one_many(ref, others, method="euclidean", num_threads=4)
```

## Probability Estimation

Estimate probabilities from count data:

```python
import numpy as np
from philentropy import distance

# Count data
counts = np.array([
    [10, 20, 30],
    [30, 20, 10]
], dtype=np.float64)

# Automatically estimate probabilities
result = distance(counts, method="euclidean", est_prob="empirical")
```

## API Reference

### Main Functions

- `distance(x, method, ...)`: Main distance computation
- `distance_one_one(p, q, method, ...)`: Distance between two distributions
- `distance_one_many(ref, others, method, ...)`: Distance from one to many
- `estimate_probability(x, method)`: Probability estimation from counts
- `get_dist_methods()`: List available distance methods

### Parameters

- `method`: Distance measure name (see `get_dist_methods()`)
- `unit`: Logarithm unit: "log", "log2", "log10"
- `epsilon`: Small value for zero handling (default: 0.00001)
- `p`: Power parameter for Minkowski distance
- `est_prob`: If "empirical", estimate probabilities from counts
- `num_threads`: Number of parallel threads
- `mute_message`: Suppress informative messages

## Design Notes

- Distributions expected in **rows** (consistent with NumPy conventions)
- Internal row→column transposition mirrors R implementation
- Thread defaults match R: `RCPP_PARALLEL_NUM_THREADS` env or 2
- Epsilon handling identical to R for numerical stability
- All distance metrics use same C++ backend as R package
