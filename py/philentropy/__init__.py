"""
Philentropy: Similarity and Distance Quantification between Probability Functions

This package provides Python bindings for the philentropy R package,
implementing 46 fundamental distance and similarity measures between
probability density functions.
"""

import pathlib
import re

from .core import (
    distance,
    distance_one_one,
    distance_one_many,
    estimate_probability,
    get_dist_methods,
)


def _get_version():
    pyproject_path = pathlib.Path(__file__).parent.parent / "pyproject.toml"
    with open(pyproject_path, "r") as f:
        content = f.read()
    match = re.search(r'^version\s*=\s*"([^"]+)"', content, re.MULTILINE)
    if match:
        return match.group(1)
    raise ValueError("Version not found in pyproject.toml")


__version__ = _get_version()
__all__ = [
    "distance",
    "distance_one_one",
    "distance_one_many", 
    "estimate_probability",
    "get_dist_methods",
]
