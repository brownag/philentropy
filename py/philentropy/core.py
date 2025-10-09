"""Core distance computation functions for philentropy Python bindings."""

import numpy as np
from typing import Optional, Union, Literal

try:
    from . import philentropy_cpp
except ImportError:
    raise ImportError(
        "philentropy_cpp extension not found. "
        "Please build the package using: python setup.py build_ext --inplace"
    )


def estimate_probability(
    x: np.ndarray,
    method: Literal["empirical"] = "empirical"
) -> np.ndarray:
    """
    Estimate probability vectors from count vectors.
    
    This function takes a numeric count vector and returns estimated
    probabilities of the corresponding counts.
    
    Parameters
    ----------
    x : np.ndarray
        A numeric vector storing count values
    method : str, default "empirical"
        Estimation method. Currently only "empirical" is supported,
        which generates relative frequency: x/sum(x)
        
    Returns
    -------
    np.ndarray
        A numeric probability vector
        
    Examples
    --------
    >>> import numpy as np
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> estimate_probability(x, method='empirical')
    array([0.06666667, 0.13333333, 0.2       , 0.26666667, 0.33333333])
    """
    if method != "empirical":
        raise ValueError("Only 'empirical' method is currently supported")
    
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("x must be a 1D array")
    
    return x / np.sum(x)


def get_dist_methods() -> list:
    """
    Get method names for distance computation.
    
    Returns a list of distance/similarity measure names that can be used
    with the distance function.
    
    Returns
    -------
    list
        List of available distance method names
        
    Examples
    --------
    >>> methods = get_dist_methods()
    >>> 'euclidean' in methods
    True
    >>> 'jensen-shannon' in methods
    True
    """
    return philentropy_cpp.get_dist_methods()


def distance_one_one(
    p: np.ndarray,
    q: np.ndarray,
    method: str = "euclidean",
    unit: str = "log",
    epsilon: float = 0.00001,
    p_param: Optional[float] = None,
    est_prob: Optional[str] = None
) -> float:
    """
    Compute distance between two probability distributions.
    
    Parameters
    ----------
    p : np.ndarray
        First probability distribution (1D array)
    q : np.ndarray
        Second probability distribution (1D array)
    method : str, default "euclidean"
        Distance measure to compute
    unit : str, default "log"
        Logarithm unit for information-theoretic measures: "log", "log2", "log10"
    epsilon : float, default 0.00001
        Small value to address division by zero cases
    p_param : float, optional
        Power parameter for Minkowski distance
    est_prob : str, optional
        If "empirical", estimate probabilities from counts first
        
    Returns
    -------
    float
        Distance value
        
    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
    >>> q = np.array([5, 4, 3, 2, 1], dtype=np.float64) / 15
    >>> distance_one_one(p, q, method="euclidean")
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    if p.ndim != 1 or q.ndim != 1:
        raise ValueError("p and q must be 1D arrays")
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    
    result = philentropy_cpp.distance_one_one(
        p, q, method, unit, epsilon, p_param, est_prob
    )
    return float(result[0])


def distance_one_many(
    reference: np.ndarray,
    others: np.ndarray,
    method: str = "euclidean",
    unit: str = "log",
    epsilon: float = 0.00001,
    p_param: Optional[float] = None,
    est_prob: Optional[str] = None,
    num_threads: Optional[int] = None
) -> np.ndarray:
    """
    Compute distances from one reference distribution to many others.
    
    Parameters
    ----------
    reference : np.ndarray
        Reference probability distribution (1D array)
    others : np.ndarray
        Multiple distributions (2D array with distributions in rows)
    method : str, default "euclidean"
        Distance measure to compute
    unit : str, default "log"
        Logarithm unit for information-theoretic measures
    epsilon : float, default 0.00001
        Small value to address division by zero cases
    p_param : float, optional
        Power parameter for Minkowski distance
    est_prob : str, optional
        If "empirical", estimate probabilities from counts first
    num_threads : int, optional
        Number of threads for parallel computation. If None, checks
        PHILENTROPY_NUM_THREADS or RCPP_PARALLEL_NUM_THREADS environment
        variables, or defaults to 2
        
    Returns
    -------
    np.ndarray
        Array of distance values
        
    Examples
    --------
    >>> import numpy as np
    >>> ref = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
    >>> others = np.array([[5, 4, 3, 2, 1], [1, 1, 1, 1, 1]], dtype=np.float64)
    >>> others = others / others.sum(axis=1, keepdims=True)
    >>> distance_one_many(ref, others, method="euclidean")
    """
    reference = np.asarray(reference, dtype=np.float64)
    others = np.asarray(others, dtype=np.float64)
    
    if reference.ndim != 1:
        raise ValueError("reference must be a 1D array")
    if others.ndim != 2:
        raise ValueError("others must be a 2D array")
    if reference.shape[0] != others.shape[1]:
        raise ValueError("reference length must match others column count")
    
    if num_threads is None:
        num_threads = 0
    
    return philentropy_cpp.distance_one_many(
        reference, others, method, unit, epsilon, p_param, est_prob, num_threads
    )


def distance(
    x: np.ndarray,
    method: str = "euclidean",
    p: Optional[float] = None,
    unit: str = "log",
    epsilon: float = 0.00001,
    est_prob: Optional[str] = None,
    num_threads: Optional[int] = None,
    mute_message: bool = False
) -> Union[float, np.ndarray]:
    """
    Distances and similarities between probability density functions.
    
    This function computes distance/dissimilarity between probability
    density functions. For two distributions, returns a single value.
    For multiple distributions, returns a distance matrix.
    
    Parameters
    ----------
    x : np.ndarray
        2D array with probability distributions in rows
    method : str, default "euclidean"
        Distance measure to compute. See get_dist_methods() for options
    p : float, optional
        Power parameter for Minkowski distance
    unit : str, default "log"
        Logarithm unit: "log", "log2", or "log10"
    epsilon : float, default 0.00001
        Small value to address division by zero. Should be adjusted based
        on input vector size and expected similarity. For very large vectors
        with many zeros, use smaller epsilon (e.g., 1e-9). For small vectors
        or divergent distributions, larger epsilon may be appropriate (e.g., 0.01)
    est_prob : str, optional
        If "empirical", estimate probabilities from counts first
    num_threads : int, optional
        Number of threads for parallel computation. If None, checks
        PHILENTROPY_NUM_THREADS or RCPP_PARALLEL_NUM_THREADS environment
        variables, or defaults to 2
    mute_message : bool, default False
        Whether to suppress informative messages
        
    Returns
    -------
    float or np.ndarray
        For 2 distributions: single distance value
        For >2 distributions: distance matrix
        
    Examples
    --------
    >>> import numpy as np
    >>> # Two distributions
    >>> x = np.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]], dtype=np.float64)
    >>> x = x / x.sum(axis=1, keepdims=True)
    >>> distance(x, method="euclidean")
    
    >>> # Multiple distributions
    >>> x = np.array([
    ...     [1, 2, 3, 4, 5],
    ...     [5, 4, 3, 2, 1],
    ...     [1, 1, 1, 1, 1]
    ... ], dtype=np.float64)
    >>> x = x / x.sum(axis=1, keepdims=True)
    >>> dist_matrix = distance(x, method="euclidean")
    >>> dist_matrix.shape
    (3, 3)
    
    Notes
    -----
    Convention for handling edge cases:
    - 0/0 is treated as 0
    - n/0 is replaced by epsilon
    - 0 * log(0) is treated as 0
    - log(0) is replaced by log(epsilon)
    """
    x = np.asarray(x, dtype=np.float64)
    
    if x.ndim != 2:
        raise ValueError("x must be a 2D array with distributions in rows")
    
    if unit not in ["log", "log2", "log10"]:
        raise ValueError("unit must be 'log', 'log2', or 'log10'")
    
    if method not in get_dist_methods():
        raise ValueError(
            f"Method '{method}' is not implemented. "
            f"Use get_dist_methods() to see available methods."
        )
    
    if num_threads is None:
        num_threads = 0
    
    n_distributions = x.shape[0]
    
    if n_distributions == 2:
        result = distance_one_one(
            x[0], x[1], method, unit, epsilon, p, est_prob
        )
        if not mute_message:
            print(f"Metric: '{method}' with unit: '{unit}'; comparing: 2 vectors")
        return result
    else:
        result = philentropy_cpp.distance_matrix(
            x, method, unit, epsilon, p, est_prob, num_threads, mute_message
        )
        return result
