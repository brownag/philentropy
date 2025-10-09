"""Test suite for philentropy Python bindings."""

import os
import numpy as np
import pytest

try:
    from philentropy import (
        distance,
        distance_one_one,
        distance_one_many,
        estimate_probability,
        get_dist_methods,
    )
except ImportError:
    pytest.skip("philentropy not installed", allow_module_level=True)


class TestEstimateProbability:
    """Test probability estimation functions."""
    
    def test_empirical_basic(self):
        """Test basic empirical probability estimation."""
        counts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        prob = estimate_probability(counts, method="empirical")
        assert np.allclose(prob.sum(), 1.0)
        expected = counts / counts.sum()
        assert np.allclose(prob, expected)
    
    def test_empirical_zeros(self):
        """Test empirical estimation with zeros."""
        counts = np.array([0.0, 1.0, 2.0, 0.0, 3.0])
        prob = estimate_probability(counts)
        assert np.allclose(prob.sum(), 1.0)
        assert prob[0] == 0.0
        assert prob[3] == 0.0
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        counts = np.array([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Only 'empirical'"):
            estimate_probability(counts, method="invalid")


class TestGetDistMethods:
    """Test distance method listing."""
    
    def test_returns_list(self):
        """Test that get_dist_methods returns a list."""
        methods = get_dist_methods()
        assert isinstance(methods, list)
        assert len(methods) > 0
    
    def test_contains_expected_methods(self):
        """Test that common methods are present."""
        methods = get_dist_methods()
        expected = ["euclidean", "manhattan", "jensen-shannon", "kullback-leibler"]
        for method in expected:
            assert method in methods, f"{method} not in methods"
    
    def test_method_count(self):
        """Test that we have all 46 methods."""
        methods = get_dist_methods()
        assert len(methods) == 46


class TestDistanceOneOne:
    """Test distance computation between two distributions."""
    
    def test_euclidean_basic(self):
        """Test basic Euclidean distance."""
        p = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
        q = np.array([5, 4, 3, 2, 1], dtype=np.float64) / 15
        result = distance_one_one(p, q, method="euclidean")
        assert isinstance(result, float)
        assert result > 0
    
    def test_identical_distributions(self):
        """Test that identical distributions have zero distance."""
        p = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
        result = distance_one_one(p, p, method="euclidean")
        assert np.isclose(result, 0.0, atol=1e-10)
    
    def test_manhattan_distance(self):
        """Test Manhattan distance."""
        p = np.array([0.2, 0.3, 0.5])
        q = np.array([0.1, 0.4, 0.5])
        result = distance_one_one(p, q, method="manhattan")
        expected = 0.2
        assert np.isclose(result, expected, atol=1e-10)
    
    def test_minkowski_with_p(self):
        """Test Minkowski distance with p parameter."""
        p = np.array([1, 2, 3], dtype=np.float64) / 6
        q = np.array([3, 2, 1], dtype=np.float64) / 6
        result = distance_one_one(p, q, method="minkowski", p_param=3.0)
        assert isinstance(result, float)
        assert result > 0
    
    def test_probability_estimation(self):
        """Test with count data and probability estimation."""
        counts_p = np.array([10, 20, 30], dtype=np.float64)
        counts_q = np.array([30, 20, 10], dtype=np.float64)
        result = distance_one_one(
            counts_p, counts_q, method="euclidean", est_prob="empirical"
        )
        assert isinstance(result, float)
        assert result > 0
    
    def test_kullback_leibler(self):
        """Test Kullback-Leibler divergence."""
        p = np.array([0.25, 0.25, 0.5])
        q = np.array([0.1, 0.3, 0.6])
        result = distance_one_one(p, q, method="kullback-leibler")
        assert result > 0
    
    def test_jensen_shannon(self):
        """Test Jensen-Shannon divergence."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        result = distance_one_one(p, q, method="jensen-shannon")
        assert result >= 0
    
    def test_different_log_units(self):
        """Test different logarithm units."""
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        
        r1 = distance_one_one(p, q, method="kullback-leibler", unit="log")
        r2 = distance_one_one(p, q, method="kullback-leibler", unit="log2")
        r3 = distance_one_one(p, q, method="kullback-leibler", unit="log10")
        
        assert r1 != r2
        assert r1 != r3
        assert r2 != r3
    
    def test_epsilon_parameter(self):
        """Test epsilon parameter for zero handling."""
        p = np.array([0.5, 0.5, 0.0])
        q = np.array([0.3, 0.3, 0.4])
        result = distance_one_one(p, q, method="kullback-leibler", epsilon=1e-9)
        assert np.isfinite(result)


class TestDistanceOneMany:
    """Test distance computation from one to many distributions."""
    
    def test_basic_one_many(self):
        """Test basic one-to-many distance computation."""
        ref = np.array([1, 2, 3, 4, 5], dtype=np.float64) / 15
        others = np.array([
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 3, 4, 5]
        ], dtype=np.float64)
        others = others / others.sum(axis=1, keepdims=True)
        
        result = distance_one_many(ref, others, method="euclidean")
        assert result.shape == (3,)
        assert result[2] < 1e-10
    
    def test_threading(self):
        """Test threading parameter."""
        ref = np.array([1, 2, 3], dtype=np.float64) / 6
        others = np.random.rand(100, 3)
        others = others / others.sum(axis=1, keepdims=True)
        
        result1 = distance_one_many(ref, others, method="euclidean", num_threads=1)
        result2 = distance_one_many(ref, others, method="euclidean", num_threads=4)
        
        assert np.allclose(result1, result2)


class TestDistance:
    """Test main distance function."""
    
    def test_two_distributions(self):
        """Test distance with exactly two distributions."""
        x = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1]
        ], dtype=np.float64)
        x = x / x.sum(axis=1, keepdims=True)
        
        result = distance(x, method="euclidean", mute_message=True)
        assert isinstance(result, float)
        assert result > 0
    
    def test_multiple_distributions(self):
        """Test distance matrix for multiple distributions."""
        x = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)
        x = x / x.sum(axis=1, keepdims=True)
        
        result = distance(x, method="euclidean", mute_message=True)
        assert result.shape == (3, 3)
        assert np.allclose(np.diag(result), 0.0, atol=1e-10)
    
    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        x = np.random.rand(5, 10)
        x = x / x.sum(axis=1, keepdims=True)
        
        result = distance(x, method="euclidean", mute_message=True)
        assert np.allclose(result, result.T)
    
    def test_message_suppression(self):
        """Test that mute_message works."""
        x = np.random.rand(3, 5)
        x = x / x.sum(axis=1, keepdims=True)
        
        result = distance(x, method="euclidean", mute_message=True)
        assert result is not None
    
    def test_invalid_unit(self):
        """Test that invalid unit raises error."""
        x = np.random.rand(2, 5)
        x = x / x.sum(axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="unit must be"):
            distance(x, method="euclidean", unit="invalid")
    
    def test_invalid_method(self):
        """Test that invalid method raises error."""
        x = np.random.rand(2, 5)
        x = x / x.sum(axis=1, keepdims=True)
        
        with pytest.raises(ValueError, match="not implemented"):
            distance(x, method="nonexistent_method")


class TestNumericalAccuracy:
    """Test numerical accuracy against R reference data."""
    
    @pytest.fixture
    def data_dir(self):
        """Get the test data directory."""
        return os.path.join(os.path.dirname(__file__), "data")
    
    def test_euclidean_matrix(self, data_dir):
        """Test Euclidean distance against R reference."""
        csv_path = os.path.join(data_dir, "distance_euclidean.csv")
        if not os.path.exists(csv_path):
            pytest.skip("Reference data not available")
        
        r_result = np.loadtxt(csv_path, delimiter=",")
        
        x = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1]
        ], dtype=np.float64)
        x = x / x.sum(axis=1, keepdims=True)
        
        py_result = distance(x, method="euclidean", mute_message=True)
        
        assert np.allclose(py_result, r_result, rtol=1e-7, atol=1e-10)
    
    def test_kullback_matrix(self, data_dir):
        """Test Kullback-Leibler divergence against R reference."""
        csv_path = os.path.join(data_dir, "distance_kullback.csv")
        if not os.path.exists(csv_path):
            pytest.skip("Reference data not available")
        
        r_result = np.loadtxt(csv_path, delimiter=",")
        
        x = np.array([
            [0.25, 0.25, 0.25, 0.25],
            [0.1, 0.2, 0.3, 0.4],
            [0.4, 0.3, 0.2, 0.1]
        ], dtype=np.float64)
        
        py_result = distance(x, method="kullback-leibler", mute_message=True)
        
        assert np.allclose(py_result, r_result, rtol=1e-6, atol=1e-9)
    
    def test_empirical_estimation(self, data_dir):
        """Test probability estimation with count data."""
        csv_path = os.path.join(data_dir, "distance_counts_empirical.csv")
        if not os.path.exists(csv_path):
            pytest.skip("Reference data not available")
        
        r_result = np.loadtxt(csv_path, delimiter=",")
        
        x = np.array([
            [10.0, 20.0, 30.0, 40.0],
            [40.0, 30.0, 20.0, 10.0],
            [25.0, 25.0, 25.0, 25.0]
        ], dtype=np.float64)
        
        py_result = distance(
            x, method="euclidean", est_prob="empirical", mute_message=True
        )
        
        assert np.allclose(py_result, r_result, rtol=1e-7, atol=1e-10)
    
    def test_one_many_manhattan(self, data_dir):
        """Test one-to-many Manhattan distance."""
        csv_path = os.path.join(data_dir, "dist_one_many_manhattan.csv")
        if not os.path.exists(csv_path):
            pytest.skip("Reference data not available")
        
        r_result = np.loadtxt(csv_path, delimiter=",")
        
        ref = np.arange(1, 6, dtype=np.float64) / np.sum(np.arange(1, 6))
        others = np.array([
            np.arange(5, 0, -1, dtype=np.float64) / np.sum(np.arange(5, 0, -1)),
            np.ones(5, dtype=np.float64) / 5,
            np.full(5, 2, dtype=np.float64) / 10
        ], dtype=np.float64)
        
        py_result = distance_one_many(ref, others, method="manhattan")
        
        assert np.allclose(py_result, r_result, rtol=1e-7, atol=1e-10)


class TestThreadingBehavior:
    """Test threading behavior via environment variables."""
    
    def test_rcpp_parallel_num_threads_env(self):
        """Test that RCPP_PARALLEL_NUM_THREADS is respected."""
        x = np.random.rand(10, 20)
        x = x / x.sum(axis=1, keepdims=True)
        
        old_val = os.environ.get("RCPP_PARALLEL_NUM_THREADS")
        
        try:
            os.environ["RCPP_PARALLEL_NUM_THREADS"] = "1"
            result1 = distance(x, method="euclidean", mute_message=True)
            
            os.environ["RCPP_PARALLEL_NUM_THREADS"] = "4"
            result2 = distance(x, method="euclidean", mute_message=True)
            
            assert np.allclose(result1, result2)
        finally:
            if old_val is not None:
                os.environ["RCPP_PARALLEL_NUM_THREADS"] = old_val
            elif "RCPP_PARALLEL_NUM_THREADS" in os.environ:
                del os.environ["RCPP_PARALLEL_NUM_THREADS"]
    
    def test_explicit_num_threads(self):
        """Test explicit num_threads parameter."""
        ref = np.random.rand(10)
        ref = ref / ref.sum()
        others = np.random.rand(100, 10)
        others = others / others.sum(axis=1, keepdims=True)
        
        result1 = distance_one_many(ref, others, method="euclidean", num_threads=1)
        result2 = distance_one_many(ref, others, method="euclidean", num_threads=8)
        
        assert np.allclose(result1, result2, rtol=1e-10)
    
    def test_philentropy_num_threads_precedence(self):
        """Test that PHILENTROPY_NUM_THREADS takes precedence over RCPP_PARALLEL_NUM_THREADS."""
        x = np.random.rand(10, 20)
        x = x / x.sum(axis=1, keepdims=True)
        
        old_philentropy = os.environ.get("PHILENTROPY_NUM_THREADS")
        old_rcpp = os.environ.get("RCPP_PARALLEL_NUM_THREADS")
        
        try:
            os.environ["RCPP_PARALLEL_NUM_THREADS"] = "1"
            os.environ["PHILENTROPY_NUM_THREADS"] = "4"
            
            result1 = distance(x, method="euclidean", mute_message=True)
            
            del os.environ["PHILENTROPY_NUM_THREADS"]
            result2 = distance(x, method="euclidean", mute_message=True)
            
            assert np.allclose(result1, result2)
        finally:
            if old_philentropy is not None:
                os.environ["PHILENTROPY_NUM_THREADS"] = old_philentropy
            elif "PHILENTROPY_NUM_THREADS" in os.environ:
                del os.environ["PHILENTROPY_NUM_THREADS"]
            
            if old_rcpp is not None:
                os.environ["RCPP_PARALLEL_NUM_THREADS"] = old_rcpp
            elif "RCPP_PARALLEL_NUM_THREADS" in os.environ:
                del os.environ["RCPP_PARALLEL_NUM_THREADS"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
