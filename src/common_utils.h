#ifndef PHILENTROPY_COMMON_UTILS_H
#define PHILENTROPY_COMMON_UTILS_H

#include <string>
#include <cstdlib>
#include <cmath>
#include <stdexcept>

namespace philentropy {

// Validate 'p' parameter for methods that require it
// This function is safe to call from both R and Python code
inline void validate_p_parameter(const std::string& method, double p) {
    if (method == "minkowski" && std::isnan(p)) {
        throw std::invalid_argument("Please specify p for the Minkowski distance.");
    }
}

// Resolve number of threads from requested value or environment variable
// Returns the number of threads to use:
// - If requested_threads > 0, returns that value
// - Otherwise checks PHILENTROPY_NUM_THREADS environment variable
// - Otherwise checks RCPP_PARALLEL_NUM_THREADS environment variable
// - Defaults to 2 if none are specified
inline int resolve_num_threads(int requested_threads) {
    if (requested_threads > 0) {
        return requested_threads;
    }
    
    const char* philentropy_env = std::getenv("PHILENTROPY_NUM_THREADS");
    if (philentropy_env) {
        int env_threads = std::atoi(philentropy_env);
        if (env_threads > 0) {
            return env_threads;
        }
    }
    
    const char* rcpp_env = std::getenv("RCPP_PARALLEL_NUM_THREADS");
    if (rcpp_env) {
        int env_threads = std::atoi(rcpp_env);
        if (env_threads > 0) {
            return env_threads;
        }
    }
    
    return 2;
}

} // namespace philentropy

#endif // PHILENTROPY_COMMON_UTILS_H
