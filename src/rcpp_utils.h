#ifndef PHILENTROPY_RCPP_UTILS_H
#define PHILENTROPY_RCPP_UTILS_H

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::interfaces(r, cpp)]]

#include <Rcpp.h>
#include "common_utils.h"

namespace philentropy {

// Check for NA values in input vectors (Rcpp-specific)
inline void check_na(const Rcpp::NumericVector& P, const Rcpp::NumericVector& Q) {
    if (P.size() != Q.size())
        Rcpp::stop("Input vectors do not have the same length.");

    for (R_xlen_t i = 0; i < P.size(); ++i) {
        if (Rcpp::NumericVector::is_na(P[i]) || Rcpp::NumericVector::is_na(Q[i]))
            Rcpp::stop("Input vectors contain NA values.");
    }
}

// Rcpp wrapper for validate_p_parameter that uses Rcpp::stop instead of throw
inline void validate_p_parameter_rcpp(const std::string& method, double p) {
    try {
        validate_p_parameter(method, p);
    } catch (const std::invalid_argument& e) {
        Rcpp::stop(e.what());
    }
}

} // namespace philentropy

#endif // PHILENTROPY_RCPP_UTILS_H
