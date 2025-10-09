// [[Rcpp::plugins(cpp11)]]

#include <Rcpp.h>
#include <cmath>
#include <limits>
#include <set>

#include "dist_interface.h"
#include "rcpp_utils.h"

using philentropy::DistConfig;
using philentropy::DistributionAxis;
using philentropy::MatrixView;

namespace {

DistConfig make_config(const std::string& method,
                       const std::string& unit,
                       double epsilon,
                       Rcpp::Nullable<double> p) {
    double p_value = std::numeric_limits<double>::quiet_NaN();
    bool has_p = false;
    if (p.isNotNull()) {
        p_value = Rcpp::as<double>(p);
        has_p = true;
    }
    philentropy::validate_p_parameter(method, p_value);
    return DistConfig{method, unit, epsilon, p_value, has_p};
}

int extract_thread_count(Rcpp::Nullable<int> num_threads) {
    int requested = 0;
    if (num_threads.isNotNull()) {
        requested = Rcpp::as<int>(num_threads);
    }
    return philentropy::resolve_num_threads(requested);
}

MatrixView make_matrix_view(Rcpp::NumericMatrix& mat) {
    return MatrixView(mat.begin(),
                      static_cast<std::size_t>(mat.nrow()),
                      static_cast<std::size_t>(mat.ncol()),
                      1,
                      static_cast<std::size_t>(mat.nrow()));
}

void subtract_from_one(Rcpp::NumericMatrix& matrix) {
    for (int i = 0; i < matrix.nrow(); ++i) {
        for (int j = 0; j < matrix.ncol(); ++j) {
            matrix(i, j) = 1.0 - matrix(i, j);
        }
    }
}

void invert_in_place(Rcpp::NumericMatrix& matrix) {
    for (int i = 0; i < matrix.nrow(); ++i) {
        for (int j = 0; j < matrix.ncol(); ++j) {
            matrix(i, j) = (matrix(i, j) != 0.0) ? 1.0 / matrix(i, j) : R_PosInf;
        }
    }
}

} // namespace

//' @title Distances and Similarities between Two Probability Density Functions
//' @name dist_one_one
//' @description This functions computes the distance/dissimilarity between two probability density functions.
//' @param P a numeric vector storing the first distribution.
//' @param Q a numeric vector storing the second distribution.
//' @param method a character string indicating whether the distance measure that should be computed.
//' @param p power of the Minkowski distance.
//' @param testNA a logical value indicating whether or not distributions shall be checked for \code{NA} values.
//' @param unit type of \code{log} function. Option are 
//' \itemize{
//' \item \code{unit = "log"}
//' \item \code{unit = "log2"}
//' \item \code{unit = "log10"}   
//' }
//' @param epsilon epsilon a small value to address cases in the distance computation where division by zero occurs. In
//' these cases, x / 0 or 0 / 0 will be replaced by \code{epsilon}. The default is \code{epsilon = 0.00001}.
//' However, we recommend to choose a custom \code{epsilon} value depending on the size of the input vectors,
//' the expected similarity between compared probability density functions and 
//' whether or not many 0 values are present within the compared vectors.
//' As a rough rule of thumb we suggest that when dealing with very large 
//' input vectors which are very similar and contain many \code{0} values,
//' the \code{epsilon} value should be set even smaller (e.g. \code{epsilon = 0.000000001}),
//' whereas when vector sizes are small or distributions very divergent then
//' higher \code{epsilon} values may also be appropriate (e.g. \code{epsilon = 0.01}).
//' Addressing this \code{epsilon} issue is important to avoid cases where distance metrics
//' return negative values which are not defined and only occur due to the
//' technical issues of computing x / 0 or 0 / 0 cases.
//' @return A single distance value
//' @examples
//' P <- 1:10 / sum(1:10)
//' Q <- 20:29 / sum(20:29)
//' dist_one_one(P, Q, method = "euclidean", testNA = FALSE)
//' @export
// [[Rcpp::export]]
double dist_one_one(const Rcpp::NumericVector& P,
                    const Rcpp::NumericVector& Q,
                    const Rcpp::String& method,
                    Rcpp::Nullable<double> p = R_NilValue,
                    const bool& testNA = true,
                    const Rcpp::String& unit = "log",
                    const double& epsilon = 0.00001) {
    std::string method_str(method);
    std::string unit_str(unit);
    DistConfig config = make_config(method_str, unit_str, epsilon, p);
    return philentropy::dist_one_one(P.begin(),
                                     Q.begin(),
                                     static_cast<std::size_t>(P.size()),
                                     config);
}

//' @title Distances and Similarities between One and Many Probability Density Functions
//' @description This functions computes the distance/dissimilarity between one probability density functions and a set of probability density functions.
//' @param P a numeric vector storing the first distribution.
//' @param dists a numeric matrix storing distributions in its rows.
//' @param method a character string indicating whether the distance measure that should be computed.
//' @param p power of the Minkowski distance.
//' @param testNA a logical value indicating whether or not distributions shall be checked for \code{NA} values.
//' @param unit type of \code{log} function. Option are
//' \itemize{
//' \item \code{unit = "log"}
//' \item \code{unit = "log2"}
//' \item \code{unit = "log10"}
//' }
//' @param epsilon epsilon a small value to address cases in the distance computation where division by zero occurs. In
//' these cases, x / 0 or 0 / 0 will be replaced by \code{epsilon}. The default is \code{epsilon = 0.00001}.
//' However, we recommend to choose a custom \code{epsilon} value depending on the size of the input vectors,
//' the expected similarity between compared probability density functions and
//' whether or not many 0 values are present within the compared vectors.
//' As a rough rule of thumb we suggest that when dealing with very large
//' input vectors which are very similar and contain many \code{0} values,
//' the \code{epsilon} value should be set even smaller (e.g. \code{epsilon = 0.000000001}),
//' whereas when vector sizes are small or distributions very divergent then
//' higher \code{epsilon} values may also be appropriate (e.g. \code{epsilon = 0.01}).
//' Addressing this \code{epsilon} issue is important to avoid cases where distance metrics
//' return negative values which are not defined and only occur due to the
//' technical issues of computing x / 0 or 0 / 0 cases.
//' @param num_threads an integer specifying the number of threads to be used for parallel computations. Default is taken from the \code{RCPP_PARALLEL_NUM_THREADS} environment variable, or \code{2} if not set.
//' @return A vector of distance values
//' @examples
//'   set.seed(2020-08-20)
//'   P <- 1:10 / sum(1:10)
//'   M <- t(replicate(100, sample(1:10, size = 10) / 55))
//'   dist_one_many(P, M, method = "euclidean", testNA = FALSE)
//' @export
// [[Rcpp::export(name = "dist_one_many")]]
Rcpp::NumericVector dist_one_many_cpp(const Rcpp::NumericVector& P,
                                      Rcpp::NumericMatrix dists,
                                      Rcpp::String method,
                                      Rcpp::Nullable<double> p = R_NilValue,
                                      bool testNA = true,
                                      Rcpp::String unit = "log",
                                      double epsilon = 0.00001,
                                      Rcpp::Nullable<int> num_threads = R_NilValue) {
    std::string method_str(method);
    std::string unit_str(unit);
    DistConfig config = make_config(method_str, unit_str, epsilon, p);
    int threads = extract_thread_count(num_threads);

    Rcpp::NumericVector dist_values(dists.nrow());
    MatrixView matrix = make_matrix_view(dists);
    philentropy::dist_one_many(P.begin(),
                               static_cast<std::size_t>(P.size()),
                               matrix,
                               DistributionAxis::Rows,
                               dist_values.begin(),
                               config,
                               threads);
    return dist_values;
}

//' @title Distances and Similarities between Many Probability Density Functions
//' @description This functions computes the distance/dissimilarity between two sets of probability density functions.
//' @param dists1 a numeric matrix storing distributions in its rows.
//' @param dists2 a numeric matrix storing distributions in its rows.
//' @param method a character string indicating whether the distance measure that should be computed.
//' @param p power of the Minkowski distance.
//' @param testNA a logical value indicating whether or not distributions shall be checked for \code{NA} values.
//' @param unit type of \code{log} function. Option are
//' \itemize{
//' \item \code{unit = "log"}
//' \item \code{unit = "log2"}
//' \item \code{unit = "log10"}
//' }
//' @param epsilon epsilon a small value to address cases in the distance computation where division by zero occurs. In
//' these cases, x / 0 or 0 / 0 will be replaced by \code{epsilon}. The default is \code{epsilon = 0.00001}.
//' However, we recommend to choose a custom \code{epsilon} value depending on the size of the input vectors,
//' the expected similarity between compared probability density functions and
//' whether or not many 0 values are present within the compared vectors.
//' As a rough rule of thumb we suggest that when dealing with very large
//' input vectors which are very similar and contain many \code{0} values,
//' the \code{epsilon} value should be set even smaller (e.g. \code{epsilon = 0.000000001}),
//' whereas when vector sizes are small or distributions very divergent then
//' higher \code{epsilon} values may also be appropriate (e.g. \code{epsilon = 0.01}).
//' Addressing this \code{epsilon} issue is important to avoid cases where distance metrics
//' return negative values which are not defined and only occur due to the
//' technical issues of computing x / 0 or 0 / 0 cases.
//' @param num_threads an integer specifying the number of threads to be used for parallel computations. Default is taken from the \code{RCPP_PARALLEL_NUM_THREADS} environment variable, or \code{2} if not set.
//' @return A matrix of distance values
//' @examples
//'   set.seed(2020-08-20)
//'   M1 <- t(replicate(10, sample(1:10, size = 10) / 55))
//'   M2 <- t(replicate(10, sample(1:10, size = 10) / 55))
//'   result <- dist_many_many(M1, M2, method = "euclidean", testNA = FALSE)
//' @export
// [[Rcpp::export(name = "dist_many_many")]]
Rcpp::NumericMatrix dist_many_many_cpp(Rcpp::NumericMatrix& dists1,
                                       Rcpp::NumericMatrix& dists2,
                                       Rcpp::String method,
                                       Rcpp::Nullable<double> p = R_NilValue,
                                       bool testNA = true,
                                       Rcpp::String unit = "log",
                                       double epsilon = 0.00001,
                                       Rcpp::Nullable<int> num_threads = R_NilValue) {
    std::string method_str(method);
    std::string unit_str(unit);
    DistConfig config = make_config(method_str, unit_str, epsilon, p);
    int threads = extract_thread_count(num_threads);

    Rcpp::NumericMatrix dist_matrix(dists1.nrow(), dists2.nrow());
    MatrixView left = make_matrix_view(dists1);
    MatrixView right = make_matrix_view(dists2);
    philentropy::dist_many_many(left,
                                DistributionAxis::Rows,
                                right,
                                DistributionAxis::Rows,
                                dist_matrix.begin(),
                                static_cast<std::size_t>(dist_matrix.nrow()),
                                config,
                                threads);
    return dist_matrix;
}

// [[Rcpp::export]]
Rcpp::NumericMatrix distance_cpp(Rcpp::NumericMatrix x,
                                 std::string method,
                                 Rcpp::Nullable<double> p,
                                 bool test_na,
                                 std::string unit,
                                 double epsilon,
                                 Rcpp::Nullable<int> num_threads) {
    const std::set<std::string> unit_methods = {
        "lorentzian", "bhattacharyya", "kullback-leibler", "jeffreys",
        "k_divergence", "topsoe", "jensen-shannon", "jensen_difference", "taneja"
    };

    DistConfig config = make_config(method, unit, epsilon, p);
    int threads = extract_thread_count(num_threads);

    MatrixView view = make_matrix_view(x);
    Rcpp::NumericMatrix result(x.ncol(), x.ncol());

    if (unit_methods.count(method) || method == "minkowski") {
        philentropy::dist_self_symmetric(view,
                                         DistributionAxis::Columns,
                                         result.begin(),
                                         static_cast<std::size_t>(result.nrow()),
                                         config,
                                         threads);
        return result;
    }

    if (method == "non-intersection") {
        DistConfig intersection_config = make_config("intersection", unit, epsilon, Rcpp::Nullable<double>());
        philentropy::dist_self_symmetric(view,
                                         DistributionAxis::Columns,
                                         result.begin(),
                                         static_cast<std::size_t>(result.nrow()),
                                         intersection_config,
                                         threads);
        subtract_from_one(result);
        return result;
    }

    if (method == "kulczynski_s") {
        DistConfig d_config = make_config("kulczynski_d", unit, epsilon, p);
        philentropy::dist_self_symmetric(view,
                                         DistributionAxis::Columns,
                                         result.begin(),
                                         static_cast<std::size_t>(result.nrow()),
                                         d_config,
                                         threads);
        invert_in_place(result);
        return result;
    }

    philentropy::dist_self_symmetric(view,
                                     DistributionAxis::Columns,
                                     result.begin(),
                                     static_cast<std::size_t>(result.nrow()),
                                     config,
                                     threads);
    return result;
}
