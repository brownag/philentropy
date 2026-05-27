

#ifndef philentropy_UTILS_H
#define philentropy_UTILS_H philentropy_UTILS_H

// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppParallel)]]

// #include <Rcpp.h>
#include <algorithm>
#include <RcppParallel.h>
#include "dist_dispatch.h"

//[[Rcpp::export]]
Rcpp::NumericMatrix as_matrix(Rcpp::DataFrame x) {
// taken from: http://stackoverflow.com/questions/24352208/best-way-to-convert-dataframe-to-matrix-in-rcpp?rq=1        
  int nRows=x.nrows();  
  Rcpp::NumericMatrix y(nRows,x.size());
  for (int i=0; i<x.size();i++) {
       y(Rcpp::_,i) = Rcpp::NumericVector(x[i]);
  }  
  return y;
}

// @export
// [[Rcpp::export]]
Rcpp::DataFrame as_data_frame(Rcpp::NumericMatrix mat){
 
 //int nRows=mat.nrow();  
  Rcpp::DataFrame y;
  for (int i=0; i<mat.ncol();i++) {
       y[i] = mat(Rcpp::_,i);
  }  
  return y;
}

// @export
// [[Rcpp::export]]
SEXP sum_rcpp( SEXP vec ){
   Rcpp::NumericVector x(vec);
   double res = sum( x );
   return Rcpp::wrap( res );
}


// @export
// [[Rcpp::export]]
SEXP est_prob_empirical( SEXP CountVec ){
   Rcpp::NumericVector x(CountVec);
   double ProbMass = sum( x );
   Rcpp::NumericVector EmpiricalProb(x.size());
   
   EmpiricalProb = x / ProbMass;
   
   //for(int i = 0; i < x.size(); i++){
   //        EmpiricalProb[i] = static_cast<double>(x[i]/ProbMass);
   //}
   return Rcpp::wrap( EmpiricalProb );
}

inline void check_na(const Rcpp::NumericVector& P, const Rcpp::NumericVector& Q) {
    if (P.size() != Q.size())
        Rcpp::stop("Input vectors do not have the same length.");

    for (R_xlen_t i = 0; i < P.size(); ++i) {
        if (Rcpp::NumericVector::is_na(P[i]) || Rcpp::NumericVector::is_na(Q[i]))
            Rcpp::stop("Input vectors contain NA values.");
    }
}


// validate 'p' for methods that require it before entering parallel regions
inline void validate_p_parameter(const std::string& method, double p) {
    const std::set<std::string> p_methods = {"minkowski"};
    if (p_methods.count(method) && std::isnan(p)) {
        Rcpp::stop("Please specify the 'p' parameter for the '" + method + "' distance.");
    }
}

// C++ helper to resolve number of threads
inline int get_num_threads_cpp(Rcpp::Nullable<int> num_threads) {
    if (num_threads.isNotNull()) {
        return Rcpp::as<int>(num_threads);
    } else {
        char* env_var = std::getenv("RCPP_PARALLEL_NUM_THREADS");
        if (env_var) {
            return std::atoi(env_var);
        } else {
            return 2;
        }
    }
}


struct DistMatrixWorker : public RcppParallel::Worker {
    RcppParallel::RMatrix<double> dists;
    RcppParallel::RMatrix<double> dist_matrix;
    std::string method;
    double epsilon;
    double p;

    DistMatrixWorker(Rcpp::NumericMatrix& dists,
                            Rcpp::NumericMatrix& dist_matrix,
                            std::string method,
                            double epsilon,
                            double p)
        : dists(dists), dist_matrix(dist_matrix), method(method), epsilon(epsilon), p(p) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            for (std::size_t j = i; j < (std::size_t)dists.ncol(); ++j) {
                double dist = 0.0;
                auto col_i = dists.column(i);
                auto col_j = dists.column(j);
                dist = dispatch_dist_internal(col_i.begin(), col_i.end(), col_j.begin(),
                                              method, "log", epsilon, p);
                dist_matrix(i, j) = dist;
                dist_matrix(j, i) = dist;
            }
        }
    }
};

Rcpp::NumericMatrix DistMatrixWithoutUnitMAT_internal(Rcpp::NumericMatrix dists,
                                                      std::string method,
                                                      bool testNA,
                                                      double epsilon,
                                                      Rcpp::Nullable<double> p,
                                                      Rcpp::Nullable<int> num_threads) {
    int n = dists.ncol();
    Rcpp::NumericMatrix dist_matrix(n, n);
    int n_threads = get_num_threads_cpp(num_threads);

    double p_val = NAN;
    if (p.isNotNull()) p_val = Rcpp::as<double>(p);
    validate_p_parameter(method, p_val);
    DistMatrixWorker worker(dists, dist_matrix, method, epsilon, p_val);
    RcppParallel::parallelFor(0, n, worker, 1, n_threads);

    return dist_matrix;
}

struct DistMatrixWorkerWithUnit : public RcppParallel::Worker {
    RcppParallel::RMatrix<double> dists;
    RcppParallel::RMatrix<double> dist_matrix;
    std::string method;
    std::string unit;
    double epsilon;

    DistMatrixWorkerWithUnit(Rcpp::NumericMatrix& dists,
                             Rcpp::NumericMatrix& dist_matrix,
                             std::string method,
                             std::string unit,
                             double epsilon)
        : dists(dists), dist_matrix(dist_matrix), method(method), unit(unit), epsilon(epsilon) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            for (std::size_t j = i; j < (std::size_t)dists.ncol(); ++j) {
                double dist = 0.0;
                auto col_i = dists.column(i);
                auto col_j = dists.column(j);
                dist = dispatch_dist_internal(col_i.begin(), col_i.end(), col_j.begin(), method, unit, epsilon, NAN);
                dist_matrix(i, j) = dist;
                dist_matrix(j, i) = dist;
            }
        }
    }
};

Rcpp::NumericMatrix DistMatrixWithUnitMAT_internal(Rcpp::NumericMatrix dists,
                                                   std::string method,
                                                   bool testNA,
                                                   double epsilon,
                                                   std::string unit,
                                                   Rcpp::Nullable<int> num_threads) {
    int n = dists.ncol();
    Rcpp::NumericMatrix dist_matrix(n, n);
    int n_threads = get_num_threads_cpp(num_threads);

    DistMatrixWorkerWithUnit worker(dists, dist_matrix, method, unit, epsilon);
    RcppParallel::parallelFor(0, n, worker, 1, n_threads);

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

    // Define groups of methods
    const std::set<std::string> unit_methods = {
      "lorentzian", "bhattacharyya", "kullback-leibler", "jeffreys",
      "k_divergence", "topsoe", "jensen-shannon", "jensen_difference", "taneja"
    };

    if (unit_methods.count(method)) {
        return DistMatrixWithUnitMAT_internal(x, method, test_na, epsilon, unit, num_threads);
    } else if (method == "minkowski") {
        if (!p.isNotNull()) {
            Rcpp::stop("Please specify p for the Minkowski distance.");
        }
        return DistMatrixWithoutUnitMAT_internal(x, method, test_na, epsilon, p, num_threads);
    } else if (method == "non-intersection") {
        Rcpp::NumericMatrix intersection_matrix = DistMatrixWithoutUnitMAT_internal(x, "intersection", test_na, epsilon, p, num_threads);
        return 1.0 - intersection_matrix;
    } else if (method == "kulczynski_s") {
        Rcpp::NumericMatrix kulczynski_d_matrix = DistMatrixWithoutUnitMAT_internal(x, "kulczynski_d", test_na, epsilon, p, num_threads);
        // Element-wise division, handling potential division by zero
        for (int i = 0; i < kulczynski_d_matrix.nrow(); ++i) {
            for (int j = 0; j < kulczynski_d_matrix.ncol(); ++j) {
                if (kulczynski_d_matrix(i, j) != 0) {
                    kulczynski_d_matrix(i, j) = 1.0 / kulczynski_d_matrix(i, j);
                } else {
                    kulczynski_d_matrix(i, j) = R_PosInf; 
                }
            }
        }
        return kulczynski_d_matrix;
    } else {
        return DistMatrixWithoutUnitMAT_internal(x, method, test_na, epsilon, p, num_threads);
    }
}

#endif // philentropy_UTILS_H
