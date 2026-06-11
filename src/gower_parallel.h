#ifndef philentropy_Gower_Parallel_H
#define philentropy_Gower_Parallel_H

#include <Rcpp.h>
#include <RcppParallel.h>
#include "gower_internal.h"

struct GowerMixedWorker : public RcppParallel::Worker {
    const RcppParallel::RMatrix<double> x_num;
    const RcppParallel::RMatrix<int> x_cat;
    RcppParallel::RMatrix<double> dist_matrix;

    const double* ranges;
    const double* w_num;
    const double* w_cat;
    int num_num_cols;
    int num_cat_cols;

    GowerMixedWorker(const Rcpp::NumericMatrix& x_num_in,
                     const Rcpp::IntegerMatrix& x_cat_in,
                     Rcpp::NumericMatrix& dist_matrix_in,
                     const double* ranges_in,
                     const double* w_num_in,
                     const double* w_cat_in,
                     int num_num_cols_in,
                     int num_cat_cols_in)
        : x_num(x_num_in), x_cat(x_cat_in), dist_matrix(dist_matrix_in),
          ranges(ranges_in), w_num(w_num_in), w_cat(w_cat_in),
          num_num_cols(num_num_cols_in), num_cat_cols(num_cat_cols_in) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            for (std::size_t j = 0; j < dist_matrix.ncol(); ++j) {
                dist_matrix(i, j) = gower_mixed_row_pair(
                    x_num.row(i), x_num.row(j), num_num_cols,
                    ranges, w_num,
                    x_cat.row(i), x_cat.row(j), num_cat_cols,
                    w_cat
                );
            }
        }
    }
};

struct GowerCrossWorker : public RcppParallel::Worker {
    const RcppParallel::RMatrix<double> x_num;
    const RcppParallel::RMatrix<int> x_cat;
    const RcppParallel::RMatrix<double> y_num;
    const RcppParallel::RMatrix<int> y_cat;
    RcppParallel::RMatrix<double> dist_matrix;

    const double* ranges;
    const double* w_num;
    const double* w_cat;
    int num_num_cols;
    int num_cat_cols;

    GowerCrossWorker(const Rcpp::NumericMatrix& x_num_in,
                     const Rcpp::IntegerMatrix& x_cat_in,
                     const Rcpp::NumericMatrix& y_num_in,
                     const Rcpp::IntegerMatrix& y_cat_in,
                     Rcpp::NumericMatrix& dist_matrix_in,
                     const double* ranges_in,
                     const double* w_num_in,
                     const double* w_cat_in,
                     int num_num_cols_in,
                     int num_cat_cols_in)
        : x_num(x_num_in), x_cat(x_cat_in),
          y_num(y_num_in), y_cat(y_cat_in),
          dist_matrix(dist_matrix_in),
          ranges(ranges_in), w_num(w_num_in), w_cat(w_cat_in),
          num_num_cols(num_num_cols_in), num_cat_cols(num_cat_cols_in) {}

    void operator()(std::size_t begin, std::size_t end) {
        for (std::size_t i = begin; i < end; ++i) {
            for (std::size_t j = 0; j < (std::size_t)dist_matrix.ncol(); ++j) {
                dist_matrix(i, j) = gower_mixed_row_pair(
                    x_num.row(i), y_num.row(j), num_num_cols,
                    ranges, w_num,
                    x_cat.row(i), y_cat.row(j), num_cat_cols,
                    w_cat
                );
            }
        }
    }
};

#endif // philentropy_Gower_Parallel_H
