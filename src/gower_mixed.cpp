// [[Rcpp::plugins(cpp11)]]
// [[Rcpp::depends(RcppParallel)]]
// [[Rcpp::interfaces(r, cpp)]]

#include <Rcpp.h>
#include "gower_parallel.h"

using namespace Rcpp;

// [[Rcpp::export]]
NumericMatrix gower_mixed_cpp(NumericMatrix x_num,
                           IntegerMatrix x_cat,
                           NumericVector ranges,
                           NumericVector w_num,
                           NumericVector w_cat,
                           int num_threads = 2) {
    int n = x_num.nrow();
    if (n != x_cat.nrow()) {
        stop("x_num and x_cat must have the same number of rows.");
    }

    int num_num_cols = x_num.ncol();
    int num_cat_cols = x_cat.ncol();

    if ((int)ranges.size() != num_num_cols) {
        stop("ranges must have length equal to ncol(x_num).");
    }
    if ((int)w_num.size() != num_num_cols) {
        stop("w_num must have length equal to ncol(x_num).");
    }
    if ((int)w_cat.size() != num_cat_cols) {
        stop("w_cat must have length equal to ncol(x_cat).");
    }

    NumericMatrix dist_matrix(n, n);

    GowerMixedWorker worker(x_num, x_cat, dist_matrix,
                            ranges.begin(), w_num.begin(), w_cat.begin(),
                            num_num_cols, num_cat_cols);

    RcppParallel::parallelFor(0, n, worker, 1, num_threads);

    return dist_matrix;
}

// [[Rcpp::export]]
NumericMatrix gower_cross_cpp(NumericMatrix x_num,
                              IntegerMatrix x_cat,
                              NumericMatrix y_num,
                              IntegerMatrix y_cat,
                              NumericVector ranges,
                              NumericVector w_num,
                              NumericVector w_cat,
                              int num_threads = 2) {
    int nx = x_num.nrow();
    int ny = y_num.nrow();

    if (nx != x_cat.nrow()) {
        stop("x_num and x_cat must have the same number of rows.");
    }
    if (ny != y_cat.nrow()) {
        stop("y_num and y_cat must have the same number of rows.");
    }
    if (x_num.ncol() != y_num.ncol()) {
        stop("x and y must have the same number of numeric columns.");
    }
    if (x_cat.ncol() != y_cat.ncol()) {
        stop("x and y must have the same number of categorical columns.");
    }

    int num_num_cols = x_num.ncol();
    int num_cat_cols = x_cat.ncol();

    if ((int)ranges.size() != num_num_cols) {
        stop("ranges must have length equal to ncol(x_num).");
    }
    if ((int)w_num.size() != num_num_cols) {
        stop("w_num must have length equal to ncol(x_num).");
    }
    if ((int)w_cat.size() != num_cat_cols) {
        stop("w_cat must have length equal to ncol(x_cat).");
    }

    NumericMatrix dist_matrix(nx, ny);

    GowerCrossWorker worker(x_num, x_cat, y_num, y_cat, dist_matrix,
                            ranges.begin(), w_num.begin(), w_cat.begin(),
                            num_num_cols, num_cat_cols);
    RcppParallel::parallelFor(0, nx, worker, 1, num_threads);

    return dist_matrix;
}

// [[Rcpp::export]]
NumericVector compute_gower_ranges_cpp(NumericMatrix x_num) {
    int ncol = x_num.ncol();
    int nrow = x_num.nrow();
    NumericVector ranges(ncol, 0.0);

    for (int j = 0; j < ncol; ++j) {
        double mn = R_PosInf, mx = R_NegInf;
        for (int i = 0; i < nrow; ++i) {
            double v = x_num(i, j);
            if (!std::isnan(v)) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        if (mn <= mx) {
            ranges[j] = mx - mn;
        } else {
            ranges[j] = 0.0;
        }
    }
    return ranges;
}

// [[Rcpp::export]]
NumericVector compute_gower_ranges_cross_cpp(NumericMatrix x_num, NumericMatrix y_num) {
    if (x_num.ncol() != y_num.ncol()) {
        stop("x_num and y_num must have the same number of columns.");
    }
    int ncol = x_num.ncol();
    int nrow_x = x_num.nrow();
    int nrow_y = y_num.nrow();
    NumericVector ranges(ncol, 0.0);

    for (int j = 0; j < ncol; ++j) {
        double mn = R_PosInf, mx = R_NegInf;
        // Iterate over x
        for (int i = 0; i < nrow_x; ++i) {
            double v = x_num(i, j);
            if (!std::isnan(v)) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        // Iterate over y
        for (int i = 0; i < nrow_y; ++i) {
            double v = y_num(i, j);
            if (!std::isnan(v)) {
                mn = std::min(mn, v);
                mx = std::max(mx, v);
            }
        }
        if (mn <= mx) {
            ranges[j] = mx - mn;
        } else {
            ranges[j] = 0.0;
        }
    }
    return ranges;
}
