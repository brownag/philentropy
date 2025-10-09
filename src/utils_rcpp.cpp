// Compilation unit for utility functions exported to R

#include <Rcpp.h>

//[[Rcpp::export]]
Rcpp::NumericMatrix as_matrix(Rcpp::DataFrame x) {
  int nRows=x.nrows();  
  Rcpp::NumericMatrix y(nRows,x.size());
  for (int i=0; i<x.size();i++) {
       y(Rcpp::_,i) = Rcpp::NumericVector(x[i]);
  }  
  return y;
}

// [[Rcpp::export]]
Rcpp::DataFrame as_data_frame(Rcpp::NumericMatrix mat){
  Rcpp::DataFrame y;
  for (int i=0; i<mat.ncol();i++) {
       y[i] = mat(Rcpp::_,i);
  }  
  return y;
}

// [[Rcpp::export]]
SEXP sum_rcpp( SEXP vec ){
   Rcpp::NumericVector x(vec);
   double res = sum( x );
   return Rcpp::wrap( res );
}

// [[Rcpp::export]]
SEXP est_prob_empirical( SEXP CountVec ){
   Rcpp::NumericVector x(CountVec);
   double ProbMass = sum( x );
   Rcpp::NumericVector EmpiricalProb(x.size());
   EmpiricalProb = x / ProbMass;
   return Rcpp::wrap( EmpiricalProb );
}
