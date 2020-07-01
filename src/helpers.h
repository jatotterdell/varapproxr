#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>

arma::mat woodbury(arma::mat& A, arma::mat& B, arma::mat& C, arma::mat& D);
Rcpp::NumericVector arma2vec(arma::vec x);
arma::mat blockDiag(arma::field<arma::mat>& x);
arma::mat bind_cols(arma::field<arma::mat>& x);
  
double mvn_entropy(arma::mat& S);
double ig_entropy(double a, double b);
double ig_E(double a, double b);
double ig_E_inv(double a, double b);
double ig_E_log(double a, double b);
arma::mat pnorm_mat(arma::mat& m);
arma::mat dnorm_mat(arma::mat& m);

#endif