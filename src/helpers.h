#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>

arma::mat woodbury(arma::mat& A, arma::mat& B, arma::mat& C, arma::mat& D);
Rcpp::NumericVector arma2vec(arma::vec x);
arma::mat blockDiag(arma::field<arma::mat>& x);
arma::mat bind_cols(arma::field<arma::mat>& x);
  
arma::mat pnorm_mat(arma::mat& m);
arma::mat dnorm_mat(arma::mat& m);

// quadrature constants used in knowles-minka-wand updates
const arma::vec MS_p;
const arma::vec MS_s;

#endif