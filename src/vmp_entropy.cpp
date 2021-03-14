// [[Rcpp::depends(RcppArmadillo)]]
#include "vmp_fragments.h"
#include <RcppArmadillo.h>
#include <Rmath.h>

//' Calculate entropy for Gaussian density
//' 
//' @param eta Natural parameter
// [[Rcpp::export]]
double GaussianEntropy(
    arma::vec& eta
) {
  arma::field<arma::mat> theta = GaussianCommonParameters(eta);
  int d = theta(1).n_rows;
  return 0.5*(d*(1 + log(2*M_PI)) + real(log_det(theta(1))));
}

double BernoulliEntropy(
  double eta
) {
  return log(1 + exp(eta)) - eta * exp(eta) / (1 + exp(eta));
}