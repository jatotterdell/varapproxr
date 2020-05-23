#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>

arma::mat blockDiag(arma::field<arma::mat>& x);
double mvn_entropy(arma::mat& S);
double ig_entropy(double a, double b);
double ig_E(double a, double b);
double ig_E_inv(double a, double b);
double ig_E_log(double a, double b);
arma::mat pnorm_mat(arma::mat& m);
arma::mat dnorm_mat(arma::mat& m);

#endif