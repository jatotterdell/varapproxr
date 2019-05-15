#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>

double mvn_entropy(arma::mat& S);
double ig_entropy(double a, double b);
arma::mat pnorm_mat(arma::mat& m);
arma::mat dnorm_mat(arma::mat& m);

#endif