#ifndef DIST_FUN_H
#define DIST_FUN_H

#include <RcppArmadillo.h>

double mvn_entropy(arma::mat& S);
double ig_entropy(double a, double b);
double ig_E(double a, double b);
double ig_E_inv(double a, double b);
double ig_E_log(double a, double b);

#endif