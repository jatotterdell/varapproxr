#ifndef HELPERS_H
#define HELPERS_H

#include <RcppArmadillo.h>

double mvn_entropy(arma::mat& S);
double ig_entropy(double a, double b);

#endif