#ifndef DIST_FUN_H
#define DIST_FUN_H

#include <RcppArmadillo.h>

double mvn_entropy(arma::mat& S);
double mvn_E_lpdf(arma::vec& mu0, arma::mat& Sigma0, arma::vec& mu, arma::mat& Sigma);
double dot_y_minus_Xb(double yty, arma::vec Xty, arma::mat& XtX, arma::vec mu, arma::mat& Sigma);

double ig_entropy(double a, double b);
double ig_E(double a, double b);
double ig_E_inv(double a, double b);
double ig_E_log(double a, double b);
double ig_E_lpdf(double a0, double b0, double a, double b);

#endif