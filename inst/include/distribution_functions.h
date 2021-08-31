#ifndef DIST_FUN_H
#define DIST_FUN_H

#include <RcppArmadillo.h>

double mvn_entropy(const arma::mat& S);
double mvn_E_lpdf(const arma::vec& mu0, const arma::mat& Sigma0, const arma::vec& mu, const arma::mat& Sigma);
double dot_y_minus_Xb(double yty, arma::vec& Xty, arma::mat& XtX, arma::vec& mu, arma::mat& Sigma);

double ig_entropy(double a, double b);
double ig_E(double a, double b);
double ig_E_inv(double a, double b);
double ig_E_log(double a, double b);
double ig_E_lpdf(double a0, double b0, double a, double b);

arma::mat inv_wishart_E_invX(double nu, const arma::mat& S);
double inv_wishart_E_logdet(double nu, const arma::mat& S);

#endif