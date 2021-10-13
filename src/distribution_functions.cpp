// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rmath.h>
#include "helpers.h"
#include "gcem.hpp"

//' Multivariate Normal Entropy H[x]
//' 
//' Calculate and return the entropy for multivariate distribution
//' with supplied covariance matrix.
//' 
//' @param S Covariance matrix 
// [[Rcpp::export]]
double mvn_entropy(const arma::mat& S) {
  // if(!S.is_sympd()) 
  //   Rcpp::stop("S matrix must be symmetric positive definite.");
  int d = S.n_rows;
  return 0.5*(d*(1.0 + log(2) + log(M_PI)) + log_det_sympd(S));
}


//' Calculate E_q[ln p(x)] where q(x) = MVN(x | mu, Sigma) and p(x) = MVN(x | mu0, Sigma0)
//' 
//' E_q[ln p(x)] where x ~ MVN(mu0, Sigma0) and q(x) = MVN(x | mu, Sigma)
//'
//' @param mu0 Mean prior
//' @param Sigma0 Covariance prior
//' @param mu Variational mean
//' @param Sigma Variational covariance
// [[Rcpp::export]]
double mvn_E_lpdf(
    const arma::vec& mu0, 
    const arma::mat& Sigma0, 
    const arma::vec& mu, 
    const arma::mat& Sigma) {
  
  if(!Sigma0.is_sympd() || !Sigma.is_sympd()) 
    Rcpp::stop("Covariance matrices must be symmetric positive definite.");
  int d = Sigma.n_cols;
  arma::mat invSigma0 = inv(Sigma0);
  return -0.5*(d * log(2*M_PI) + log_det_sympd(Sigma0) + 
               dot(mu - mu0, invSigma0 * (mu - mu0)) + trace(invSigma0 * Sigma));
}


//' E[||y - Xb||^2]
//' 
//' Calculate E[||y - Xb||^2] for b ~ MVN(mu, Sigma)
//'
//' @param yty Statistic y'y
//' @param Xty Statistic X'y
//' @param XtX Statistic X'X
//' @param mu Variational mean mu
//' @param Sigma Variational variance Sigma
// [[Rcpp::export]]
double dot_y_minus_Xb(
    double yty,
    arma::vec& Xty,
    arma::mat& XtX,
    arma::vec& mu,
    arma::mat& Sigma) {
  return yty - 2*dot(mu, Xty) + arma::trace(XtX * (Sigma + mu * trans(mu)));
}


//' Inverse Gamma H[x]
//' 
//' Calculate H[x] where x ~ IG(a,b)
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_entropy(double a, double b) {
  return a + log(b) + gcem::lgamma(a) - (a + 1)*R::digamma(a);
}


//' Inverse Gamma E[x]
//' 
//' Calculate and return the expected value for x ~ IG(a,b).
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E(double a, double b) {
  if(a <= 1 || b <= 0 || ISNAN(a) || ISNAN(b))
    return NAN;
  return b / (a - 1);
}


//' Inverse Gamma E[1/x]
//' 
//' Calculate and return E[1/x] where x ~ IG(a,b).
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E_inv(double a, double b) {
  if(a <= 0 || b <= 0 || ISNAN(a) || ISNAN(b))
    return NAN;
  return a / b;
}


//' Inverse Gamma E[log(x)]
//' 
//' Calculate and return E[log(x)] where x ~ IG(a,b).
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E_log(double a, double b) {
  if(a <= 0 || b <= 0 || ISNAN(a) || ISNAN(b))
    return NAN;
  return log(b) - R::digamma(a);
}


//' Calculate E_q[ln p(x)] where q(x) = IG(x | a, b) and p(x) = IG(x | a0, b0)
//' 
//' E_q[ln p(x)] where x ~ IG(a0, b0) and q(x) = IG(x | a, b)
//'
//' @param a0 Inverse gamma prior parameter
//' @param b0 Inverse gamma prior parameter
//' @param a Inverse gamma variational parameter
//' @param b Inverse gamma variational parameter
// [[Rcpp::export]]
double ig_E_lpdf(double a0, double b0, double a, double b) {
  if(a0 <= 0 || b0 <= 0 || a <= 0 || b <= 0 || ISNAN(a0) || ISNAN(b0) || ISNAN(a) || ISNAN(b))
    return NAN;
  return a0 * log(b0) - gcem::lgamma(a0) - (a0 + 1) * ig_E_log(a, b) - b0 * ig_E_inv(a, b);
}


//' Scaled Inverse Chi squared H[x]
//' 
//' Calculate H[x] where x ~ Scaled-Inv-Chisq(nu, tau)
//' 
//' @param nu shape
//' @param tau scale 
// [[Rcpp::export]]
double scaled_inv_chisq_H(double nu, double tau) {
  return ig_entropy(nu / 2, (nu * tau) / 2);
}


//' Scaled Inverse Chi squared E[x]
//' 
//' Calculate E[x] where x ~ Scaled-Inv-Chisq(nu, tau)
//' 
//' @param nu shape
//' @param tau scale 
// [[Rcpp::export]]
double scaled_inv_chisq_E(double nu, double tau) {
  return ig_E_inv(nu / 2, (nu * tau) / 2);
}


//' Scaled Inverse Chi squared E[1/x]
//' 
//' Calculate E[1/x] where x ~ Scaled-Inv-Chisq(nu, tau^2)
//' 
//' @param nu shape
//' @param tau scale 
// [[Rcpp::export]]
double scaled_inv_chisq_E_inv(double nu, double tau) {
  return ig_E_inv(0.5 * nu, 0.5 * (nu * tau));
}


//' Scaled Inverse Chi squared E[log(x)]
//' 
//' Calculate E[log(x)] where x ~ Scaled-Inv-Chisq(nu, tau).
//' 
//' @param nu shape
//' @param tau scale 
// [[Rcpp::export]]
double scaled_inv_chisq_E_log(double nu, double tau) {
  return ig_E_log(0.5 * nu, 0.5 * nu * tau);
}


//' Inverse Wishart E[X^-1]
//' 
//' Calculate E[X^-1] where X ~ Inverse-Wishart(nu, S).
//' 
//' @param nu Degrees of freedom
//' @param S Scale matrix
// [[Rcpp::export]]
arma::mat inv_wishart_E_invX(double nu, const arma::mat& S) {
  if(!S.is_sympd()) 
    Rcpp::stop("S matrix must be symmetric positive definite.");
  if(nu <= S.n_rows)
    Rcpp::stop("nu must be greater than nrow(S).");
  return nu * inv(S);
}


//' Inverse Wishart E[log|X|]
//' 
//' Calculate E[log|X|] where X ~ Inverse-Wishart(nu, S).
//' 
//' @param nu Degrees of freedom
//' @param S Scale matrix
// [[Rcpp::export]]
double inv_wishart_E_logdet(double nu, const arma::mat& S) {
  if(!S.is_sympd()) 
    Rcpp::stop("S matrix must be symmetric positive definite.");
  if(nu <= S.n_rows)
    Rcpp::stop("nu must be greater than nrow(S).");
  double ldetS = log_det_sympd(S);
  return S.n_rows*log(0.5) + ldetS + mvdigamma(nu, S.n_rows);
}
