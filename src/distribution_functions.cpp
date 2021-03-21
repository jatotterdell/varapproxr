#include <RcppArmadillo.h>
#include <Rmath.h>

//' Multivariate Normal H[x]
//' 
//' Calculate and return the entropy for multivariate distribution
//' with supplied covariance matrix.
//' 
//' @param S Covariance matrix 
// [[Rcpp::export]]
double mvn_entropy(arma::mat& S) {
  int d = S.n_rows;
  return 0.5*(d*(1 + log(2*M_PI)) + real(log_det(S)));
}


double mvn_E_lpdf(arma::vec& mu0, arma::mat& Sigma0, arma::vec& mu, arma::mat& Sigma) {
  int d = Sigma.n_cols;
  arma::mat invSigma0 = inv(Sigma0);
  return -0.5*(d * log(2*M_PI) + real(log_det(Sigma0)) + 
               dot(mu - mu0, invSigma0 * (mu - mu0)) + trace(invSigma0 * Sigma));
}

//' Calculate E[||y - Xb||^2] for b ~ MVN(mu, Sigma)
//'
//' @param yty Statistic y'y
//' @param Xty Statistic X'y
//' @param XtX Statistic X'X
//' @param mu Variational mean mu
//' @param Sigma Variational variance Sigma
// [[Rcpp::export]]
double dot_y_minus_Xb(double yty, arma::vec Xty, arma::mat& XtX, arma::vec mu, arma::mat& Sigma
) {
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
  return a + log(b) + lgamma(a) - (a + 1)*R::digamma(a);
}

//' Inverse Gamma E[x]
//' 
//' Calculate and return the expected value for x ~ IG(a,b).
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E(double a, double b) {
  double ret = 0.0;
  if(a > 1)
    ret = b / (a - 1);
  return ret;
}

//' Inverse Gamma E[1/x]
//' 
//' Calculate and return E[1/x] where x ~ IG(a,b).
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_E_inv(double a, double b) {
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
  return log(b) - R::digamma(a);
}

//' E_q[ln p(x)] where x ~ IG(a0, b0) and q(x) = IG(x | a, b)
//'
//' @param a0 Inverse gamma prior parameter
//' @param b0 Inverse gamma prior parameter
//' @param a Inverse gamma variational parameter
//' @param b Inverse gamma variational parameter
// [[Rcpp::export]]
double ig_E_lpdf(double a0, double b0, double a, double b) {
  return a0 * log(b0) - lgamma(a0) - (a0 + 1) * ig_E_log(a, b) - b0 * ig_E_inv(a, b);
}
