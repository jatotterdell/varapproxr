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
