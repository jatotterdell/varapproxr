#include <RcppArmadillo.h>
#include <Rmath.h>

double lnfactorial( int a) {
  int y;
  double z;
  if (a == 1)
    return 0;
  else
  {
    z = 0;
    for (y = 2; y<=a; y++ )
      z = log(y)+z;
    return z;
  }
}

//' Multivariate Normal Entropy
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

//' Inverse Gamma Entropy
//' 
//' Calculate and return the entropy for inverse gamma distribution
//' with supplied shape and scale.
//' 
//' @param a shape
//' @param b scale 
// [[Rcpp::export]]
double ig_entropy(double a, double b) {
  return a + log(b) + lgamma(a) - (a + 1)*R::digamma(a);
}

//' Evaluate standard normal cdf for matrix of variates
//' 
//' @param m A matrix of variates 
// [[Rcpp::export]]
arma::mat pnorm_mat(arma::mat& m) {
  int p = m.n_cols;
  int n = m.n_rows;
  arma::mat out(n, p);
  
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < p; j++) {
      out(i, j) = R::pnorm(m(i, j), 0.0, 1.0, 1, 0);
    }
  }
  return out;
}

//' Evaluate standard normal density for matrix of variates
//' 
//' @param m A matrix of variates 
// [[Rcpp::export]]
arma::mat dnorm_mat(arma::mat& m) {
  int p = m.n_cols;
  int n = m.n_rows;
  arma::mat out(n, p);
  
  for (int i = 0; i < n; i++) {
    for(int j = 0; j < p; j++) {
      out(i, j) = R::dnorm(m(i, j), 0.0, 1.0, 0);
    }
  }
  return out;
}

// Integration constants for quadrature
const arma::vec MS_p = {0.003246343272134, 0.051517477033972,
                        0.195077912673858, 0.315569823632818,
                        0.274149576158423, 0.131076880695470,
                        0.027912418727972, 0.001449567805354};
const arma::vec MS_s = {1.365340806296348, 1.059523971016916, 
                        0.830791313765644, 0.650732166639391,
                        0.508135425366489, 0.396313345166341,
                        0.308904252267995, 0.238212616409306};
