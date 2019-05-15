#include <RcppArmadillo.h>
#include <Rmath.h>

double mvn_entropy(arma::mat& S) {
  int d = S.n_rows;
  return 0.5*(d*(1 + log(2*M_PI)) + real(log_det(S)));
}

double ig_entropy(double a, double b) {
  return a + log(b) + lgamma(a) - (a + 1)*R::digamma(a);
}