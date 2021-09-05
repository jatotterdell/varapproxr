// The boost library doesn't implement entropy for the available distributions.
// Entropy functions implemented here.

#include <Rcpp.h>
#include <Rmath.h>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>

// [[Rcpp::depends(BH)]]

using namespace Rcpp;

template <class RealType, class Policy>
inline RealType entropy(const boost::math::beta_distribution<RealType, Policy>& dist) {
  RealType a {dist.alpha()};
  RealType b {dist.beta()};
  return R::lbeta(a, b) - (a - 1)*R::digamma(a) - (b - 1)*R::digamma(b) +
    (a + b - 2)*R::digamma(a + b);
}


template <class RealType, class Policy>
inline RealType entropy(const boost::math::inverse_gamma_distribution<RealType, Policy>& dist) {
  RealType a {dist.shape()};
  RealType b {dist.scale()};
  return a + log(b) + lgamma(a) - (1 + a) * R::digamma(a);
}


// [[Rcpp::export]]
double test(double a, double b) {
  boost::math::inverse_gamma_distribution<> dist(a, b);
  return entropy(dist);
}
