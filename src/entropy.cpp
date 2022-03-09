// extend the boost distribution functions
// so entropy of common distributions can be calculated
// looks like entropy of common distributions not currently implemented in boost

#include <Rcpp.h>
#include <boost/math/distributions.hpp>
#include <boost/math/special_functions.hpp>

// [[Rcpp::depends(BH)]]

namespace bmath = boost::math;
using namespace Rcpp;


template <class RealType, class Policy>
inline RealType entropy(const bmath::beta_distribution<RealType, Policy>& dist)
{ // Entropy of beta distribution
  RealType a = dist.alpha();
  RealType b = dist.beta();
  return  log(bmath::beta(a, b)) - 
    (a - 1) * R::digamma(a) - 
    (b - 1) * R::digamma(b) +
    (a + b - 2) * R::digamma(a + b);
}


// [[Rcpp::export]]
double test(double a, double b) {
  bmath::beta_distribution<double> dist(a, b);
  return entropy(dist);
}
