// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

//' Normal parametric variational Bayes for Exponential PH Model.
//' 
//' Perform Normal approximation variational inference for 
//' proportional-hazards model with exponential base.
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param v The censoring vector
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return A list containing:
//' \describe{
//'   \item{converged}{Indicator for algorithm convergence.}
//'   \item{elbo}{Vector of the ELBO sequence.} 
//'   \item{mu}{The optimised value of mu.}
//'   \item{Sigma}{The optimised value of Sigma.}
//' }
//' 
//' @export
// [[Rcpp::export]]
List ph_exponential(
    const arma::mat& X, 
    const arma::vec& y, 
    const arma::vec& v,
    const arma::vec& mu0,
    const arma::mat& Sigma0,
    double tol = 1e-8, int maxiter = 100,
    bool verbose = false) {
  
  int N = X.n_rows;
  int P = X.n_cols;
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  arma::mat invSig0 = inv(Sigma0);
  arma::mat mu = arma::zeros(P);
  arma::mat Sigma = arma::diagmat(arma::ones(P));
  arma::vec omega = arma::zeros(N);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    // Update variational parameters
    omega = y % exp(X * mu + diagvec(X * Sigma * trans(X))/2);
    Sigma = inv(trans(X) * diagmat(omega) * X + invSig0);
    mu += Sigma * (trans(X) * (v - omega) - invSig0 * (mu - mu0));
    elbo[i] = mvn_entropy(Sigma) - 
      0.5*P*real(log_det(Sigma0)) - 
      0.5*as_scalar(trans(mu - mu0) * invSig0 * (mu - mu0)) -
      0.5*trace( invSig0 * Sigma ) +
      dot(v, X*mu) - dot(y, exp(X*mu + diagvec(X*Sigma*trans(X))/2));
    
    if(verbose)
      Rcpp::Rcout << "Iter: " << std::setw(3) << i << "; ELBO = " << std::fixed << elbo[i] << std::endl;
    
    // Check for convergence
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol) {
      converged = 1;
      iterations = i;
    }
  }
  
  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("Sigma") = Sigma);
}