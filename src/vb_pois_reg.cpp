// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <iostream>
#include <iomanip>

using namespace Rcpp;

template <typename T>
Rcpp::NumericVector arma2vec(const T& x) {
  return Rcpp::NumericVector(x.begin(), x.end());
}

//' Perform mean-field variational inference for 
//' a Poisson regression model.
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param n The offset term
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param a0 The scale hyper-parameter
//' @param b0 The shape hyper-parameter
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @return v A list of relevant outputs
//' 
//' @export
// [[Rcpp::export]]
List vb_pois_reg(
    const arma::mat& X, 
    const arma::vec& y,
    const arma::vec& n,
    const arma::vec& mu0,
    const arma::mat& Sigma0,
    double tol = 1e-8, 
    int maxiter = 100, 
    bool verbose = false) {
  
  double N = X.n_rows;
  double P = X.n_cols;
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  arma::mat invSig0 = inv(Sigma0);
  arma::mat mu = arma::zeros(P);
  arma::mat Sigma = arma::diagmat(arma::ones(P));
  arma::vec omega = arma::zeros(N);
  arma::vec lnn = log(n);
  arma::mat Xt = trans(X);
  arma::vec Xty = Xt * y;
  double lfacy = -sum(lfactorial(arma2vec(y)));
  
  for(int i = 0; i < maxiter && !converged; i++) {
    // Update variational parameters
    omega = exp(lnn + X * mu + diagvec(X * Sigma * Xt)/2);
    Sigma = inv(Xt * diagmat(omega) * X + invSig0);
    mu += Sigma * ((Xty - Xt*omega) - invSig0 * (mu - mu0));

    // Update ELBO
    elbo[i] = mvn_entropy(Sigma) -
      0.5*P*real(log_det(Sigma0)) -
      0.5*as_scalar(trans(mu - mu0) * invSig0 * (mu - mu0)) -
      0.5*trace( invSig0 * Sigma ) +
      dot(y, lnn + X*mu) - sum(exp(lnn + X*mu + diagvec(X*Sigma*Xt)/2)) + lfacy;
    
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