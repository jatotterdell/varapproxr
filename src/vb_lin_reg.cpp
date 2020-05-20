// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <iostream>
#include <iomanip>

using namespace Rcpp;


//' Perform mean-field variational inference for 
//' a basic linear regression model.
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param a0 The scale hyperparameter
//' @param b0 The shape hyperparameter
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return v A list of relevant outputs
//' 
//' @export
// [[Rcpp::export]]
List vb_lin_reg(
    const arma::mat& X, 
    const arma::vec& y,
    const arma::vec& mu0,
    const arma::mat& Sigma0,
    const double a0,
    const double b0,
    double tol = 1e-8, 
    int maxiter = 100, 
    bool verbose = false) {
  
  double N = X.n_rows;
  double P = X.n_cols;
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);

  arma::vec elbo(maxiter);
  
  double a_div_b = 0.0;
  arma::mat I = arma::eye<arma::mat>(P, P);
  arma::mat XtX = trans(X) * X;
  arma::vec Xty = trans(X) * y;
  arma::mat invSigma0 = inv(Sigma0);
  arma::vec invSigma0_mu0 = invSigma0 * mu0;
  double ldetSigma0 = real(log_det(Sigma0));
  arma::vec y_m_Xmu;
  
  arma::mat mu;
  arma::mat Sigma;
  double a = a0 + N / 2;
  double b = b0;
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update variational parameters
    a_div_b = a/b;
    Sigma = inv(a_div_b * XtX + invSigma0);
    mu = Sigma * (a_div_b * Xty + invSigma0_mu0);
    y_m_Xmu = y - X*mu;
    b = b0 + 0.5*(dot(y_m_Xmu, y_m_Xmu) + arma::trace(XtX * Sigma));
    
    // Update ELBO
    elbo(i) = 
      mvn_entropy(Sigma) + ig_entropy(a, b) + 
      a0*log(b0) - lgamma(a0) - (a0 + 1)*(log(b) - R::digamma(a)) - b0*a_div_b -
      0.5*(P * log(2*M_PI) + ldetSigma0 + dot(mu - mu0, invSigma0 * (mu - mu0)) + trace(invSigma0 * Sigma)) -
      0.5*(N * log(2*M_PI) + N*(log(b) - R::digamma(a)) + a_div_b * (dot(y_m_Xmu, y_m_Xmu) + trace(XtX * Sigma)));

    // Check for convergence
    if(verbose)
      Rcpp::Rcout << "Iter: " << std::setw(3) << i << "; ELBO = " << std::fixed << elbo[i] << std::endl;
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol) {
      converged = 1;
    }
    iterations = i;
  }

  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("Sigma") = Sigma,
                      Named("a") = a,
                      Named("b") = b);
}