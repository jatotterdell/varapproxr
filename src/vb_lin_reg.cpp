// [[Rcpp::depends(RcppArmadillo)]]

#include "helpers.h"
#include "distribution_functions.h"
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <iostream>
#include <iomanip>

using namespace Rcpp;


//' Mean-field variational inference for a normal linear model.
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param a0 The scale hyper-parameter
//' @param b0 The shape hyper-parameter
//' @param prior The prior to be used, `1` - inverse-gamma(a0,b0), `2`- half-t(a0,b0)
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return v A list of relevant outputs
//' 
//' @export
// [[Rcpp::export]]
List vb_lm(
    const arma::mat& X, 
    const arma::vec& y,
    const arma::vec& mu0,
    const arma::mat& Sigma0,
    const double a0,
    const double b0,
    const int prior = 1,
    double tol = 1e-8, 
    int maxiter = 100, 
    bool verbose = false) {
  
  double N = X.n_rows;
  double P = X.n_cols;
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  arma::mat I = arma::eye<arma::mat>(P, P);
  arma::mat invSigma0 = inv(Sigma0);
  arma::vec invSigma0_mu0 = invSigma0 * mu0;
  double ldetSigma0 = real(log_det(Sigma0));
  
  // sufficient statistics
  double yty = dot(y, y);
  arma::mat XtX = trans(X) * X;
  arma::vec Xty = trans(X) * y;
  
  // variational parameters  
  arma::mat mu;
  arma::mat Sigma;
  double a = a0 + N / 2;
  double b = b0;
  double a_div_b = a/b;
  if(prior == 2) {
    a = (b0 + N) / 2;
  }
  double a_lam = (b0 + 1)/2;
  double b_lam = (b0 * a_div_b + pow(a0, -2));
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update variational parameters
    a_div_b = a/b;
    Sigma = inv(a_div_b * XtX + invSigma0);
    mu = Sigma * (a_div_b * Xty + invSigma0_mu0);
    if(prior == 1) {
      b = b0 + 0.5*yty - dot(mu, Xty) + 0.5*arma::trace(XtX * (Sigma + mu * trans(mu)));
    } else if (prior == 2) {
      // need to check these
      b_lam = (b0 * a_div_b + pow(a0, -2));
      b = b0 * a_lam / b_lam + 0.5*yty - dot(mu, Xty) + 0.5*arma::trace(XtX * (Sigma + mu * trans(mu)));
    }
    
    // Update ELBO
    elbo(i) = 
      mvn_entropy(Sigma) + ig_entropy(a, b) + 
      -0.5*(P * log(2*M_PI) + ldetSigma0 + dot(mu - mu0, invSigma0 * (mu - mu0)) + trace(invSigma0 * Sigma)) +
      -0.5*(N * log(2*M_PI) + N*(log(b) - R::digamma(a)) + a_div_b * (yty - 2*dot(mu, Xty) + arma::trace(XtX * (Sigma + mu * trans(mu)))));
    
    // Update ELBO prior specific terms
    if(prior == 1) {
      elbo(i) += a0*log(b0) - lgamma(a0) - (a0 + 1)*(log(b) - R::digamma(a)) - b0*a_div_b;
    } else if(prior == 2) {
      elbo(i) += ig_entropy(a_lam, b_lam) +
        b0/2*(log(b0) - ig_E_log(a_lam, b_lam)) - lgamma(b0/2) - (b0/2 + 1)*ig_E_log(a, b) - b0*ig_E_inv(a_lam, b_lam)*ig_E_inv(a, b);
        // -log(a0) - lgamma(1/2) - 3/2*ig_E_log(a_lam, b_lam) - pow(a0, -2) * ig_E_inv(a_lam, b_lam);
        // Issue with calculation above, need to fix...
    }

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
                      Named("b") = b,
                      Named("N") = N,
                      Named("yty") = yty,
                      Named("Xty") = Xty,
                      Named("XtX") = XtX,
                      Named("mu0") = mu0,
                      Named("Sigma0") = Sigma0,
                      Named("a0") = a0,
                      Named("b0") = b0);
}


//' Update mean-field variational inference for a normal linear model.
//' 
//' @param vb_fit A previous fit
//' @param X The new design matrix
//' @param y The new response vector
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return v A list of relevant outputs
//' 
//' @export
// [[Rcpp::export]]
List update_vb_lm(
    List vb_fit,
    const arma::mat& X, 
    const arma::vec& y,
    double tol = 1e-8, 
    int maxiter = 100, 
    bool verbose = false 
) {
  
  double N_old = vb_fit["N"];
  double N_new = X.n_rows;
  double N = N_old + N_new;
  double P = X.n_cols;
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  // prior parameters
  const arma::vec& mu0 = vb_fit["mu0"];
  const arma::mat& Sigma0 = vb_fit["Sigma0"];
  const double a0 = vb_fit["a0"];
  const double b0 = vb_fit["b0"];
  arma::mat I = arma::eye<arma::mat>(P, P);
  arma::mat invSigma0 = inv(Sigma0);
  arma::vec invSigma0_mu0 = invSigma0 * mu0;
  double ldetSigma0 = real(log_det(Sigma0));
  
  // sufficient statistics
  double yty_old = vb_fit["yty"];
  arma::mat XtX_old = vb_fit["XtX"];
  arma::vec Xty_old = vb_fit["Xty"];
  double yty_new = dot(y, y);
  arma::mat XtX_new = trans(X) * X;
  arma::vec Xty_new = trans(X) * y;
  double yty = yty_old + yty_new;
  arma::mat XtX = XtX_old + XtX_new;
  arma::vec Xty = Xty_old + Xty_new;
  
  // variational parameters  
  arma::mat mu;
  arma::mat Sigma;
  double a = a0 + N / 2;
  double b = vb_fit["b"];
  double a_div_b = 0.0;
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update variational parameters
    a_div_b = a/b;
    Sigma = inv(a_div_b * XtX + invSigma0);
    mu = Sigma * (a_div_b * Xty + invSigma0_mu0);
    b = b0 + 0.5*yty - dot(mu, Xty) + 0.5*arma::trace(XtX * (Sigma + mu * trans(mu)));
    
    // Update ELBO
    elbo(i) = 
      mvn_entropy(Sigma) + ig_entropy(a, b) + 
      a0*log(b0) - lgamma(a0) - (a0 + 1)*(log(b) - R::digamma(a)) - b0*a_div_b -
      0.5*(P * log(2*M_PI) + ldetSigma0 + dot(mu - mu0, invSigma0 * (mu - mu0)) + trace(invSigma0 * Sigma)) -
      0.5*(N * log(2*M_PI) + N*(log(b) - R::digamma(a)) + 
      a_div_b * (yty - 2*dot(mu, Xty) + arma::trace(XtX * (Sigma + mu * trans(mu)))));
    
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
                      Named("b") = b,
                      Named("N") = N,
                      Named("yty") = yty,
                      Named("Xty") = Xty,
                      Named("XtX") = XtX,
                      Named("mu0") = mu0,
                      Named("Sigma0") = Sigma0,
                      Named("a0") = a0,
                      Named("b0") = b0);
}