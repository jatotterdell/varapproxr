// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include "distribution_functions.h"
#include <Rcpp.h>
#include <RcppArmadillo.h>
#include <Rmath.h>
#include <iostream>
#include <iomanip>

using namespace Rcpp;

//' Perform mean-field variational inference for 
//' a Poisson regression model.
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param n The offset term
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
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
    }
    iterations = i;
  }
  
  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("Sigma") = Sigma);
}


//' Perform mean-field variational inference for 
//' a Poisson mixed-effects regression model.
//' 
//' @param X The design matrix
//' @param Zlist The random effect design matrices
//' @param y The response vector
//' @param n The offset term
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @param a0 Half-Cauchy scale hyper-parameter
//' @param tol Tolerance for convergence of the elbo
//' @param maxiter Maximum number of iterations allowed
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return v A list of relevant outputs
//' 
//' @export
// [[Rcpp::export]]
List vb_pois_mm(
    const arma::mat& X, 
    arma::field<arma::mat>& Zlist,
    const arma::vec& y,
    const arma::vec& n,
    arma::vec& mu0,
    const arma::mat& Sigma0,
    arma::vec& a0,
    double tol = 1e-8, 
    int maxiter = 100, 
    bool verbose = false) {
  
  // Should do dimensions checks etc.
  
  // Monitor
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  // data and design
  double N = X.n_rows;
  double P = X.n_cols;
  double R = Zlist.n_rows;
  arma::vec K(R);
  for(unsigned int i = 0; i < R; i++) {
    K(i) = Zlist(i,0).n_cols; 
  }
  arma::vec K_ind = cumsum(arma::join_cols(arma::zeros(1), K));
  arma::mat Z = bind_cols(Zlist);
  arma::mat C = arma::join_rows(X, Z);
  
  // statistics
  arma::vec lnn = log(n);
  arma::mat Ct = trans(C);
  arma::mat CtC = Ct*C;
  arma::vec Cty = Ct*y;
  double lfacy = -sum(lfactorial(arma2vec(y)));
  
  // prior
  mu0 = arma::join_cols(mu0, arma::zeros(sum(K)));
  arma::mat invSig0 = inv(Sigma0);
  arma::mat M = arma::zeros(P + sum(K), P + sum(K));
  M.submat(0, 0, P - 1, P - 1) = inv(Sigma0);
  
  // variational parameters
  arma::vec beta = arma::zeros(P);
  arma::vec gamma = arma::zeros(sum(K));
  arma::vec mu = arma::join_cols(beta, gamma);
  arma::mat Sigma = arma::diagmat(arma::ones(P + sum(K)));
  
  arma::vec a_sigma = (K + 1) / 2;
  arma::vec b_sigma = arma::ones(R);
  arma::vec E_inv_sigma = a_sigma / b_sigma;
  arma::vec a_lambda = arma::ones(R);
  arma::vec b_lambda = 1 / (E_inv_sigma + pow(a0, -2));
  arma::vec E_inv_lambda = a_lambda / b_lambda;
  arma::vec omega = arma::zeros(N);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    // Update variational parameters
    for(int r = 0; r < R; r++) {
      M.submat(P + K_ind(r),P + K_ind(r), P + K_ind(r+1) - 1, P + K_ind(r+1) - 1) =
        E_inv_sigma(r) * arma::eye<arma::mat>(K(r), K(r));
    }
    omega = exp(lnn + C * mu + diagvec(C * Sigma * Ct)/2);
    Sigma = inv(Ct * diagmat(omega) * C + M);
    mu += Sigma * (Cty - Ct*omega - M * (mu - mu0));
    beta = mu.subvec(0, P - 1);
    gamma = mu.subvec(P, P + sum(K) - 1);
    for(int r = 0; r < R; r++) {
      double tmp = arma::norm(gamma.subvec(K_ind(r), K_ind(r+1) - 1)) +
        arma::trace(Sigma.submat(P + K_ind(r), P + K_ind(r), P + K_ind(r+1) - 1, P + K_ind(r+1) - 1));
      E_inv_lambda(r) = 1/(E_inv_sigma(r) + pow(a0(r), -2));
      b_sigma(r) = E_inv_lambda(r) + 0.5 * tmp;
      E_inv_sigma(r) = a_sigma(r) / b_sigma(r);
    }
    
    // Update ELBO
    elbo(i) = mvn_entropy(Sigma) -
      0.5*P*real(log_det(Sigma0)) -
      0.5*as_scalar(trans(mu - mu0) * M * (mu - mu0)) -
      0.5*trace( M * Sigma ) +
      dot(y, lnn + C*mu) - sum(exp(lnn + C*mu + diagvec(C*Sigma*Ct)/2)) + lfacy;
    for(int r = 0; r < R; r++) {
      double tmp = arma::norm(gamma.subvec(K_ind(r), K_ind(r+1) - 1)) +
        arma::trace(Sigma.submat(P + K_ind(r), P + K_ind(r), P + K_ind(r+1) - 1, P + K_ind(r+1) - 1));
      elbo(i) += E_inv_lambda(r)*E_inv_sigma(r) - log(a0(r)) - log(E_inv_sigma(r) + pow(a0(r), -2)) + 
        lgamma(a_sigma(r)) - a_sigma(r) * log(E_inv_lambda(r)) + 0.5 * tmp;
    }

    if(verbose)
      Rcpp::Rcout << "Iter: " << std::setw(3) << i << "; ELBO = " << std::fixed << elbo(i) << std::endl;

    // Check for convergence
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol) {
      converged = 1;
    }
    iterations = i;
  }

  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("beta") = beta,
                      Named("gamma") = gamma,
                      Named("Sigma") = Sigma,
                      Named("a_sigma") = a_sigma,
                      Named("b_sigma") = b_sigma,
                      Named("a_lambda") = a_lambda,
                      Named("b_lambda") = b_lambda,
                      Named("CtC") = CtC,
                      Named("Cty") = Cty);
}