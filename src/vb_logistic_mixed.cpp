// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

//' Variational Bayes for logistic mixed model.
//' 
//' 
//' @param X The design matrix
//' @param Z Group design matrix
//' @param y The response vector
//' @param mu_beta The prior mean for beta
//' @param sigma_beta The prior covariance for beta
//' @param mu Initial value for mu
//' @param sigma Initial value for sigma
//' @param A Initial value for A
//' @param E_inv_sigsq Initial value for E(1/sigma^2)
//' @param E_inv_a Initial value for E(1/a)
//' @param tol Tolerance level
//' @param maxiter Maximum iterations
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
//[[Rcpp::export]]
List jj_logistic_mixed(
    const arma::mat& X, 
    const arma::mat& Z,
    const arma::vec& y,
    const arma::vec& mu_beta, 
    const arma::mat& sigma_beta, 
    arma::vec& mu, 
    arma::mat& sigma,
    double A = 1.0,
    double E_inv_sigsq = 1.0, 
    double E_inv_a = 1.0,
    double tol = 1e-8, 
    int maxiter = 100,
    bool verbose = false
) {
  
  int K = Z.n_cols;
  int P = X.n_cols;
  int N = X.n_rows;
  
  arma::mat Ik = arma::eye<arma::mat>(K, K);
  arma::mat Xi(P + K, P + K);
  arma::vec xi = y;
  arma::vec mu_gamma(K);
  arma::mat sigma_gamma(K,K);
  double E_gammatgamma = 0.0;
  
  // pre-compute
  double inv_Asq = 1 / (A*A);
  arma::mat C = arma::join_rows(X, Z);
  arma::vec Cty = trans(C) * (y - 0.5);
  arma::mat inv_sigma_beta =  inv(sigma_beta);
  arma::mat inv_sigma_beta_x_mu_beta = inv_sigma_beta * mu_beta;
  arma::mat inv_sigma_0 = arma::join_rows(
    arma::join_cols(inv_sigma_beta, arma::zeros(K, P)),
    arma::join_cols(arma::zeros(P, K), E_inv_sigsq * Ik));
  arma::vec mu_0 = arma::join_cols(mu_beta, arma::zeros(K));
  arma::mat sigma_0 = inv(inv_sigma_0);
  arma::vec inv_sigma_0_x_mu_0 = inv_sigma_0 * mu_0;
  arma::vec Cty_p_inv_sigma_0_x_mu_0 = Cty + inv_sigma_0_x_mu_0;
  
  bool converged = 0;
  int iterations = 0;
  arma::mat trace(maxiter, P);
  arma::vec elbo(maxiter);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    Xi = sigma + mu * trans(mu);
    xi = sqrt(diagvec(C * Xi * trans(C)));
    inv_sigma_0.submat(P, P, P + K - 1, P + K - 1) = E_inv_sigsq * Ik;
    sigma = inv(trans(C) * diagmat(tanh(xi/2)/(2*xi)) * C + inv_sigma_0);
    mu = sigma * Cty_p_inv_sigma_0_x_mu_0;
    mu_gamma = mu.subvec(P, P + K - 1);
    sigma_gamma = sigma.submat(P, P, P + K - 1, P + K - 1);
    E_gammatgamma = as_scalar(trans(mu_gamma) * mu_gamma) + arma::trace(sigma_gamma);
    E_inv_a = 1 / (inv_Asq + E_inv_sigsq);
    E_inv_sigsq = (K + 1) / (2 * E_inv_a + E_gammatgamma);
    
    elbo(i) = as_scalar(ig_entropy(1, inv_Asq + E_inv_sigsq) + 
      ig_entropy(0.5*(K + 1), E_inv_a + 0.5*E_gammatgamma) +
      0.5*real(log_det(sigma)) + 0.5*real(log_det(inv_sigma_0)) +
      0.5*trans(mu) * inv(sigma) * mu - 0.5*trans(mu_0) * inv_sigma_0_x_mu_0 +
      sum(0.5*xi - log(1 + exp(xi)) + (xi/4) % tanh(xi/2)));
    
    if(verbose)
      Rcpp::Rcout << "Iter: " << std::setw(3) << i + 1 << "; ELBO = " << std::fixed << elbo(i) << std::endl;
    // check for convergence
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol) {
      converged = 1;
    }
    iterations = i;
  }
  
  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("sigma") = sigma,
                      Named("lambda_sigsq") = E_inv_sigsq,
                      Named("lambda_a") = E_inv_a,
                      Named("xi") = xi);
}