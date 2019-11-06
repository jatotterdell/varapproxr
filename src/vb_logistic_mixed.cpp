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
//' @param Au The prior shape for u
//' @param Bu The prior scale for u
//' @param Aqu The initial value for Aqu
//' @param tol Tolerance level
//' @param maxiter Maximum iterations
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @return A list containing:
//' \describe{
//'   \item{converged}{Indicator for algorithm convergence.}
//'   \item{elbo}{Vector of the ELBO sequence.} 
//'   \item{mu}{The optimised value of mu.}
//'   \item{sigma}{The optimised value of sigma.}
//' }
//' 
//' @export
//[[Rcpp::export]]
List vb_glmm(
    const arma::mat& X, 
    const arma::mat& Z,
    const arma::vec& y,
    const arma::vec& mu_beta, 
    const arma::mat& sigma_beta, 
    arma::vec& mu, 
    arma::mat& sigma,
    double Au = 1.0,
    double Bu = 1.0,
    double Bqu = 1.0,
    double tol = 1e-8, 
    int maxiter = 100,
    bool verbose = false
) {
  
  int K = Z.n_cols;
  int P = X.n_cols;
  int N = X.n_rows;
  

  
  // pre-compute
  double Aqu   = Au + 0.5*K;
  arma::mat Ik = arma::eye<arma::mat>(K, K);
  arma::mat Xi(P + K, P + K);
  arma::vec xi = y;
  arma::vec lam = xi;
  arma::mat inv_G = Aqu / Bqu * Ik;
  arma::mat C = arma::join_rows(X, Z);
  arma::mat CtC = trans(C)*C;
  arma::vec Cty = trans(C) * (y - 0.5);
  arma::mat inv_sigma_beta =  inv(sigma_beta);
  arma::mat inv_sigma_beta_x_mu_beta = inv_sigma_beta * mu_beta;
  arma::mat inv_sigma_0 = arma::join_rows(
    arma::join_cols(inv_sigma_beta, arma::zeros(K, P)),
    arma::join_cols(arma::zeros(P, K), inv_G));
  arma::vec mu_0 = arma::join_cols(mu_beta, arma::zeros(K));
  arma::mat sigma_0 = inv(inv_sigma_0);
  arma::vec inv_sigma_0_x_mu_0 = inv_sigma_0 * mu_0;
  arma::vec Cty_p_inv_sigma_0_x_mu_0 = Cty + inv_sigma_0_x_mu_0;
  arma::vec mu_u(K);
  arma::vec mu_b(P);
  arma::mat sigma_u(K,K);
  arma::mat sigma_b(P,P);
  
  bool converged = 0;
  int iterations = 0;
  arma::mat trace(maxiter, P);
  arma::vec elbo(maxiter);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update variational parameters
    Xi = sigma + mu * trans(mu);
    xi = sqrt(diagvec(C * Xi * trans(C)));
    lam = tanh(xi/2)/(2*xi);
    
    // Update parameters of q(b, u)
    inv_sigma_0.submat(P, P, P + K - 1, P + K - 1) = inv_G;
    sigma = inv(trans(C) * diagmat(lam) * C + inv_sigma_0);
    inv_sigma_0_x_mu_0 = inv_sigma_0 * mu_0;
    Cty_p_inv_sigma_0_x_mu_0 = Cty + inv_sigma_0_x_mu_0;
    mu = sigma * Cty_p_inv_sigma_0_x_mu_0;
    
    // Update parameters of q(sigma_u)
    mu_u  = mu.subvec(P, P + K - 1);
    sigma_u = sigma.submat(P, P, P + K - 1, P + K - 1);
    mu_b  = mu.subvec(0, P - 1);
    sigma_b = sigma.submat(0, 0, P - 1, P - 1);
    Bqu   = Bu + 0.5*(dot(mu_u, mu_u) + arma::trace(sigma_u));
    inv_G = Aqu / Bqu * Ik;
    
    elbo(i) = 
      mvn_entropy(sigma) + ig_entropy(Aqu, Bqu) +
      Au*log(Bu) - lgamma(Au) - (Au + 1)*(log(Bqu) - R::digamma(Aqu)) - Bu*Aqu/Bqu -
      0.5*(P * log(2*M_PI) + real(log_det(sigma_beta)) + dot(mu_b - mu_beta, inv_sigma_beta * (mu_b - mu_beta)) + arma::trace(inv_sigma_beta * sigma_b)) -
      0.5*(K * log(2*M_PI) + log(K) - log(Bqu) + R::digamma(Aqu) + Aqu/Bqu * (dot(mu_u, mu_u) + arma::trace(sigma_u))) +
      dot(y - 0.5, C*mu) + arma::trace(trans(C) * diagmat(lam) * C * sigma + mu*trans(mu)) +
      sum(0.5*xi - log(1 + exp(xi)) + (xi/4) % tanh(xi/2));
    
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
                      Named("Aqu") = Aqu,
                      Named("Bqu") = Bqu,
                      Named("xi") = xi);
}