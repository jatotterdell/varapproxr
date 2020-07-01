// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

//' Variational Bayes for linear mixed model.
//' 
//' Variational approximation for linear mixed model assuming Gaussian distribution priors on fixed and grouped effects.
//' Priors on the variance parameters are Scaled-Inverse-Wishart distributions.
//' 
//' @param y The response vector
//' @param X The design matrix
//' @param Zlist Collection of group design matrices
//' @param J First dimension of each Z in Zlist (e.g. number of subjects)
//' @param R Second dimension of each Z in Zlist (e.g. number of variables, intercept and slope would be R = 2)
//' @param mu_beta0 The prior mean for beta
//' @param Sigma_beta0 The prior covariance for beta
//' @param xi_sigma ...
//' @param Lambda_sigma ...
//' @param xi_k ...
//' @param Lambda_k ...
//' @param tol Tolerance level
//' @param maxiter Maximum iterations
//' @param verbose Print trace of the lower bound to console. Default is \code{FALSE}.
//' @param trace Print a trace of `mu` to console.
//' @return A list containing:
//' \describe{
//'   \item{converged}{Indicator for algorithm convergence.}
//'   \item{elbo}{Vector of the ELBO sequence.} 
//'   \item{mu}{The optimised value of mu.}
//'   \item{Sigma}{The optimised value of Sigma.}
//' }
//' @export
//[[Rcpp::export]]
List vb_lmm(
    arma::vec& y,
    arma::mat& X, 
    arma::field<arma::mat>& Zlist,
    arma::vec& J,
    arma::vec& R,
    arma::vec& mu_beta0, 
    arma::mat& Sigma_beta0,
    double xi_sigma,
    arma::mat Lambda_sigma,
    arma::vec& xi_k,
    arma::field<arma::mat> Lambda_k,
    double tol = 1e-8, 
    int maxiter = 100,
    bool verbose = false,
    bool trace = false
) {
  
  // Need dimension checks, e.g. size(J) = size(R) = K
  // J(k) x R(k) == Zlist(k).n_col
  // sum(J % R) == Z.n_col
  
  int P = X.n_cols;
  int N = X.n_rows;
  int K = Zlist.n_rows;
  arma::vec K_ind = cumsum(arma::join_cols(arma::zeros(1), J % R));

  arma::mat Z = bind_cols(Zlist);
  arma::mat C = arma::join_rows(X, Z);
  // 
  // // statistics
  arma::mat CtC = trans(C)*C;
  arma::vec Cty = trans(C)*y;
  double yty = norm(y);

  // prior
  arma::vec mu0 = arma::join_cols(mu_beta0, arma::zeros(Z.n_cols));
  arma::mat inv_Sigma_beta0 = inv(Sigma_beta0);
  arma::field<arma::mat> inv_Omega_k;
  arma::mat inv_Omega = arma::zeros(P + Z.n_cols, P + Z.n_cols);
  inv_Omega.submat(0, 0, P - 1, P - 1) = inv_Sigma_beta0;

  // variational parameters for coefficients
  arma::vec beta = arma::zeros(P);
  arma::field<arma::vec> gamma_k(K);
  arma::vec gamma(sum(J%R));
  for(int k = 0; k < K; k++) {
    gamma_k(k) = arma::zeros(J(k)*R(k));
    if(k == 0) {
      gamma.subvec(0, J(k)*R(k) - 1);
    } else {
      gamma.subvec(J(k-1)*R(k-1), J(k)*R(k) - 1);
    }
  }
  arma::vec mu = arma::join_cols(beta, gamma);
  arma::mat Sigma = arma::diagmat(arma::ones(P + Z.n_cols));

  // variational parameters for variance components


  // Monitor
  bool converged = 0;
  int iterations = 0;
  arma::vec elbo(maxiter);
  arma::mat tr(P + K, maxiter);

  // for(int i = 0; i < maxiter && !converged; i++) {
  //   
  //   // Update parameters of q(beta,u)
  //   inv_sigma_0.submat(P, P, P + K - 1, P + K - 1) = inv_G;
  //   sigma = inv(Aqeps / Bqeps * CtC + inv_sigma_0);
  //   mu    = sigma * (Aqeps / Bqeps * Cty + inv_sigma_0 * mu_0);
  //   
  //   // Update parameters of q(sigma_eps)
  //   ymCmu = y - C*mu;
  //   Bqeps = Beps + 0.5*(dot(ymCmu, ymCmu) + arma::trace(CtC*sigma));
  //   
  //   // Update parameters of q(sigma_u)
  //   mu_u  = mu.subvec(P, P + K - 1);
  //   sigma_u = sigma.submat(P, P, P + K - 1, P + K - 1);
  //   mu_b  = mu.subvec(0, P - 1);
  //   sigma_b = sigma.submat(0, 0, P - 1, P - 1);
  //   Bqu   = Bu + 0.5*(dot(mu_u, mu_u) + arma::trace(sigma_u));
  //   inv_G = Aqu / Bqu * Ik;
  //   
  //   // Update ELBO
  //   elbo(i) =
  //     mvn_entropy(sigma) + ig_entropy(Aqeps, Bqeps) + ig_entropy(Aqu, Bqu) +
  //     Aeps*log(Beps) - lgamma(Aeps) - (Aeps + 1)*(log(Bqeps) - R::digamma(Aqeps)) - Beps*Aqeps/Bqeps +
  //     Au*log(Bu) - lgamma(Au) - (Au + 1)*(log(Bqu) - R::digamma(Aqu)) - Bu*Aqu/Bqu -
  //     0.5*(P * log(2*M_PI) + real(log_det(sigma_beta)) + dot(mu_b - mu_beta, inv_sigma_beta * (mu_b - mu_beta)) + arma::trace(inv_sigma_beta * sigma_b)) -
  //     0.5*(K * log(2*M_PI) + K*(log(Bqu) - R::digamma(Aqu)) + Aqu/Bqu * (dot(mu_u, mu_u) + arma::trace(sigma_u))) -
  //     0.5*(N * log(2*M_PI) + N*(log(Bqeps) - R::digamma(Aqeps)) + Aqeps/Bqeps * (dot(ymCmu, ymCmu) + arma::trace(CtC * sigma)));
  // 
  //   // Monitor convergence
  //   if(verbose) {
  //     Rcpp::Rcout << 
  //       "Iter: " << std::setw(3) << i + 1 << 
  //       "; ELBO = " << std::fixed << elbo(i) << std::endl;
  //   }
  //   
  //   if(trace)
  //     tr.col(i) = mu;
  //   
  //   if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol)
  //     converged = 1;
  //   
  //   iterations = i;
  // }

  List out = List::create(
    Named("converged") = converged,
    Named("elbo") = elbo.subvec(0, iterations),
    Named("mu") = mu,
    Named("beta") = beta,
    Named("gamma") = gamma);

  if(trace) out.push_back(tr.submat(0, 0, P + K - 1, iterations), "trace");

  return(out);
}