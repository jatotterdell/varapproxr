// [[Rcpp::depends(RcppArmadillo)]]
#include "helpers.h"
#include "distribution_functions.h"
#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

//' Variational Bayes for linear mixed model (random intercept only).
//' 
//' 
//' @param X The design matrix
//' @param Z Group design matrix
//' @param y The response vector
//' @param mu_beta The prior mean for beta
//' @param sigma_beta The prior covariance for beta
//' @param mu Initial value for mu
//' @param sigma Initial value for sigma
//' @param Aeps The prior shape for sigma_eps
//' @param Beps The prior scale for sigma_eps
//' @param Au The prior shape for sigma_u
//' @param Bu The prior scale for sigma_u
//' @param Bqeps The intial value for Bqeps
//' @param Bqu The initial value for Bqu
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
//' 
//' @examples
//' library(nlme)
//' X <- model.matrix( ~ age + factor(Sex, levels = c("Female", "Male")), data = Orthodont)
//' Z <- kronecker(diag(1, 27), rep(1, 4))
//' y <- Orthodont$distance
//' mu0 <- rep(0, ncol(X))
//' S0 <- diag(1e8, ncol(X))
//' mu <- rep(0, ncol(X) + ncol(Z))
//' S <- diag(1, ncol(X) + ncol(Z))
//' A <- 1/100
//' B <- 1/100
//' fit <- vb_lmm_randint(X, Z, y, mu0, S0, mu, S, A, B, A, B, verbose = TRUE)
//' 
//' @export
//[[Rcpp::export]]
List vb_lmm_randint(
    const arma::mat& X, 
    const arma::mat& Z,
    const arma::vec& y,
    const arma::vec& mu_beta, 
    const arma::mat& sigma_beta, 
    arma::vec& mu, 
    arma::mat& sigma,
    double Aeps = 1.0,
    double Beps = 1.0,
    double Au = 1.0,
    double Bu = 1.0,
    double Bqeps = 1.0, 
    double Bqu = 1.0,
    double tol = 1e-8, 
    int maxiter = 100,
    bool verbose = false,
    bool trace = false
) {
  
  int K = Z.n_cols;
  int P = X.n_cols;
  int N = X.n_rows;
  
  // Pre-compute
  double Aqeps = Aeps + 0.5*N;
  double Aqu   = Au + 0.5*K;
  arma::mat Ik = arma::eye<arma::mat>(K, K);
  arma::mat inv_G = Aqu / Bqu * Ik;
  arma::mat C = arma::join_rows(X, Z);
  arma::mat CtC = trans(C)*C;
  arma::vec Cty = trans(C)*y;
  double yty = dot(y, y);
  arma::mat inv_sigma_beta =  inv(sigma_beta);
  arma::mat inv_sigma_0 = arma::join_rows(
    arma::join_cols(inv_sigma_beta, arma::zeros(K, P)),
    arma::join_cols(arma::zeros(P, K), inv_G));
  arma::vec mu_0 = arma::join_cols(mu_beta, arma::zeros(K));
  arma::vec ymCmu = y - C*mu;
  double E_dot_ymCmu;
  arma::vec mu_u(K);
  arma::vec mu_b(P);
  arma::mat sigma_u(K,K);
  arma::mat sigma_b(P,P);
  double E_dot_mu_u;
  
  // Monitor
  bool converged = 0;
  int iterations = 0;
  arma::vec elbo(maxiter);
  arma::mat tr(P + K, maxiter);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update parameters of q(beta,u)
    inv_sigma_0.submat(P, P, P + K - 1, P + K - 1) = inv_G;
    sigma = inv(Aqeps / Bqeps * CtC + inv_sigma_0);
    mu    = sigma * (Aqeps / Bqeps * Cty + inv_sigma_0 * mu_0);
    
    // Update parameters of q(sigma_eps)
    ymCmu = y - C*mu;
    E_dot_ymCmu = dot(ymCmu, ymCmu) + arma::trace(CtC*sigma);
    Bqeps = Beps + 0.5*E_dot_ymCmu;
    
    // Update parameters of q(sigma_u)
    mu_u  = mu.subvec(P, P + K - 1);
    sigma_u = sigma.submat(P, P, P + K - 1, P + K - 1);
    mu_b  = mu.subvec(0, P - 1);
    sigma_b = sigma.submat(0, 0, P - 1, P - 1);
    E_dot_mu_u = dot(mu_u, mu_u) + arma::trace(sigma_u);
    Bqu   = Bu + 0.5*E_dot_mu_u;
    inv_G = Aqu / Bqu * Ik;
    
    // Update ELBO
    elbo(i) =
      mvn_entropy(sigma) + ig_entropy(Aqeps, Bqeps) + ig_entropy(Aqu, Bqu) +
      Aeps*log(Beps) - lgamma(Aeps) - (Aeps + 1)*(log(Bqeps) - R::digamma(Aqeps)) - Beps*Aqeps/Bqeps +
      Au*log(Bu) - lgamma(Au) - (Au + 1)*(log(Bqu) - R::digamma(Aqu)) - Bu*Aqu/Bqu -
      0.5*(P * log(2*M_PI) + real(log_det(sigma_beta)) + dot(mu_b - mu_beta, inv_sigma_beta * (mu_b - mu_beta)) + arma::trace(inv_sigma_beta * sigma_b)) -
      0.5*(K * log(2*M_PI) + K*(log(Bqu) - R::digamma(Aqu)) + Aqu/Bqu * E_dot_mu_u) -
      0.5*(N * log(2*M_PI) + N*(log(Bqeps) - R::digamma(Aqeps)) + Aqeps/Bqeps * E_dot_ymCmu);
    
    // Monitor convergence
    if(verbose) {
      Rcpp::Rcout << 
        "Iter: " << std::setw(3) << i + 1 << 
          "; ELBO = " << std::fixed << elbo(i) << std::endl;
    }
    
    if(trace)
      tr.col(i) = mu;
    
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol)
      converged = 1;
    
    iterations = i;
  }
  
  List out = List::create(
    Named("converged") = converged,
    Named("elbo") = elbo.subvec(0, iterations),
    Named("mu") = mu,
    Named("sigma") = sigma,
    Named("Aqeps") = Aqeps,
    Named("Bqeps") = Bqeps,
    Named("Aqu") = Aqu,
    Named("Bqu") = Bqu);
  
  if(trace) out.push_back(tr.submat(0, 0, P + K - 1, iterations), "trace");
  
  return out;
}


//' @export
// [[Rcpp::export]]
List vb_lmm_randintslope(
  arma::field<arma::mat>& Xlist, 
  arma::field<arma::mat>& Zlist,
  arma::field<arma::vec>& ylist,
  double tol = 1e-8, 
  int maxiter = 100,
  bool verbose = false,
  bool trace = false
) {
  
  if(Zlist.n_rows != Xlist.n_rows || Zlist.n_rows != ylist.n_rows)
    Rcpp::stop("Dimension mismatch between Xlist, Zlist, ylist.");
  
  // input dimensions
  int M = Zlist.n_rows;
  int P = Xlist(0).n_cols;
  int Q = Zlist(0).n_cols;
  
  arma::vec y;
  arma::mat X;
  arma::mat Z = blockDiag(Zlist);
  for(int i = 0; i < M; i++) {
    y = arma::join_cols(y, ylist(i));
    X = arma::join_cols(X, Xlist(i));
  }
  
  // Monitor
  bool converged = 0;
  int iterations = 0;
  arma::vec elbo(maxiter);
  
  // for(int i = 0; i < maxiter && !converged; i++) {
  // }
  
  List out = List::create(
    Named("M") = M,
    Named("P") = P,
    Named("Q") = Q,
    Named("y") = y,
    Named("X") = X,
    Named("Z") = Z
  );
  
  return out;
}