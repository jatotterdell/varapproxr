// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rmath.h>
#include "helpers.h"
#include "distribution_functions.h"


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

//' Variational Bayes for linear mixed model (random intercept and coefficient only).
//' 
//' Performs variational inference for random intercept and coefficient model.
//' Currently assumes that all groups have same number of parameters.
//' That is, that all Zlist elements are of equal dimension.
//' 
//' @param Xlist A list of subject specific design matrices
//' @param Zlist A list of subject specific group matrices
//' @param ylist A list of subject specific responses
//' @param beta_mu0 Prior mean for beta
//' @param beta_sigma0 Prior covariance for beta
//' @param nu_Omega0 Prior df for Omega
//' @param lambda_Omega0 Prior scale matrix for Omega
//' @param pr_Omega Prior type for Omega - 1 is IW(nu, lambda), 2 is HW(nu, 2*nu*diag(1/lambda^2))
//' @param sigma_a0 The first hyper-parameter for prior on sigma
//' @param sigma_b0 The second hyper-parameter for prior on sigma
//' @param pr_sigma The prior to use for sigma_epsilon - 1 is IG(a0,b0) and 2 is Half-t(a0, b0)
//' @param tol Tolerance level for assessing convergence
//' @param maxiter Maximum number of fixed-update iterations
//' @param verbose Print trace of ELBO
//' @param trace Return trace of parameters beta, gamma
//' @param streamlined Use streamlined updates (more efficient if dim(Zlist) is large).
//' @param use_elbo Should the ELBO be calculated and used for convergence checks?
//' 
//' @export
// [[Rcpp::export]]
List vb_lmm_randintslope(
  arma::field<arma::mat>& Xlist, 
  arma::field<arma::mat>& Zlist,
  arma::field<arma::vec>& ylist,
  const arma::vec& beta_mu0, 
  const arma::mat& beta_sigma0, 
  double nu_Omega0,
  arma::mat lambda_Omega0,
  int pr_Omega = 1,
  double sigma_a0 = 1e-2,
  double sigma_b0 = 1e-2,
  int pr_sigma = 1,
  double tol = 1e-8, 
  int maxiter = 100,
  bool verbose = false,
  bool trace = false,
  bool streamlined = false,
  bool use_elbo = true
) {
  
  if(Zlist.n_rows != Xlist.n_rows || Zlist.n_rows != ylist.n_rows)
    Rcpp::stop("Dimension mismatch between Xlist, Zlist, ylist.");
  
  // input dimensions
  int M = Zlist.n_rows;
  int P = Xlist(0).n_cols;
  arma::vec q(M);
  arma::vec r(M);
  arma::vec idq = arma::zeros(M+1);
  arma::vec idr = arma::zeros(M+1);
  
  arma::vec y;
  arma::mat X;
  arma::mat Z = blockDiag(Zlist);
  for(int m = 0; m < M; m++) {
    q(m) = Zlist(m).n_cols;
    r(m) = Zlist(m).n_rows;
    idq(m+1) = idq(m) + q(m);
    idr(m+1) = idr(m) + r(m);
    y = arma::join_cols(y, ylist(m));
    X = arma::join_cols(X, Xlist(m));
  }
  int Q = Z.n_cols;
  int N = y.n_elem;
  
  // sufficient statistics
  arma::mat C = arma::join_rows(X, Z);
  arma::mat CtC = trans(C)*C;
  arma::vec Cty = trans(C)*y;
  arma::mat XtX = trans(X)*X;
  double yty = dot(y, y);
  
  // additional variables
  arma::mat inv_beta_sigma0 = inv(beta_sigma0);
  double E_dot_y_Cb = 0.0;
  arma::mat Im = arma::eye(M, M);
  arma::vec betagamma_mu0 = join_vert(beta_mu0, arma::zeros(Q));
  arma::field<arma::mat> G_inv(2);
  arma::mat bdG;
  
  // variational parameters and associated functions
  
  // beta and gamma
  arma::vec q_betagamma_mu(P + Q);
  arma::vec q_gamma_mu(M*Q);
  arma::vec q_beta_mu(P);
  arma::mat q_betagamma_sigma(P + Q, P + Q);
  arma::mat q_gamma_sigma(Q, Q);
  
  // Omega
  double q_omega_nu = nu_Omega0 + M;
  arma::mat q_omega_lambda = lambda_Omega0;
  arma::mat E_q_omega_inv = inv_wishart_E_invX(q_omega_nu, q_omega_lambda);
  // if(pr_Omega == 2) {
  //   q_omega_nu = nu_Omega0 + M + lambda_Omega0.n_rows - 1;
  // }
  
  // note prior two only
  // arma::vec q_xi_a(q(0));
  // arma::vec q_xi_b(q(0));
  // arma::vec E_ig_inv_xi(q(0));
  
  // sigma
  double q_sigma_a = sigma_a0 + 0.5 * N;
  double q_sigma_b = sigma_b0;
  double q_lambda_a = 0.5*(sigma_b0 + 1);
  double q_lambda_b = (sigma_b0 * ig_E_inv(q_sigma_a, q_sigma_b) + pow(sigma_a0, -2));
  if(pr_sigma == 2) {
    q_sigma_a = 0.5 * (sigma_b0 + N);
  }
  
  // Monitor
  bool converged = 0;
  int iterations = 0;
  arma::vec elbo(maxiter);
  arma::mat tr(P + Q, maxiter);
  
  for(int i = 0; i < maxiter && !converged; i++) {

    // Update parameters of q(beta,u)
    G_inv(0) = inv_beta_sigma0;
    G_inv(1) = kron(Im, E_q_omega_inv);
    bdG = blockDiag(G_inv);
    q_betagamma_sigma = inv(ig_E_inv(q_sigma_a, q_sigma_b) * CtC + bdG);
    q_betagamma_mu = q_betagamma_sigma * (ig_E_inv(q_sigma_a, q_sigma_b) * Cty + bdG*betagamma_mu0);
    E_dot_y_Cb = dot_y_minus_Xb(yty, Cty, CtC, q_betagamma_mu, q_betagamma_sigma);
    q_gamma_mu = q_betagamma_mu.subvec(P, P + Q - 1);
    q_gamma_sigma = q_betagamma_sigma.submat(P, P, P + Q - 1, P + Q - 1);

    // Update parameters of q(sigma)
    
    // Inverse-Gamma prior
    if(pr_sigma == 1) {
      q_sigma_b = sigma_b0 + 0.5*E_dot_y_Cb;
    // Half-t prior (hierarchical inverse gamma)
    } else if (pr_sigma == 2) {
      q_lambda_b = (sigma_b0 * ig_E_inv(q_sigma_a, q_sigma_b) + pow(sigma_a0, -2));
      q_sigma_b = sigma_b0 * ig_E_inv(q_lambda_a, q_lambda_b) + 0.5*E_dot_y_Cb;
    }

    // Update parameters of q(Omega)
    
    // Inverse-Wishart prior
    if(pr_Omega == 1) {
      q_omega_lambda = lambda_Omega0;
      for(int m = 0; m < M; m++) {
        q_omega_lambda += q_gamma_mu.subvec(idq(m), idq(m+1)-1) * 
          q_gamma_mu.subvec(idq(m), idq(m+1)-1).t() +
          q_gamma_sigma.submat(idq(m), idq(m), idq(m+1)-1, idq(m+1)-1);
      }
      E_q_omega_inv = inv_wishart_E_invX(q_omega_nu, q_omega_lambda);
      
    // Huang-Wand hierarchical prior
    } else if (pr_Omega == 2) {
      // for(int z = 0; z < q(0); z++) {
      //   q_xi_b(z) = nu_Omega0 * E_q_omega_inv(z, z) + pow(lambda_Omega0(z, z), -2);
      //   q_xi_a(z) = 0.5 * (nu_Omega0 + q(0)) / q_xi_b(z);
      //   E_ig_inv_xi(z) = ig_E_inv(q_xi_a(z),  q_xi_b(z));
      // }
      // q_omega_lambda = 2*nu_Omega0*arma::diagmat(E_ig_inv_xi);
      // for(int m = 0; m < M; m++) {
      //   q_omega_lambda += q_gamma_mu.subvec(idq(m), idq(m+1)-1) * 
      //     q_gamma_mu.subvec(idq(m), idq(m+1)-1).t() +
      //     q_gamma_sigma.submat(idq(m), idq(m), idq(m+1)-1, idq(m+1)-1);
      // }
      // E_q_omega_inv = inv_wishart_E_invX(q_omega_nu, q_omega_lambda);
    }
    
    // update ELBO to check convergence
    if(use_elbo) {
      
    }
    
    if(trace)
      tr.col(i) = q_betagamma_mu;
    iterations = i;
  }
  
  List out = List::create(
    Named("q_betagamma_mu") = q_betagamma_mu,
    Named("q_betagamma_sigma") = q_betagamma_sigma,
    Named("q_omega_lambda") = q_omega_lambda,
    Named("q_omega_nu") = q_omega_nu,
    Named("q_sigma_a") = q_sigma_a,
    Named("q_sigma_b") = q_sigma_b
  );
  
  if(trace) out.push_back(tr.submat(0, 0, P + Q - 1, iterations), "trace");
  
  return out;
}


//' Variational Bayes for linear mixed model (random intercept and coefficient only, streamlined).
//' 
//' Performs variational inference for random intercept and coefficient model.
//' Currently assumes that all groups have same number of parameters.
//' That is, that all Zlist elements are of equal dimension.
//' 
//' @param Xlist A list of subject specific design matrices
//' @param Zlist A list of subject specific group matrices
//' @param ylist A list of subject specific responses
//' @param beta_mu0 Prior mean for beta
//' @param beta_sigma0 Prior covariance for beta
//' @param nu_Omega0 Prior df for Omega
//' @param lambda_Omega0 Prior scale matrix for Omega
//' @param pr_Omega Prior type for Omega - 1 is IW(nu, lambda), 2 is HW(nu, 2*nu*diag(1/lambda^2))
//' @param sigma_a0 The first hyper-parameter for prior on sigma
//' @param sigma_b0 The second hyper-parameter for prior on sigma
//' @param pr_sigma The prior to use for sigma_epsilon - 1 is IG(a0,b0) and 2 is Half-t(a0, b0)
//' @param tol Tolerance level for assessing convergence
//' @param maxiter Maximum number of fixed-update iterations
//' @param verbose Print trace of ELBO
//' @param trace Return trace of parameters beta, gamma
//' @param use_elbo Should the ELBO be calculated and used for convergence checks?
//' 
//' @export
// [[Rcpp::export]]
List vb_lmm_randintslope_streamlined(
    const arma::field<arma::mat>& Xlist, 
    const arma::field<arma::mat>& Zlist,
    const arma::field<arma::vec>& ylist,
    const arma::vec& beta_mu0, 
    const arma::mat& beta_sigma0, 
    double nu_Omega0,
    const arma::mat lambda_Omega0,
    int pr_Omega = 1,
    double sigma_a0 = 1e-2,
    double sigma_b0 = 1e-2,
    int pr_sigma = 1,
    double tol = 1e-8, 
    int maxiter = 100,
    bool verbose = false,
    bool trace = false,
    bool use_elbo = true   
) {
  
  if(Zlist.n_rows != Xlist.n_rows || Zlist.n_rows != ylist.n_rows)
    Rcpp::stop("Dimension mismatch between Xlist, Zlist, ylist.");
  
  // input dimensions
  int M = Zlist.n_rows;
  int P = Xlist(0).n_cols;
  arma::vec q(M);
  arma::vec r(M);
  // rows and columns per subject
  arma::vec idq = arma::zeros(M+1);
  arma::vec idr = arma::zeros(M+1);
  arma::field<arma::mat> XtZ(M);
  arma::field<arma::mat> ZtZ(M);
  arma::field<arma::vec> Zty(M);
  arma::mat X;
  arma::vec y;
  
  for(int m = 0; m < M; m++) {
    // allow for different sized Z_i matrices
    q(m) = Zlist(m).n_cols;
    r(m) = Zlist(m).n_rows;
    idq(m+1) = idq(m) + q(m);
    idr(m+1) = idr(m) + r(m);
    XtZ(m) = Xlist(m).t() * Zlist(m);
    ZtZ(m) = Zlist(m).t() * Zlist(m);
    Zty(m) = Zlist(m).t() * ylist(m);
    X = arma::join_vert(X, Xlist(m));
    y = arma::join_vert(y, ylist(m));
  }
  int Q = sum(q);
  int N = sum(r);

  arma::mat XtX = trans(X)*X;
  double yty = dot(y, y);
  arma::field<arma::mat> G(M);
  arma::field<arma::mat> H(M);
  arma::mat S = arma::zeros(P, P);
  arma::vec s = arma::zeros(P);
  arma::vec Cmu(N);
  double trCtCSigma = 0.0;
  arma::mat inv_beta_sigma0 =  inv(beta_sigma0);
  double E_dot_y_Cb = 0.0;
  
  // beta and gamma
  arma::vec q_betagamma_mu(P + Q);
  arma::vec q_gamma_mu(M*Q);
  arma::vec q_beta_mu(P);
  arma::mat q_betagamma_sigma(P + Q, P + Q);
  arma::mat q_gamma_sigma(Q, Q);
  // Omega
  double q_omega_nu = nu_Omega0 + M;
  arma::mat q_omega_lambda = lambda_Omega0;
  arma::mat E_q_omega_inv = inv_wishart_E_invX(q_omega_nu, q_omega_lambda);
  // sigma
  double q_sigma_a = sigma_a0 + 0.5 * N;
  double q_sigma_b = sigma_b0;
  double q_lambda_a = 0.5*(sigma_b0 + 1);
  double q_lambda_b = (sigma_b0 * ig_E_inv(q_sigma_a, q_sigma_b) + pow(sigma_a0, -2));
  if(pr_sigma == 2) {
    q_sigma_a = 0.5 * (sigma_b0 + N);
  }
  
  // Monitor
  bool converged = 0;
  int iterations = 0;
  arma::vec elbo(maxiter);
  arma::mat tr(P + Q, maxiter);
  
  for(int i = 0; i < maxiter && !converged; i++) {
    
    // Update parameters of q(beta,u)
    S.zeros();
    s.zeros();
    trCtCSigma = 0.0;
    for(int m = 0; m < M; m++) {
      G(m) = ig_E_inv(q_sigma_a, q_sigma_b) * XtZ(m);
      H(m) = inv(ig_E_inv(q_sigma_a, q_sigma_b) * ZtZ(m) + E_q_omega_inv);
      S += G(m)*H(m)*G(m).t();
      s += G(m)*H(m)*Zty(m);
    }
    q_betagamma_sigma.submat(0, 0, P-1, P-1) = inv(ig_E_inv(q_sigma_a, q_sigma_b) * XtX + inv_beta_sigma0 - S);
    q_betagamma_mu.subvec(0, P-1) = ig_E_inv(q_sigma_a, q_sigma_b) * q_betagamma_sigma.submat(0, 0, P-1, P-1) * (X.t()*y + inv_beta_sigma0 * beta_mu0 - s);
    for(int m = 0; m < M; m++) {
      // bottom right block
      q_betagamma_sigma.submat(P + idq(m), P + idq(m), P + idq(m+1) - 1, P + idq(m+1) - 1) =
        H(m) + H(m) * G(m).t() * q_betagamma_sigma.submat(0, 0, P-1, P-1) * G(m) * H(m);
      // upper right block
      q_betagamma_sigma.submat(0, P + idq(m), P - 1, P + idq(m+1) - 1) =
        -q_betagamma_sigma.submat(0, 0, P-1, P-1) * G(m) * H(m);
      // bottom left block
      q_betagamma_sigma.submat(P + idq(m), 0, P + idq(m+1) - 1, P - 1) =
        q_betagamma_sigma.submat(0, P + idq(m), P - 1, P + idq(m+1) - 1).t();
      // update gamma_mu
      q_betagamma_mu.subvec(P + idq(m), P + idq(m+1) - 1) =
        H(m) * (ig_E_inv(q_sigma_a, q_sigma_b) * Zty(m) - G(m).t()*q_betagamma_mu.subvec(0, P-1));
      // E[||y-Cmu||]
      Cmu.subvec(idr(m), idr(m+1)-1) = Zlist(m)*q_betagamma_mu.subvec(P + idq(m), P + idq(m+1)-1);
      trCtCSigma += -2/ig_E_inv(q_sigma_a, q_sigma_b)*
      arma::trace(G(m)*H(m)*G(m).t()*q_betagamma_sigma.submat(0, 0, P-1, P-1)) +
      arma::trace(ZtZ(m)*q_betagamma_sigma.submat(P + idq(m), P + idq(m), P + idq(m+1)-1, P + idq(m+1)-1));
    }
    Cmu += X * q_betagamma_mu.subvec(0, P-1);
    trCtCSigma += arma::trace(XtX*q_betagamma_sigma.submat(0, 0, P-1, P-1));
    E_dot_y_Cb = arma::dot(y - Cmu, y - Cmu) + trCtCSigma;
    q_gamma_mu = q_betagamma_mu.subvec(P, P + Q - 1);
    q_gamma_sigma = q_betagamma_sigma.submat(P, P, P + Q - 1, P + Q - 1);
    
    // Update parameters of q(sigma)
    
    // Inverse-Gamma prior
    if(pr_sigma == 1) {
      q_sigma_b = sigma_b0 + 0.5*E_dot_y_Cb;
      // Half-t prior (hierarchical inverse gamma)
    } else if (pr_sigma == 2) {
      q_lambda_b = (sigma_b0 * ig_E_inv(q_sigma_a, q_sigma_b) + pow(sigma_a0, -2));
      q_sigma_b = sigma_b0 * ig_E_inv(q_lambda_a, q_lambda_b) + 0.5*E_dot_y_Cb;
    }
    
    // Update parameters of q(Omega)
    
    // Inverse-Wishart prior
    if(pr_Omega == 1) {
      q_omega_lambda = lambda_Omega0;
      for(int m = 0; m < M; m++) {
        q_omega_lambda += q_gamma_mu.subvec(idq(m), idq(m+1)-1) * 
          q_gamma_mu.subvec(idq(m), idq(m+1)-1).t() +
          q_gamma_sigma.submat(idq(m), idq(m), idq(m+1)-1, idq(m+1)-1);
      }
      E_q_omega_inv = inv_wishart_E_invX(q_omega_nu, q_omega_lambda);
    }
    
    // update ELBO to check convergence
    if(use_elbo) {
      
    }
    
    if(trace)
      tr.col(i) = q_betagamma_mu;
    iterations = i;
  }
  
  List out = List::create(
    Named("q_betagamma_mu") = q_betagamma_mu,
    Named("q_betagamma_sigma") = q_betagamma_sigma,
    Named("q_omega_lambda") = q_omega_lambda,
    Named("q_omega_nu") = q_omega_nu,
    Named("q_sigma_a") = q_sigma_a,
    Named("q_sigma_b") = q_sigma_b
  );
  
  if(trace) out.push_back(tr.submat(0, 0, P + Q - 1, iterations), "trace");
  
  return out;
}
