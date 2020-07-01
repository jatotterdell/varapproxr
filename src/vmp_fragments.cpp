#include <RcppArmadillo.h>
#include <Rmath.h>


//' Calculate vec^(-1)
//' 
//' @param v A vector of dimension d^2 by 1
// [[Rcpp::export]]
arma::mat inv_vectorise(arma::vec v) {
  int d = sqrt(v.n_elem);
  arma::mat V = arma::reshape(v, d, d);
  return(V);
}


//' Calculate
//' 
//' @param v1 A d x 1 vector
//' @param v2 A d*d x 1 vector
//' @param Q A d x d matrix
//' @param r A d x 1 vector
//' @param s A double
// [[Rcpp::export]]
arma::vec G_VMP(
  arma::vec v,
  arma::mat Q,
  arma::vec r,
  double s
) {
  int d = Q.n_cols;
  arma::vec v1 = v.subvec(0, d - 1);
  arma::vec v2 = v.subvec(d, d + d*d - 1);
  arma::mat V = inv(inv_vectorise(v2));
  arma::vec out = -1/8*arma::trace(
    Q*V*(v1*trans(v1)*V - 2.0*arma::eye(d,d))) +
      -0.5*trans(r)*V*v1-0.5*s;
  return(out);
}


//' Gaussian prior fragment update
//' 
//' @param mu The prior mean
//' @param Sigma The prior variance
// [[Rcpp::export]]
arma::vec GaussianPriorFragment(
  arma::vec mu,
  arma::mat Sigma
) {
  arma::mat invSigma = inv(Sigma);
  arma::vec eta1 = invSigma * mu;
  arma::vec eta2 = -0.5*arma::vectorise(invSigma);
  arma::vec eta = arma::join_vert(eta1, eta2);
  return(eta);
}


//' Inverse-Gamma prior fragment update
//' 
//' @param a Shape parameter
//' @param b Scale parameter
// [[Rcpp::export]]
arma::vec InverseGammaPriorFragment(
    double a,
    double b
) {
  arma::vec eta(2);
  eta(0) = -a - 1;
  eta(1) = b;
  return(eta);
}


//' Inverse Wishart prior fragment update
//' 
//' @param xi The prior shape
//' @param Lambda The prior symmetric positive definite matrix
// [[Rcpp::export]]
arma::vec InverseWishartPriorFragment(
  double xi,
  arma::mat Lambda
) {
  int d = Lambda.n_cols;
  arma::vec eta(d + 2);
  eta(0) = -0.5*(xi + d + 1);
  eta.subvec(1, d) = -0.5*arma::vectorise(Lambda);
  return(eta);  
}


//' Inverse G-Wishart prior fragment update
//' 
//' @param G The graph matrix
//' @param xi The prior shape
//' @param Lambda The prior symmetric positive definite matrix
// [[Rcpp::export]]
arma::field<arma::vec> InverseGWishartPriorFragment(
  arma::mat G,
  arma::vec xi,
  arma::mat Lambda
) {
  arma::field<arma::vec> eta(2);
  eta(0) = G;
  eta(1) = arma::join_vert(-0.5*(xi + 2), -0.5*arma::vectorise(Lambda));
  return(eta);
}




//' Iterated Inverse G-Wishart fragment update
//' 
//' @param G The graph matrix.
//' @param xi The prior shape \eqn{\xi \in \mathbb{R}}.
//' @param eta1_in A d + 1 x 1 vector.
//' @param eta2_in A d + 1 x 1 vector.
// [[Rcpp::export]]
arma::field<arma::vec> IteratedInverseGWishartFragment(
  arma::mat G,
  arma::vec xi,
  arma::vec eta1_in,
  arma::vec eta2_in
) {
  int d = eta1_in.n_elem - 1;
  double eta11 = eta1_in(0);
  double eta21 = eta2_in(0);
  arma::vec eta12 = eta1_in.subvec(1, d);
  arma::vec eta22 = eta2_in.subvec(1, d);
  double omega;
  if(G.is_diagmat()) {
    omega = 1.0;
  } else {
    omega = (d + 1.0) / 2.0;
  }
  arma::mat Omega1 = (eta11 + omega) * inv(inv_vectorise(eta12));
  arma::mat Omega2 = (eta21 + omega) * inv(inv_vectorise(eta22));
  arma::vec eta1 = arma::join_vert(-0.5*(xi + 2), -0.5 * vectorise(Omega2));
  arma::vec eta2 = arma::join_vert(-0.5*(xi + 2 - 2*omega), -0.5*vectorise(Omega1));
  arma::field<arma::vec> eta(2);
  eta(0) = eta1;
  eta(1) = eta2;
  return(eta);
}


//' Gaussian likelihood fragment update
//' 
// [[Rcpp::export]]
arma::field<arma::vec> GaussianLikelihoodFragment(
  arma::vec n,
  arma::mat XtX,
  arma::vec Xty,
  double yty,
  arma::vec eta1_in,
  arma::vec eta2_in
) {
  double eta2_in1 = eta2_in(0);
  double eta2_in2 = eta2_in(1);
  arma::vec eta1 = (eta2_in1 + 1) / eta2_in2 *
    arma::join_vert(Xty, -0.5*arma::vectorise(XtX));
  arma::vec eta2 = arma::join_vert(-n/2, G_VMP(eta1_in, XtX, Xty, yty));
  arma::field<arma::vec> eta(2);
  eta(0) = eta1;
  eta(1) = eta2;
  return(eta);
}


//' Gaussian transform natural parameters to E[T(X)] parameters.
//' 
//' @param eta The natural parameter vector.
//' 
// [[Rcpp::export]]
arma::field<arma::vec> ExpectationGaussianSufficientStatistics(
  arma::vec eta
) {
  int n = eta.n_elem;
  int d = (sqrt(4*n + 1) - 1) / 2;
  arma::vec eta1 = eta.subvec(0, d - 1);
  arma::vec eta2 = eta.subvec(d, d + d*d - 1);
  arma::mat tmp  = inv(inv_vectorise(eta2));
  arma::vec mu = -0.5*tmp*eta1;
  arma::vec Omega = -0.5*tmp + mu * trans(mu);
  arma::field<arma::vec> out(2);
  out(0) = mu;
  out(1) = vectorise(Omega);
  return(out);
}

//' Gaussian transform natural parameters to E[T(X)] parameters.
//' 
//' @param eta The natural parameter vector.
//' 
// [[Rcpp::export]]
arma::field<arma::mat> GaussianCommonParameters(
    arma::vec& eta
) {
  int n = eta.n_elem;
  int d = (sqrt(4*n + 1) - 1) / 2;
  arma::vec eta1 = eta.subvec(0, d - 1);
  arma::vec eta2 = eta.subvec(d, d + d*d - 1);
  arma::mat tmp  = inv(inv_vectorise(eta2));
  arma::vec mu = -0.5*tmp*eta1;
  arma::mat Sigma = -0.5*tmp;
  arma::field<arma::mat> out(2);
  out(0) = mu;
  out(1) = Sigma;
  return(out);
}

//' Calculate entropy for Gaussian density
//' 
//' @param eta Natural parameter
// [[Rcpp::export]]
double GaussianEntropy(
  arma::vec& eta
) {
  arma::field<arma::mat> theta = GaussianCommonParameters(eta);
  int d = theta(1).n_rows;
  return 0.5*(d*(1 + log(2*M_PI)) + real(log_det(theta(1))));
}


//' Inverse-G-Wishart transform natural parameters to E[T(X)] parameters.
//' 
//' @param G The graph.
//' @param eta The natural parameter vector.
//' 
// [[Rcpp::export]]
arma::field<arma::mat> ExpectationInverseGWishartSufficientStatistics(
    arma::mat G,
    arma::vec eta
) {
  int d = G.n_cols;
  double eta1 = eta(0);
  arma::vec eta2 = eta.subvec(1, d*d);
  double omega = 0.0;
  arma::mat Omega;
  arma::mat tmp = inv(inv_vectorise(eta2));
  if(G.is_diagmat()) {
    omega = real(arma::log_det(-tmp));
    for(int i = 1; i <= d; i++)
      omega -= R::digamma(-0.5*(eta1 + d - i));
    Omega = (eta + 1)*tmp;
  } else {
    omega = real(arma::log_det(-tmp)) - d*R::digamma(eta1/2);
    Omega = (eta + 0.5*(d + 1))*tmp;
  }
  arma::field<arma::mat> out;
  out(0) = omega;
  out(1) = Omega;
  return(out);
}


//' Inverse-G-Wishart Common parameters
//' 
//' @param eta The natural parameter vector.
//' 
// [[Rcpp::export]]
arma::field<arma::mat> InverseGWishartCommonParameters(
    arma::vec eta
) {
  double eta1 = eta(0);
  arma::vec eta2 = eta.subvec(1, eta.n_elem - 1);
  double xi = -2 - 2*eta1;
  arma::mat Lambda = -2*inv_vectorise(eta2);
  arma::field<arma::mat> out(2);
  out(0) = xi;
  out(1) = Lambda;
  return(out);
}


//' Variational Message Passing for Normal linear model.
//' 
//' @param X The design matrix n by d.
//' @param y The observation vector n by 1.
//' @param mu0 The prior mean on coefficients
//' @param Sigma0 The prior variance on coefficients
//' @param A The prior scale on variance
//' @param maxiter The maximum number of iterations
//' @param verbose Print trace of the ELBO
// [[Rcpp::export]]
Rcpp::List vmp_lm(
  arma::vec& n,
  arma::mat& X,
  arma::vec& y,
  arma::vec& mu0,
  arma::mat& Sigma0,
  double A,
  int maxiter = 1e2,
  double tol = 1e-10,
  bool verbose = true
) {
  int d = X.n_cols;
  
  arma::mat G          = diagmat(arma::ones(1.0));
  arma::vec xi         = arma::ones<arma::vec>(1.0);
  arma::mat Lambda     = diagmat(arma::ones(1.0)/pow(A,2));
  arma::vec eta_p_beta = GaussianPriorFragment(mu0, Sigma0);
  arma::vec eta_p_a    = InverseGWishartPriorFragment(G, xi, Lambda)(1);
  
  arma::field<arma::vec> eta_p_y(2);
  eta_p_y(0) = GaussianPriorFragment(arma::ones(d), arma::eye(d, d));
  eta_p_y(1) = {-2.0, -1.0};
  
  arma::field<arma::vec> eta_p_sigma(2);
  eta_p_sigma(0) = {-2.0, -1.0};
  eta_p_sigma(1) = {-2.0, -1.0};
  
  // Statistics
  arma::mat XtX = trans(X)*X;
  arma::vec Xty = trans(X)*y;
  double yty    = norm(y);
  
  // Natural variational parameters for densities
  arma::vec eta_beta  = eta_p_beta + eta_p_y(0);
  arma::vec eta_sigma = eta_p_sigma(0) + eta_p_y(1);
  arma::vec eta_a     = eta_p_a + eta_p_sigma(1);
  
  // Common parameters
  arma::field<arma::mat> theta_beta  = GaussianCommonParameters(eta_beta);
  arma::field<arma::mat> theta_sigma = InverseGWishartCommonParameters(eta_sigma);
  arma::field<arma::mat> theta_a     = InverseGWishartCommonParameters(eta_a);
  
  bool converged = 0;
  int iterations = 0;
  Rcpp::Rcout.precision(10);
  arma::vec elbo(maxiter);
  
  // Iterate updates until convergence
  for(int i = 0; i < maxiter && !converged; i++) {
    eta_p_y     = GaussianLikelihoodFragment(n, XtX, Xty, yty, eta_beta, eta_sigma);
    eta_p_sigma = IteratedInverseGWishartFragment(G, xi, eta_sigma, eta_a);
    eta_beta    = eta_p_beta + eta_p_y(0);
    eta_sigma   = eta_p_sigma(0) + eta_p_y(1);
    eta_a       = eta_p_a + eta_p_sigma(1);
    
    theta_beta  = GaussianCommonParameters(eta_beta);
    theta_sigma = InverseGWishartCommonParameters(eta_sigma);
    theta_a     = InverseGWishartCommonParameters(eta_a);
    
    Rcpp::Rcout << theta_beta(1) << std::endl;
    
    elbo(i)     = as_scalar(d/2 + 1/2*real(log_det(theta_beta(1))) + 
      -1/2*real(log_det(Sigma0)) -0.5*arma::trace(inv(Sigma0) * theta_beta(1)) +
      -1/2*trans(theta_beta(0) - mu0)*inv(theta_beta(1))*(theta_beta(0) - mu0) +
      +(n+2)*log(eta_sigma(1)/2) + 1 + log((n + 1)/2) -7/2*R::digamma(1) - (n+1)/2*log(2*M_PI) +
      -(n+3)/2*R::digamma((as_scalar(n) + 1)/2) +
      -(n+1)/(theta_sigma(1)*theta_a(1)) -4*log(theta_a(1)/2) +
      -(n+1)/(2*theta_sigma(1))*(arma::trace(XtX*theta_beta(1)) + norm(y - X*theta_beta(0))));
    
    // Check for convergence
    if(verbose)
      Rcpp::Rcout << "Iter: " << std::setw(3) << i << "; ELBO = " << std::fixed << elbo(i) << std::endl;
    if(i > 0 && fabs(elbo(i) / elbo(i - 1) - 1) < tol) {
      converged = 1;
    }
    iterations = i;
  }
  return(Rcpp::List::create(
    Rcpp::Named("eta_beta")  = eta_beta,
    Rcpp::Named("eta_sigma") = eta_sigma,
    Rcpp::Named("theta_beta") = theta_beta,
    Rcpp::Named("theta_sigma") = theta_sigma
  ));
}