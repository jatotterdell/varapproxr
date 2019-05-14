// [[Rcpp::depends(RcppArmadillo)]]

#include <RcppArmadillo.h>
#include <Rmath.h>

using namespace Rcpp;

//' Return the entropy of multivariate normal density
//' 
//' @param S The covariate matrix
//' @return value The entropy
// [[Rcpp::export]]
double mvn_entropy(arma::mat& S) {
  int d = S.n_rows;
  return 0.5*d*(1 + log(2*M_PI)) + 0.5*real(log_det(S));
}

//' Return the entropy of inverse gamma density
//' 
//' @param shape
//' @param scale
// [[Rcpp::export]]
double ig_entropy(double shape, double scale) {
  return shape + log(scale) + lgamma(shape) - (1 + shape)*R::digamma(shape);
}

//' Perform mean-field variational inference for 
//' basic linear regression model.
//' 
//' y | beta, sigma ~ N(X*beta, sigma^2*I)
//' beta ~ N(mu0, Sigma0)
//' sigma^2 | a ~ IG(1/2, 1/a)
//' a ~ IG(1/2, 1/A^2)
//' 
//' @param X The design matrix
//' @param y The response vector
//' @param mu0 The prior mean for beta
//' @param Sigma0 The prior covariance for beta
//' @return v A list of relevant outputs
//' @export
// [[Rcpp::export]]
List lin_reg(
    const arma::mat& X, 
    const arma::vec& y,
    const double sigmasq0,
    const double A,
    const double B,
    double tol = 1e-8, int maxiter = 100) {
  
  int N = X.n_rows;
  int P = X.n_cols;
  bool converged = 0;
  int iterations = 0;

  arma::vec elbo(maxiter);
  
  arma::mat I = arma::eye<arma::mat>(P, P);
  arma::mat XtX = trans(X) * X;
  arma::vec Xty = trans(X) * y;
  double yty = dot(y, y);
  double trXtX = trace(XtX);
  
  arma::mat mu;
  arma::mat Sigma;
  double alpha = A + N / 2;
  double beta = 1.;
  
  for(int i = 0; i < maxiter && !converged; i++) {
    // Update variational parameters
    Sigma = inv(alpha / beta * XtX + (1 / sigmasq0)*I);
    mu = alpha / beta * Sigma * Xty;
    beta = B + 0.5*(dot(y - X*mu, y - X*mu) + trace(Sigma * XtX));
    
    // Update ELBO
    elbo[i] = -N/2*(log(2*M_PI*beta) - R::digamma(alpha)) - 
      0.5*alpha/beta * (dot(y - X*mu, y - X*mu) + trace(Sigma*XtX)) -
      log(2*M_PI*sigmasq0) - (dot(mu, mu) + trace(Sigma)) / (2 * sigmasq0) +
      A*log(B) - lgamma(A) - (A + 1)*(log(beta) - R::digamma(alpha)) - B*alpha/beta +
      mvn_entropy(Sigma) + ig_entropy(alpha, beta);

    // Check for convergence
    if(i > 0 && fabs(elbo(i) - elbo(i - 1)) < tol) {
      converged = 1;
      iterations = i;
    }
  }

  return List::create(Named("converged") = converged,
                      Named("elbo") = elbo.subvec(0, iterations),
                      Named("mu") = mu,
                      Named("Sigma") = Sigma,
                      Named("alpha") = alpha,
                      Named("beta") = beta);
}